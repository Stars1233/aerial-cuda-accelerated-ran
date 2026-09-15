/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "CLI/CLI.hpp" // CLI11 header
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "test_config.hpp"
#include "polar_decode_test_vec_gen.hpp"

#include "cuphy_internal.h"
#include "utils.cuh"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <limits>
#include <memory>
#include <vector>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

#include "polar_decoder/polar_cw_tree_layout.hpp" // codeword-tree buffer regions
#include "polar_decoder/polar_op_list_host.hpp" // shared host reference for the SC/fast-SSC operation lists

////////////////////////////////////////////////////////////////////////
// PolarBatchStats
// Accumulated correctness stats for one decoded generator batch.
struct PolarBatchStats
{
    uint64_t bitErrors   = 0; // payload bit errors vs source
    uint64_t infoBits    = 0; // total payload bits compared
    uint32_t blockErrors = 0; // codewords with any payload bit wrong (ground truth)
    uint32_t nBlocks     = 0; // codewords processed
    uint32_t crcFlagged  = 0; // codewords with decoder crcErrorFlag != 0
};

////////////////////////////////////////////////////////////////////////
// printPolarDeviceAndConfig()
// Print the GPU device line and a formatted polar run-configuration block,
// mirroring the cuphy_ex_ldpc report layout. When fromVector is true (file
// input) the rate-matching / code-rate / SNR fields come from the stored
// vector and are reported as such.
static void printPolarDeviceAndConfig(const char* source,
                                      uint16_t    A,
                                      uint32_t    E,
                                      uint16_t    N_cw,
                                      uint16_t    K_cw,
                                      uint8_t     nCrcBits,
                                      uint8_t     nCbs,
                                      int         listSz,
                                      uint32_t    nCws,
                                      double      snrDb,
                                      bool        rateMatched,
                                      bool        fromVector)
{
    printf("*********************************************************************\n");
    cuphy::device gpuDevice;
    printf("%s\n", gpuDevice.desc().c_str());
    printf("*********************************************************************\n");
    printf("Polar Decoder Configuration:\n");
    printf("*********************************************************************\n");
    printf("Source                           = %s\n", source);
    printf("A  (payload bits/segment, no CRC)= %u\n", static_cast<unsigned>(A));
    printf("Codeblocks per segment           = %u\n", static_cast<unsigned>(nCbs));
    printf("K  (info bits per codeblock)     = %u\n", static_cast<unsigned>(K_cw));
    printf("CRC bits                         = %u\n", static_cast<unsigned>(nCrcBits));
    printf("N  (mother code size)            = %u\n", static_cast<unsigned>(N_cw));
    if(fromVector)
    {
        printf("E  (rate-matched bits, nTxBits)  = (from input vector)\n");
        printf("R  (code rate)                   = (from input vector)\n");
    }
    else if(rateMatched)
    {
        printf("E  (rate-matched bits, nTxBits)  = %u\n", static_cast<unsigned>(E));
        printf("R  (code rate = A / E)           = %0.3f\n", (E > 0) ? static_cast<double>(A) / static_cast<double>(E) : 0.0);
    }
    else
    {
        // --no-ratematch: E is used only to derive N; the channel transmits the
        // raw mother-code bits (nCbs * N_cw), so report that effective length/rate.
        const uint32_t effE = static_cast<uint32_t>(nCbs) * N_cw;
        printf("E  (rate-matched bits, nTxBits)  = %u (effective %u; --no-ratematch transmits N)\n",
               static_cast<unsigned>(E), static_cast<unsigned>(effE));
        printf("R  (effective code rate)         = %0.3f\n", (effE > 0) ? static_cast<double>(A) / static_cast<double>(effE) : 0.0);
    }
    printf("List size (L)                    = %d\n", listSz);
    printf("Number of segments               = %u\n", static_cast<unsigned>(nCws));
    if(fromVector)
    {
        printf("Channel                          = (from input vector)\n");
        printf("SNR                              = (from input vector)\n");
    }
    else
    {
        printf("Channel                          = %s\n", rateMatched ? "rate-matched (QPSK + AWGN)" : "E == N (AWGN on coded bits)");
        printf("SNR                              = %.2f dB\n", snrDb);
    }
    printf("LLR data type                    = CUPHY_R_16F\n");
    printf("*********************************************************************\n");
}

////////////////////////////////////////////////////////////////////////
// printPolarTiming()
// LDPC-style timing line with human-readable throughput (info-bit rate and
// codeword rate) instead of a raw codewords/second figure.
static void printPolarTiming(unsigned int numRuns,
                             double       avgLatencyUs,
                             uint64_t     infoBitsPerLaunch,
                             uint32_t     cwsPerLaunch)
{
    const double secPerLaunch = avgLatencyUs * 1e-6;
    const double infoMbps     = (secPerLaunch > 0.0) ? (static_cast<double>(infoBitsPerLaunch) / secPerLaunch) / 1e6 : 0.0;
    const double cwMcps       = (secPerLaunch > 0.0) ? (static_cast<double>(cwsPerLaunch) / secPerLaunch) / 1e6 : 0.0;
    printf("Average (%u runs) elapsed time in usec = %.1f, throughput = %.2f Mbps (info bits), %.3f Mcodewords/s\n",
           numRuns, avgLatencyUs, infoMbps, cwMcps);
}

////////////////////////////////////////////////////////////////////////
// printPolarResults()
// LDPC-style BER/BLER summary plus the polar CRC-flag tally.
static void printPolarResults(uint64_t bitErrors,
                              uint64_t infoBits,
                              uint32_t blockErrors,
                              uint32_t nBlocks,
                              uint32_t crcFlagged)
{
    const double ber  = (infoBits > 0) ? static_cast<double>(bitErrors) / static_cast<double>(infoBits) : 0.0;
    const double bler = (nBlocks > 0) ? static_cast<double>(blockErrors) / static_cast<double>(nBlocks) : 0.0;
    printf("bit error count = %lu, bit error rate (BER) = (%lu / %lu) = %.5e, block error rate (BLER) = (%u / %u) = %.5e\n",
           static_cast<unsigned long>(bitErrors),
           static_cast<unsigned long>(bitErrors),
           static_cast<unsigned long>(infoBits),
           ber,
           static_cast<unsigned>(blockErrors),
           static_cast<unsigned>(nBlocks),
           bler);
    printf("CRC: %u of %u codewords flagged\n", static_cast<unsigned>(crcFlagged), static_cast<unsigned>(nBlocks));
}

////////////////////////////////////////////////////////////////////////
// runPolarDecoderTimed()
// Set up the polar decoder over the caller-prepared per-codeword buffers, launch
// it (with an optional warmup + timed repeat loop that prints the timing line),
// and destroy it. The caller owns the input/output buffers and reads back cbEsts
// afterward; this centralizes the decoder lifecycle shared by the generator and
// file-input paths.
static void runPolarDecoderTimed(uint16_t                        nPolCws,
                                 std::vector<__half*>&           cwTreeLLRsGpuAddrVec,
                                 cuphyPolarCwPrm_t*              pCwPrmsGpu,
                                 std::vector<cuphyPolarCwPrm_t>& cwPrmsCpuVec,
                                 std::vector<uint32_t*>&         cbEstsGpuAddrVec,
                                 std::vector<bool*>&             listPolScratchGpuAddrVec,
                                 uint8_t*                        pCrcErrorFlags,
                                 int                             polarListSz,
                                 unsigned int                    numRuns,
                                 bool                            doTiming,
                                 cudaStream_t                    strm)
{
    size_t        dynDescrSizeBytes, dynDescrAlignBytes;
    cuphyStatus_t statusGetWorkspaceSize = cuphyPolarDecoderGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
    if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrLLRAddrsBufCpu(sizeof(__half*) * nPolCws);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrCbAddrsBufCpu(sizeof(uint32_t*) * nPolCws);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrScratchAddrsBufCpu(sizeof(bool*) * nPolCws);

    cuphyPolarDecoderHndl_t polarDecoderHndl;
    cuphyStatus_t           statusCreate = cuphyCreatePolarDecoder(&polarDecoderHndl);
    if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

    cuphyPolarDecoderLaunchCfg_t polarDecoderLaunchCfg;
    const bool                   enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t                polarDecoderSetupStatus     = cuphySetupPolarDecoder(polarDecoderHndl,
                                                                  nPolCws,
                                                                  cwTreeLLRsGpuAddrVec.data(),
                                                                  pCwPrmsGpu,
                                                                  cwPrmsCpuVec.data(),
                                                                  cbEstsGpuAddrVec.data(),
                                                                  listPolScratchGpuAddrVec.data(),
                                                                  polarListSz,
                                                                  pCrcErrorFlags,
                                                                  static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                  dynDescrBufCpu.addr(),
                                                                  dynDescrBufGpu.addr(),
                                                                  dynDescrLLRAddrsBufCpu.addr(),
                                                                  dynDescrCbAddrsBufCpu.addr(),
                                                                  dynDescrScratchAddrsBufCpu.addr(),
                                                                  &polarDecoderLaunchCfg,
                                                                  strm);
    if(CUPHY_STATUS_SUCCESS != polarDecoderSetupStatus) throw cuphy::cuphy_exception(polarDecoderSetupStatus);

    cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, strm);
    cudaStreamSynchronize(strm);

    const CUDA_KERNEL_NODE_PARAMS& k = polarDecoderLaunchCfg.kernelNodeParamsDriver;
    auto launchPolarDecoder = [&]()
    {
        CUresult runStatus = cuLaunchKernel(k.func, k.gridDimX, k.gridDimY, k.gridDimZ,
                                            k.blockDimX, k.blockDimY, k.blockDimZ, k.sharedMemBytes,
                                            static_cast<CUstream>(strm), k.kernelParams, k.extra);
        if(CUDA_SUCCESS != runStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
    };

    if(doTiming && numRuns > 0)
    {
        launchPolarDecoder(); // warmup (not timed)
        cudaStreamSynchronize(strm);

        cuphy::event_timer tmr;
        tmr.record_begin(strm);
        for(unsigned int uRun = 0; uRun < numRuns; ++uRun) { launchPolarDecoder(); }
        tmr.record_end(strm);
        tmr.synchronize();
        cudaStreamSynchronize(strm);

        const double avgLatencyUs = (static_cast<double>(tmr.elapsed_time_ms()) * 1e3) / static_cast<double>(numRuns);
        uint64_t     infoBitsPerLaunch = 0;
        for(uint16_t cw = 0; cw < nPolCws; ++cw) { infoBitsPerLaunch += cwPrmsCpuVec[cw].A_cw; }
        printPolarTiming(numRuns, avgLatencyUs, infoBitsPerLaunch, nPolCws);
    }
    else
    {
        launchPolarDecoder();
        cudaStreamSynchronize(strm);
    }

    cuphyStatus_t statusDestroy = cuphyDestroyPolarDecoder(polarDecoderHndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);
}

////////////////////////////////////////////////////////////////////////
// decodeGeneratedBatch()
// Set up GPU buffers for a generated batch, run the polar decoder (optionally
// with a timed warmup + repeat loop), and compare the decoded payloads against
// the generator's source bits. All GPU allocations are scoped to a local
// linear_alloc and freed on return, so this is safe to call in a long loop.
static PolarBatchStats decodeGeneratedBatch(const PolarGenResult& gen,
                                            int                   polarListSz,
                                            unsigned int          numRuns,
                                            bool                  doTiming,
                                            cudaStream_t          strm)
{
    PolarBatchStats stats;

    const uint16_t nPolUciSegs = static_cast<uint16_t>(gen.segPrms.size());
    if(nPolUciSegs == 0)
    {
        return stats;
    }
    uint16_t totalNPolCws = 0; // sum of codeblocks over segments (2 per segment when segmented)
    for(const auto& segPrm : gen.segPrms)
    {
        totalNPolCws += segPrm.nCbs;
    }
    const cuphyPolarUciSegPrm_t* const pSeg = gen.segPrms.data();

    //-----------------------------------------------------------------
    // Reserve exact GPU workspace (linear_alloc aligns each request to 128 B).
    size_t max_mem = 0;
    auto   reserveAllocation = [&max_mem](size_t nBytes)
    {
        constexpr size_t alignment = 128;
        max_mem += (nBytes + alignment - 1) & ~(alignment - 1);
    };
    for(const auto& segPrm : gen.segPrms)
    {
        reserveAllocation(cuphy::polar::PolarCwTreeLayout::sizeBytes(segPrm.N_cw)); // matches the cwTree alloc below
        for(uint8_t cb = 0; cb < segPrm.nCbs; ++cb)
        {
            reserveAllocation(sizeof(__half) * 2 * segPrm.N_cw * polarListSz);
        }
    }
    reserveAllocation(totalNPolCws);
    reserveAllocation(totalNPolCws);
    reserveAllocation(totalNPolCws);
    for(const auto& segPrm : gen.segPrms)
    {
        const uint32_t nDecodedCbWords = div_round_up(static_cast<uint32_t>(segPrm.K_cw - segPrm.nCrcBits), static_cast<uint32_t>(32));
        for(uint8_t cb = 0; cb < segPrm.nCbs; ++cb)
        {
            reserveAllocation(nDecodedCbWords * sizeof(uint32_t));
        }
        if(segPrm.nCbs == 2)
        {
            const uint32_t nUciSegBits = segPrm.nCbs * (segPrm.K_cw - segPrm.nCrcBits) - segPrm.zeroInsertFlag;
            reserveAllocation(div_round_up(nUciSegBits, static_cast<uint32_t>(32)) * sizeof(uint32_t));
        }
    }
    reserveAllocation(totalNPolCws * sizeof(cuphyPolarCwPrm_t));
    if(polarListSz > 1)
    {
        for(const auto& segPrm : gen.segPrms)
        {
            for(uint8_t cb = 0; cb < segPrm.nCbs; ++cb)
            {
                reserveAllocation(sizeof(bool) * 2 * segPrm.N_cw * polarListSz);
            }
        }
    }

    cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

    //-----------------------------------------------------------------
    // Upload tree types (4*N_cw layout from the generator: tree types followed
    // by the SC and fast-SSC operation lists) and LLRs.
    uint16_t              nPolCws = 0;
    std::vector<__half*>  cwTreeLLRsGpuAddrVec(totalNPolCws);
    std::vector<uint8_t*> cwTreeTypesGpuAddrVec(nPolUciSegs);
    for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
    {
        const uint16_t N_cw             = pSeg[segIdx].N_cw;
        const uint8_t  nCbs             = pSeg[segIdx].nCbs;
        const size_t   nBytesCwTree     = cuphy::polar::PolarCwTreeLayout::sizeBytes(N_cw);
        const size_t   nBytesCwLLRs     = sizeof(__half) * N_cw;
        const size_t   nBytesCwTreeLLRs = sizeof(__half) * (2 * N_cw) * polarListSz;

        cwTreeTypesGpuAddrVec[segIdx] = static_cast<uint8_t*>(linearAlloc.alloc(nBytesCwTree));
        cudaMemcpyAsync(cwTreeTypesGpuAddrVec[segIdx], gen.treeTypes4N[segIdx].data(), nBytesCwTree, cudaMemcpyHostToDevice, strm);

        for(int i = 0; i < nCbs; ++i, ++nPolCws)
        {
            cwTreeLLRsGpuAddrVec[nPolCws] = static_cast<__half*>(linearAlloc.alloc(nBytesCwTreeLLRs));
            cudaMemcpyAsync(cwTreeLLRsGpuAddrVec[nPolCws] + N_cw, gen.llrN[nPolCws].data(), nBytesCwLLRs, cudaMemcpyHostToDevice, strm);
        }
    }
    cudaStreamSynchronize(strm);

    //-----------------------------------------------------------------
    // Output buffers.
    uint8_t* pCrcErrorFlags    = static_cast<uint8_t*>(linearAlloc.alloc(nPolCws));
    uint8_t* pCrcStatusBuffer  = static_cast<uint8_t*>(linearAlloc.alloc(nPolCws));
    uint8_t* pCrcStatus1Buffer = static_cast<uint8_t*>(linearAlloc.alloc(nPolCws));
    CU_CHECK(cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(pCrcErrorFlags), 0, nPolCws, static_cast<CUstream>(strm)));

    std::vector<uint32_t*> cbEstsGpuAddrVec(nPolCws);
    std::vector<uint32_t*> uciSegEstsGpuAddrVec(nPolUciSegs);
    uint32_t               cbIdx = 0;
    for(int uciSegIdx = 0; uciSegIdx < nPolUciSegs; ++uciSegIdx)
    {
        const uint32_t nUciSegBits   = pSeg[uciSegIdx].nCbs * (pSeg[uciSegIdx].K_cw - pSeg[uciSegIdx].nCrcBits) - pSeg[uciSegIdx].zeroInsertFlag;
        const uint32_t nUciSegWords  = div_round_up(nUciSegBits, static_cast<uint32_t>(32));
        const uint32_t nDecodedCbBits  = pSeg[uciSegIdx].K_cw - pSeg[uciSegIdx].nCrcBits;
        const uint32_t nDecodedCbWords = div_round_up(nDecodedCbBits, static_cast<uint32_t>(32));

        for(int cb = 0; cb < pSeg[uciSegIdx].nCbs; ++cb)
        {
            cbEstsGpuAddrVec[cbIdx] = static_cast<uint32_t*>(linearAlloc.alloc(nDecodedCbWords * sizeof(uint32_t)));
            cbIdx++;
        }
        if(pSeg[uciSegIdx].nCbs == 1)
        {
            uciSegEstsGpuAddrVec[uciSegIdx] = cbEstsGpuAddrVec[cbIdx - 1];
        }
        else
        {
            uciSegEstsGpuAddrVec[uciSegIdx] = static_cast<uint32_t*>(linearAlloc.alloc(nUciSegWords * sizeof(uint32_t)));
        }
    }

    //-----------------------------------------------------------------
    // Codeword parameters.
    std::vector<cuphyPolarCwPrm_t> cwPrmsCpuVec(nPolCws);
    uint16_t                       cwIdx = 0;
    for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
    {
        for(int i = 0; i < pSeg[segIdx].nCbs; ++i)
        {
            cwPrmsCpuVec[cwIdx].N_cw         = pSeg[segIdx].N_cw;
            cwPrmsCpuVec[cwIdx].nCrcBits     = pSeg[segIdx].nCrcBits;
            cwPrmsCpuVec[cwIdx].A_cw         = pSeg[segIdx].K_cw - cwPrmsCpuVec[cwIdx].nCrcBits;
            cwPrmsCpuVec[cwIdx].pCwTreeTypes = cwTreeTypesGpuAddrVec[segIdx];
            cwPrmsCpuVec[cwIdx].pCbEst       = cbEstsGpuAddrVec[cwIdx];
            cwPrmsCpuVec[cwIdx].pCrcStatus   = pCrcStatusBuffer + cwIdx;
            cwPrmsCpuVec[cwIdx].pCrcStatus1  = pCrcStatus1Buffer + cwIdx;
            cwPrmsCpuVec[cwIdx].nCbsInUciSeg      = pSeg[segIdx].nCbs;
            cwPrmsCpuVec[cwIdx].cbIdxWithinUciSeg = i;
            cwPrmsCpuVec[cwIdx].zeroInsertFlag    = pSeg[segIdx].zeroInsertFlag;
            cwPrmsCpuVec[cwIdx].pUciSegEst        = uciSegEstsGpuAddrVec[segIdx];
            cwIdx += 1;
        }
    }

    cuphyPolarCwPrm_t* pCwPrmsGpu = static_cast<cuphyPolarCwPrm_t*>(linearAlloc.alloc(nPolCws * sizeof(cuphyPolarCwPrm_t)));
    cudaMemcpyAsync(pCwPrmsGpu, cwPrmsCpuVec.data(), nPolCws * sizeof(cuphyPolarCwPrm_t), cudaMemcpyHostToDevice, strm);
    cudaStreamSynchronize(strm);

    //-----------------------------------------------------------------
    // List-decoder scratch.
    std::vector<bool*> listPolScratchGpuAddrVec;
    if(polarListSz > 1)
    {
        listPolScratchGpuAddrVec.resize(nPolCws);
        for(int cb = 0; cb < nPolCws; ++cb)
        {
            const size_t nBytesScratch = sizeof(bool) * (2 * cwPrmsCpuVec[cb].N_cw) * polarListSz;
            listPolScratchGpuAddrVec[cb] = static_cast<bool*>(linearAlloc.alloc(nBytesScratch));
        }
    }

    runPolarDecoderTimed(nPolCws, cwTreeLLRsGpuAddrVec, pCwPrmsGpu, cwPrmsCpuVec, cbEstsGpuAddrVec,
                         listPolScratchGpuAddrVec, pCrcErrorFlags, polarListSz, numRuns, doTiming, strm);

    //-----------------------------------------------------------------
    // Compare decoded payloads to the source bits + tally CRC flags.
    std::vector<uint8_t> crcErrorFlags(nPolCws, 0);
    cudaMemcpy(crcErrorFlags.data(), pCrcErrorFlags, nPolCws, cudaMemcpyDeviceToHost);

    for(uint16_t cw = 0; cw < nPolCws; ++cw)
    {
        const uint16_t A        = cwPrmsCpuVec[cw].A_cw;
        const uint32_t nCbWords = div_round_up(static_cast<uint32_t>(A), static_cast<uint32_t>(32));

        std::vector<uint32_t> decodedWords(nCbWords, 0);
        cudaMemcpy(decodedWords.data(), cbEstsGpuAddrVec[cw], nCbWords * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        const std::vector<uint8_t>& srcBits     = gen.srcPayloadBits[cw];
        uint32_t                    cwBitErrors = 0;
        for(uint16_t j = 0; j < A; ++j)
        {
            const uint8_t decodedBit = static_cast<uint8_t>((decodedWords[j >> 5] >> (j & 31)) & 1u);
            if(decodedBit != srcBits[j])
            {
                ++cwBitErrors;
            }
        }
        stats.bitErrors += cwBitErrors;
        stats.infoBits  += A;
        if(cwBitErrors > 0)
        {
            ++stats.blockErrors;
        }
        if(crcErrorFlags[cw] != 0)
        {
            ++stats.crcFlagged;
        }
    }
    stats.nBlocks = nPolCws;

    return stats;
}

////////////////////////////////////////////////////////////////////////
// runPolarSweep()
// Sweep SNR from snrStart to snrStop (inclusive). At each point, generate and
// decode batches with an incrementing seed until at least minBlockErrors block
// errors accumulate or maxCws codewords have been processed. Prints one row per
// SNR point and optionally mirrors the rows to a CSV file.
static void runPolarSweep(const PolarGenParams& baseParams,
                          int                   polarListSz,
                          int                   minBlockErrors,
                          uint32_t              maxCws,
                          double                snrStart,
                          double                snrStop,
                          double                snrStep,
                          const std::string&    csvPath,
                          cudaStream_t          strm)
{
    // The de-rate-match kernel indexes the per-launch codeword array through
    // childCbIdxs (uint8_t without ENABLE_64C), capping a launch at 256 codewords.
    // A segment emits nCbsPerSeg codeblocks (2 when segmented), so the per-batch
    // segment count is bounded by kMaxCwsPerLaunch / nCbsPerSeg.
    const uint8_t nCbsPerSeg = (((baseParams.A >= 360u) && (baseParams.E >= 1088u)) || (baseParams.A >= 1013u)) ? 2u : 1u;
    constexpr uint32_t kMaxCwsPerLaunch = 256;
    const uint32_t     batchSize        = std::min<uint32_t>(maxCws, kMaxCwsPerLaunch / nCbsPerSeg);

    std::ofstream csv;
    if(!csvPath.empty())
    {
        csv.open(csvPath);
        if(!csv.is_open())
        {
            throw std::runtime_error("polar sweep: cannot open CSV output file: " + csvPath);
        }
        csv << "snr_db,n_cw,bit_errors,ber,block_errors,bler,crc_flagged_bler\n";
    }

    printf("# Polar BER/BLER sweep: A=%u E=%u listSz=%d %s, min_block_errors=%d, max_cw=%u\n",
           baseParams.A, baseParams.E, polarListSz, baseParams.fullChain ? "rate-matched" : "simplified (no rate-match)", minBlockErrors, maxCws);
    printf("%8s %10s %12s %13s %12s %13s %15s\n",
           "SNR_dB", "nCw", "bitErrors", "BER", "blockErrors", "BLER", "crcFlaggedBLER");
    fflush(stdout);

    for(double snr = snrStart; snr <= snrStop + 1e-9; snr += snrStep)
    {
        uint64_t bitErrors   = 0;
        uint64_t infoBits    = 0;
        uint32_t blockErrors = 0;
        uint32_t crcFlagged  = 0;
        uint64_t nCw         = 0;
        uint64_t seed        = baseParams.seed;

        while((blockErrors < static_cast<uint32_t>(minBlockErrors)) && (nCw < maxCws))
        {
            // maxCws / nCw are in codewords, but bp.nCws is a segment count, so
            // convert the remaining budget to segments (each emits nCbsPerSeg CBs).
            const uint32_t remainingSegs = (maxCws - static_cast<uint32_t>(nCw)) / nCbsPerSeg;
            if(remainingSegs == 0)
            {
                break;
            }
            const uint16_t thisBatch = static_cast<uint16_t>(std::min<uint32_t>(batchSize, remainingSegs));

            PolarGenParams bp = baseParams;
            bp.snrDb          = static_cast<float>(snr);
            bp.seed           = seed;
            bp.nCws           = thisBatch;
            bp.listSz         = polarListSz;

            PolarGenResult  gen = generatePolarTestVectors(bp, strm);
            PolarBatchStats st  = decodeGeneratedBatch(gen, polarListSz, 0, false, strm);

            bitErrors   += st.bitErrors;
            infoBits    += st.infoBits;
            blockErrors += st.blockErrors;
            crcFlagged  += st.crcFlagged;
            nCw         += st.nBlocks;
            ++seed;
        }

        const double ber            = (infoBits > 0) ? static_cast<double>(bitErrors) / static_cast<double>(infoBits) : 0.0;
        const double bler           = (nCw > 0) ? static_cast<double>(blockErrors) / static_cast<double>(nCw) : 0.0;
        const double crcFlaggedBler = (nCw > 0) ? static_cast<double>(crcFlagged) / static_cast<double>(nCw) : 0.0;

        printf("%8.2f %10llu %12llu %13.4e %12u %13.4e %15.4e\n",
               snr,
               static_cast<unsigned long long>(nCw),
               static_cast<unsigned long long>(bitErrors),
               ber,
               blockErrors,
               bler,
               crcFlaggedBler);
        fflush(stdout);

        if(csv.is_open())
        {
            csv << snr << "," << nCw << "," << bitErrors << "," << ber << "," << blockErrors << "," << bler << "," << crcFlaggedBler << "\n";
        }
    }

    if(csv.is_open())
    {
        csv.close();
    }
}

////////////////////////////////////////////////////////////////////////
// Help formatter that left-aligns every option name in a single column.
// The stock CLI11 formatter reserves a separate column for short names, which
// pushes long-only options (e.g. --G) to the right and out of line with the
// short options. Rendering all names in one left-aligned column keeps every
// option's help text starting at the same indent regardless of - vs -- naming.
namespace {
class LeftAlignedFormatter : public CLI::Formatter
{
public:
    LeftAlignedFormatter()
    {
        // Widen the footer wrap so example command lines are not broken mid-command.
        footer_paragraph_width(110);
    }

    std::string make_option(const CLI::Option* opt, bool is_positional) const override
    {
        std::stringstream out;
        const std::string left = "  " + make_option_name(opt, is_positional) + make_option_opts(opt);
        const std::string desc = make_option_desc(opt);
        out << std::setw(static_cast<int>(get_column_width())) << std::left << left;
        if(!desc.empty())
        {
            bool skipFirstLinePrefix = true;
            if(left.length() >= get_column_width())
            {
                out << '\n';
                skipFirstLinePrefix = false;
            }
            CLI::detail::streamOutAsParagraph(
                out, desc, get_right_column_width(), std::string(get_column_width(), ' '), skipFirstLinePrefix);
        }
        out << '\n';
        return out.str();
    }
};
} // namespace

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    int polarListSz = 1;
#if CUDA_VERSION >= 12040
    bool     useGreenCtxs   = false;
#endif
    uint32_t SMsPerGreenCtx = 0;
    // Initialized only after arguments parse and validate, so --help and usage
    // errors do not spin up (and tear down) the nvlog framework — which would
    // otherwise print its init/close lines around the help text.
    std::optional<cuphyNvlogFmtHelper> nvlog_fmt;
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments using CLI11
        CLI::App app{"Polar Decoder Example"};
        app.formatter(std::make_shared<LeftAlignedFormatter>());
        app.footer(
            "Examples:\n"
            "# decode a stored HDF5 vector (file input):\n"
            "$ cuphy_ex_polar_decoder -i vector.h5\n"
            "# generate + decode one batch (rate-matched channel), list-8, timed:\n"
            "$ cuphy_ex_polar_decoder -A 32 -E 100 -w 256 -S 20 -L 8 -r 50\n"
            "# simplified E=N channel (skip rate matching):\n"
            "$ cuphy_ex_polar_decoder -A 32 -E 100 -S 4 -L 8 --no-ratematch\n"
            "# BER/BLER-vs-SNR sweep to a CSV file:\n"
            "$ cuphy_ex_polar_decoder --sweep -A 32 -E 100 -L 8 --snr-start -3 --snr-stop 6 -e 200 -o sweep.csv");

        std::string    inputFilename;
        PolarGenParams genParams;
        unsigned int   numRuns        = 20;
        bool           noRateMatch    = false;
        bool           useSweep       = false;
        double         snrStart       = -2.0;
        double         snrStop        = 6.0;
        double         snrStep        = 1.0;
        int            minBlockErrors = 100;
        std::string    outputCsv;

        // ---- Input: choose the stored-vector path (-i) or the generator (-A, below) ----
        auto* optInput = app.add_option("-i", inputFilename,
            "Input HDF5 filename or multi-cell YAML file (file-based input)")
            ->group("Input");

        // ---- Generator: self-contained test-vector generation (alternative to -i) ----
        // The generator uses a host uplink polar encoder (not the DL-only cuphy
        // encoder), so N is bounded by the decoder (CUPHY_POLAR_DECODER_MAX_BITS =
        // 1024; 38.212 n_max = 10 for UCI). A in [12,1706]: A in [12,19] uses CRC6 +
        // 3 PC bits, A >= 20 uses CRC11, and (A >= 360 & E >= 1088) or A >= 1013
        // splits into two codeblocks. generatePolarTestVectors rejects A < 12 (not
        // polar-coded in NR UCI).
        //   - Rate-matched E: 1 .. MAX_TX_BITS.
        // The derived N and the K <= N invariant are enforced in generatePolarTestVectors.
        constexpr int kMinPayloadA = 12;                                       // NR UCI polar starts at A = 12
        constexpr int kMaxPayloadA = 1706;                                     // NR UCI polar max payload
        auto* optA = app.add_option("-A", genParams.A,
            "Payload bits per segment, 12.." + std::to_string(kMaxPayloadA) +
            " (does not include CRC; 1 or 2 codeblocks). Providing this selects the "
            "self-contained test-vector generator.")
            ->check(CLI::Range(kMinPayloadA, kMaxPayloadA))
            ->group("Generator");
        app.add_option("-E", genParams.E,
            "Number of rate-matched bits (nTxBits), 1.." + std::to_string(CUPHY_POLAR_ENC_MAX_TX_BITS) +
            " (default: 100)")
            ->check(CLI::Range(1, CUPHY_POLAR_ENC_MAX_TX_BITS))
            ->group("Generator");
        auto* optW = app.add_option("-w", genParams.nCws,
            "Number of segments per run (default: 16). Each launch is capped at 256 "
            "codewords, so single-run max -w is 256 (single-codeblock A) or 128 "
            "(2-codeblock A). In --sweep mode -w is the max codeword budget per SNR "
            "point (default: 20000), batched across launches within that cap")
            ->check(CLI::PositiveNumber)
            ->group("Generator");
        app.add_option("-S", genParams.snrDb,
            "Channel SNR in dB (default: 20.0)")
            ->group("Generator");
        app.add_option("--seed", genParams.seed,
            "Base RNG seed (default: 0)")
            ->group("Generator");
        app.add_flag("--no-ratematch", noRateMatch,
            "Skip rate matching: modulate the N mother-code bits directly and feed "
            "them to the decoder. E is still used to derive N but is otherwise "
            "ignored, so the effective transmitted length is N (not E). Default is "
            "the full rate-matched RX chain (modulate the E rate-matched bits, then "
            "de-rate-match/de-interleave).")
            ->group("Generator");

        // ---- Sweep: BER/BLER vs SNR (used with the generator) ----
        app.add_flag("--sweep", useSweep,
            "Run a BER/BLER-vs-SNR sweep (requires -A)")
            ->group("Sweep");
        app.add_option("--snr-start", snrStart, "First SNR in dB (default: -2.0)")
            ->group("Sweep");
        app.add_option("--snr-stop", snrStop, "Last SNR in dB, inclusive (default: 6.0)")
            ->group("Sweep");
        app.add_option("--snr-step", snrStep, "SNR step in dB, must be > 0 (default: 1.0)")
            ->check(CLI::PositiveNumber)
            ->group("Sweep");
        app.add_option("-e,--min-block-errors", minBlockErrors,
            "Accumulate until this many block errors per SNR point (default: 100)")
            ->check(CLI::PositiveNumber)
            ->group("Sweep");
        app.add_option("-o", outputCsv,
            "Also write the sweep rows to this CSV file (optional; the console table is always printed)")
            ->group("Sweep");

        // ---- Decoder & execution: apply to both the file and generator paths ----
        app.add_option("-L", polarListSz,
            "List size for polar decoder (1, 2, 4, or 8) (default: 1)")
            ->check(CLI::IsMember({1, 2, 4, 8}))
            ->group("Decoder & execution");
        app.add_option("-r", numRuns,
            "Number of timed decode launches (default: 20). One warmup launch is always run first and not counted.")
            ->check(CLI::PositiveNumber)
            ->group("Decoder & execution");
        auto* optG = app.add_option("--G", SMsPerGreenCtx,
            "Use green contexts with specified SM count per context (default: off, use all SMs)")
            ->check(CLI::PositiveNumber)
            ->group("Decoder & execution");

        CLI11_PARSE(app, argc, argv);

        // Rate matching is on by default; --no-ratematch selects the simplified E=N path.
        genParams.fullChain = !noRateMatch;

#if CUDA_VERSION >= 12040
        useGreenCtxs                = (optG->count() > 0);
#else
        // Green contexts require the CUDA 12.4 driver resource APIs. Fail loudly
        // instead of silently accepting and ignoring --G on older toolkits.
        if(optG->count() > 0)
        {
            std::cerr << "ERROR: --G (green contexts) requires CUDA >= 12.4; this build uses CUDA " << CUDA_VERSION << "." << std::endl;
            return 1;
        }
#endif
        const bool useFileInput     = (optInput->count() > 0);
        const bool useGenerator     = (!useFileInput) && (optA->count() > 0);

        // Usage errors are reported to stderr and exit before nvlog is initialized,
        // so they stay free of framework log noise (as does --help above).
        if(!useFileInput && !useGenerator)
        {
            std::cerr << "ERROR: provide either -i <file> (file input) or -A <payloadBits> (generator input)\n\n"
                      << app.help() << std::endl;
            return 1;
        }
        if(useSweep && !useGenerator)
        {
            std::cerr << "ERROR: --sweep requires generator input (-A <payloadBits>)" << std::endl;
            return 1;
        }
        if(useSweep && snrStop < snrStart)
        {
            std::cerr << "ERROR: --snr-stop (" << snrStop << ") must be >= --snr-start (" << snrStart << ")" << std::endl;
            return 1;
        }

        // Per-SNR-point MAX codeword budget: default 20000 when sweeping unless -w given.
        const uint32_t sweepMaxCws = (optW->count() > 0) ? static_cast<uint32_t>(genParams.nCws) : 20000u;

        // Arguments are valid — now bring up nvlog for the actual run.
        nvlog_fmt.emplace("polar_decoder.log");

        // ---------------------------------------------------------------
        // Select GPU device and (optionally) create a green context to limit
        // the number of SMs available to the polar decoder kernel. The green
        // context must be bound before the main stream is created so that all
        // subsequent work runs within it.

        int gpuId = 0; // select GPU device 0
        CUDA_CHECK(cudaSetDevice(gpuId)); // binds device 0's primary context for the cu* calls below
        CUdevice current_device;
        CU_CHECK(cuDeviceGet(&current_device, gpuId));

#if CUDA_VERSION >= 12040
        CUdevResource     initial_device_GPU_resources = {};
        CUdevResourceType default_resource_type        = CU_DEV_RESOURCE_TYPE_SM;
        CUdevResource     split_result[2]              = {{}, {}};
        cuphy::cudaGreenContext polar_green_ctx;
        unsigned int      split_groups                 = 1;

        if(useGreenCtxs)
        {
            // Best to ensure that MPS service is not running
            int mpsEnabled = 0;
            CU_CHECK(cuDeviceGetAttribute(&mpsEnabled, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, current_device));
            if(mpsEnabled == 1)
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "MPS is enabled. Heads-up that currently using green contexts with MPS enabled can have unintended side effects. Will run regardless.");
            }
            else
            {
                NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "MPS service is not running.");
            }

            // Check SMsPerGreenCtx value is in valid range
            int32_t gpuMaxSmCount = 0;
            CU_CHECK(cuDeviceGetAttribute(&gpuMaxSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, current_device));
            if(SMsPerGreenCtx > static_cast<uint32_t>(gpuMaxSmCount))
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "ERROR: Invalid --G argument {}. It is greater than {} (GPU's max SMs).", SMsPerGreenCtx, gpuMaxSmCount);
                throw std::runtime_error("invalid --G argument: requested SM count exceeds the device maximum");
            }

            CU_CHECK(cuDeviceGetDevResource(current_device, &initial_device_GPU_resources, default_resource_type));
            CU_CHECK(cuDevSmResourceSplitByCount(&split_result[0], &split_groups, &initial_device_GPU_resources, &split_result[1], 0, SMsPerGreenCtx));
            polar_green_ctx.create(gpuId, &split_result[0]);
            polar_green_ctx.bind();
            NVLOGC_FMT(NVLOG_PUSCH, "Polar decoder green context will have access to {} SMs ({} SMs requested).", polar_green_ctx.getSmCount(), SMsPerGreenCtx);
        }
#endif

        // ---------------------------------------------------------------
        // Initialize main stream

        cuphy::stream cuStrmMain;

        //------------------------------------------------------------------
        // Acquire input either from HDF5/YAML files or the self-contained
        // test-vector generator. Both paths populate polUciSegPrmsCpuVec and
        // the aggregate segment/codeword counts consumed by the decoder below.

        struct DatasetOffset
        {
            uint16_t seg;
            uint16_t cw;
        };
        std::vector<std::unique_ptr<UciPolarDataset>> uciPolarDatasets;
        std::vector<DatasetOffset>                    datasetOffsets;
        std::vector<cuphyPolarUciSegPrm_t>            polUciSegPrmsCpuVec;
        PolarGenResult                                genResult;
        uint32_t                                      nPolUciSegs32 = 0;
        uint32_t                                      nPolCws32     = 0;

        if(useFileInput)
        {
            // Resolve input files. A YAML input describes one slot with one HDF5
            // polar vector per cell; all vectors are aggregated into one launch.
            std::vector<std::string> inputFileNames;
            const size_t             extPos     = inputFilename.find_last_of('.');
            const std::string        inFileExtn = extPos == std::string::npos ? std::string() : inputFilename.substr(extPos + 1);
            if(inFileExtn == "yaml" || inFileExtn == "yml")
            {
                cuphy::test_config testCfg(inputFilename.c_str());
                const std::string  channelName = "POLAR";
                if(testCfg.num_slots() != 1)
                {
                    throw std::runtime_error("Polar decoder YAML input must contain exactly one slot");
                }

                const auto& slot = testCfg.slots()[0];
                const auto  it   = slot.find(channelName);
                if(it == slot.end())
                {
                    throw std::runtime_error("POLAR channel name not found in the input YAML file");
                }
                inputFileNames = it->second;
                if(inputFileNames.empty() || inputFileNames.size() != testCfg.num_cells())
                {
                    throw std::runtime_error("POLAR file count does not match YAML cell count");
                }
            }
            else
            {
                inputFileNames.emplace_back(inputFilename);
            }

            for(const auto& filename : inputFileNames)
            {
                auto dataset = std::make_unique<UciPolarDataset>(filename, cuStrmMain.handle());
                uint32_t datasetCwCount = 0;
                for(const auto& segPrm : dataset->polUciSegPrmsVec)
                {
                    datasetCwCount += segPrm.nCbs;
                }
                if(datasetCwCount != dataset->nPolCws)
                {
                    throw std::runtime_error("Polar HDF5 codeword count does not match its segment parameters");
                }
                datasetOffsets.push_back({static_cast<uint16_t>(nPolUciSegs32), static_cast<uint16_t>(nPolCws32)});
                nPolUciSegs32 += dataset->nPolUciSegs;
                nPolCws32 += dataset->nPolCws;
                if(nPolUciSegs32 > CUPHY_MAX_N_POL_UCI_SEGS || nPolCws32 > CUPHY_MAX_N_POL_CWS ||
                   nPolUciSegs32 > std::numeric_limits<uint16_t>::max() || nPolCws32 > std::numeric_limits<uint16_t>::max())
                {
                    throw std::runtime_error("Aggregated polar workload exceeds the cuPHY decoder limits");
                }
                polUciSegPrmsCpuVec.insert(polUciSegPrmsCpuVec.end(), dataset->polUciSegPrmsVec.begin(), dataset->polUciSegPrmsVec.end());
                uciPolarDatasets.emplace_back(std::move(dataset));
            }
            CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle())); // synchronize HDF5 tensor loading

            NVLOGC_FMT(NVLOG_PUSCH, "Loaded {} polar cell(s): {} UCI segment(s), {} codeword(s).", inputFileNames.size(), static_cast<uint16_t>(nPolUciSegs32), static_cast<uint16_t>(nPolCws32));

            const uint16_t    cfgN_cw   = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].N_cw;
            const uint16_t    cfgK_cw   = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].K_cw;
            const uint8_t     cfgnCrc   = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].nCrcBits;
            const uint8_t     cfgnCbs   = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].nCbs;
            const uint8_t     cfgZI     = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].zeroInsertFlag;
            // Segment payload: nCbs*(K - CRC) - zeroInsert (= A for both 1 and 2 CB).
            const uint16_t    cfgA      = (cfgK_cw >= cfgnCrc) ? static_cast<uint16_t>(cfgnCbs * (cfgK_cw - cfgnCrc) - cfgZI) : 0;
            const std::string cfgSource = (inputFileNames.size() == 1)
                                              ? inputFileNames.front()
                                              : (std::to_string(inputFileNames.size()) + " input file(s)");
            printPolarDeviceAndConfig(cfgSource.c_str(), cfgA, 0, cfgN_cw, cfgK_cw, cfgnCrc, cfgnCbs,
                                      polarListSz, static_cast<uint32_t>(nPolUciSegs32), 0.0,
                                      /*rateMatched=*/false, /*fromVector=*/true);
        }
        else if(!useSweep)
        {
            // Single-run generator: one UCI segment (1 or 2 codeblocks) per -w.
            genParams.listSz = polarListSz;
            genResult        = generatePolarTestVectors(genParams, cuStrmMain.handle());
            polUciSegPrmsCpuVec = genResult.segPrms;
            nPolUciSegs32       = static_cast<uint32_t>(polUciSegPrmsCpuVec.size());
            nPolCws32           = 0;
            for(const auto& segPrm : polUciSegPrmsCpuVec) { nPolCws32 += segPrm.nCbs; }
            if(nPolUciSegs32 > CUPHY_MAX_N_POL_UCI_SEGS || nPolCws32 > CUPHY_MAX_N_POL_CWS ||
               nPolUciSegs32 > std::numeric_limits<uint16_t>::max() || nPolCws32 > std::numeric_limits<uint16_t>::max())
            {
                throw std::runtime_error("Generated polar workload exceeds the cuPHY decoder limits");
            }
            const uint16_t cfgN_cw = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].N_cw;
            const uint16_t cfgK_cw = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].K_cw;
            const uint8_t  cfgnCrc = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].nCrcBits;
            const uint8_t  cfgnCbs = polUciSegPrmsCpuVec.empty() ? 0 : polUciSegPrmsCpuVec[0].nCbs;
            printPolarDeviceAndConfig("(generated at runtime)", genParams.A, genParams.E, cfgN_cw, cfgK_cw,
                                      cfgnCrc, cfgnCbs, polarListSz, genParams.nCws, static_cast<double>(genParams.snrDb),
                                      genParams.fullChain, /*fromVector=*/false);
        }

        //------------------------------------------------------------------
        // Generator dispatch (single-run or SNR sweep). The generator paths do
        // their own buffer setup / decode / compare in decodeGeneratedBatch, so
        // they return here; the file path continues to the block below.
        if(useGenerator)
        {
            if(useSweep)
            {
                runPolarSweep(genParams, polarListSz, minBlockErrors, sweepMaxCws,
                              snrStart, snrStop, snrStep, outputCsv, cuStrmMain.handle());
            }
            else
            {
                const PolarBatchStats st = decodeGeneratedBatch(genResult, polarListSz, numRuns, true, cuStrmMain.handle());
                printPolarResults(st.bitErrors, st.infoBits, st.blockErrors, st.nBlocks, st.crcFlagged);
            }
            return returnValue;
        }

        const uint16_t               nPolUciSegs         = static_cast<uint16_t>(nPolUciSegs32);
        const uint16_t               totalNPolCws        = static_cast<uint16_t>(nPolCws32);
        cuphyPolarUciSegPrm_t* const pPolarUciSegPrmsCpu = polUciSegPrmsCpuVec.data();

        //-----------------------------------------------------------------
        // Allocate exact GPU workspace. linear_alloc aligns every request to
        // 128 bytes, so include that padding in the reservation.

        size_t max_mem = 0;
        auto reserveAllocation = [&max_mem](size_t nBytes)
        {
            constexpr size_t alignment = 128;
            max_mem += (nBytes + alignment - 1) & ~(alignment - 1);
        };

        for(const auto& segPrm : polUciSegPrmsCpuVec)
        {
            reserveAllocation(cuphy::polar::PolarCwTreeLayout::sizeBytes(segPrm.N_cw));
            for(uint8_t cbIdx = 0; cbIdx < segPrm.nCbs; ++cbIdx)
            {
                reserveAllocation(sizeof(__half) * 2 * segPrm.N_cw * polarListSz);
            }
        }
        reserveAllocation(totalNPolCws); // CRC error flags
        reserveAllocation(totalNPolCws); // primary CRC status
        reserveAllocation(totalNPolCws); // secondary CRC status
        for(const auto& segPrm : polUciSegPrmsCpuVec)
        {
            const uint32_t nDecodedCbWords = div_round_up(static_cast<uint32_t>(segPrm.K_cw - segPrm.nCrcBits), static_cast<uint32_t>(32));
            for(uint8_t cbIdx = 0; cbIdx < segPrm.nCbs; ++cbIdx)
            {
                reserveAllocation(nDecodedCbWords * sizeof(uint32_t));
            }
            if(segPrm.nCbs == 2)
            {
                const uint32_t nUciSegBits = segPrm.nCbs * (segPrm.K_cw - segPrm.nCrcBits) - segPrm.zeroInsertFlag;
                reserveAllocation(div_round_up(nUciSegBits, static_cast<uint32_t>(32)) * sizeof(uint32_t));
            }
        }
        reserveAllocation(totalNPolCws * sizeof(cuphyPolarCwPrm_t));
        if(polarListSz > 1)
        {
            for(const auto& segPrm : polUciSegPrmsCpuVec)
            {
                for(uint8_t cbIdx = 0; cbIdx < segPrm.nCbs; ++cbIdx)
                {
                    reserveAllocation(sizeof(bool) * 2 * segPrm.N_cw * polarListSz);
                }
            }
        }

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //----------------------------------------------------------------------
        // GPU input buffers

        uint16_t              nPolCws = 0;
        std::vector<__half*>  cwTreeLLRsGpuAddrVec(totalNPolCws);
        std::vector<uint8_t*> cwTreeTypesGpuAddrVec(nPolUciSegs);

        // Host opList buffers must outlive the async H2D copies below (they are freed
        // only after the post-loop cudaStreamSynchronize). Reserve so the vector never
        // reallocates and the references handed to cudaMemcpyAsync stay valid.
        std::vector<std::vector<uint8_t>> hostOpLists;
        hostOpLists.reserve(2u * static_cast<size_t>(nPolUciSegs)); // SC + fast-SSC per seg

        if(useFileInput)
        {
            uint16_t globalSegIdx = 0;
            for(const auto& dataset : uciPolarDatasets)
            {
                uint16_t datasetCwIdx = 0;
                for(uint16_t segIdx = 0; segIdx < dataset->nPolUciSegs; ++segIdx, ++globalSegIdx)
                {
                    const uint16_t N_cw               = dataset->polUciSegPrmsVec[segIdx].N_cw;
                    const uint8_t  nCbs               = dataset->polUciSegPrmsVec[segIdx].nCbs;
                    const size_t   nBytesCwTreeTypes  = cuphy::polar::PolarCwTreeLayout::treeTypesBytes(N_cw);
                    const size_t   nBytesCwTree       = cuphy::polar::PolarCwTreeLayout::sizeBytes(N_cw);
                    const size_t   nBytesCwLLRs       = sizeof(__half) * N_cw;
                    const size_t   nBytesCwTreeLLRs   = sizeof(__half) * (2 * N_cw) * polarListSz;
                    const int      treeTypesIdxOffset = 2;

                    cwTreeTypesGpuAddrVec[globalSegIdx] = static_cast<uint8_t*>(linearAlloc.alloc(nBytesCwTree));
                    CUDA_CHECK(cudaMemcpyAsync(cwTreeTypesGpuAddrVec[globalSegIdx] + treeTypesIdxOffset, dataset->refCwTreeTypesVec[segIdx].addr(), nBytesCwTreeTypes - treeTypesIdxOffset * sizeof(uint8_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));

                    // Build the SC operation list from the reference tree types and
                    // place it behind them, mirroring the compCwTreeTypes kernel output.
                    std::vector<uint8_t> hostTreeTypes(cuphy::polar::PolarCwTreeLayout::treeTypesBytes(N_cw), 0);
                    std::memcpy(hostTreeTypes.data() + treeTypesIdxOffset,
                                dataset->refCwTreeTypesVec[segIdx].addr(),
                                nBytesCwTreeTypes - treeTypesIdxOffset * sizeof(uint8_t));
                    // Async on cuStrmMain (pipelined with the tree-types copy above) instead of
                    // the null stream, which would force a device-wide sync per segment. The host
                    // buffers are retained in hostOpLists until the post-loop stream sync.
                    const std::vector<uint8_t>& opList =
                        hostOpLists.emplace_back(cuphy::polar::buildPolarOpList(hostTreeTypes.data(), N_cw));
                    CUDA_CHECK(cudaMemcpyAsync(cwTreeTypesGpuAddrVec[globalSegIdx] + cuphy::polar::PolarCwTreeLayout::scOpListOffset(N_cw), opList.data(), opList.size(), cudaMemcpyHostToDevice, cuStrmMain.handle()));
                    const std::vector<uint8_t>& fssOpList =
                        hostOpLists.emplace_back(cuphy::polar::buildPolarFssOpList(hostTreeTypes.data(), N_cw));
                    CUDA_CHECK(cudaMemcpyAsync(cwTreeTypesGpuAddrVec[globalSegIdx] + cuphy::polar::PolarCwTreeLayout::fssOpListOffset(N_cw), fssOpList.data(), fssOpList.size(), cudaMemcpyHostToDevice, cuStrmMain.handle()));

                    for(int i = 0; i < nCbs; ++i, ++datasetCwIdx, ++nPolCws)
                    {
                        cwTreeLLRsGpuAddrVec[nPolCws] = static_cast<__half*>(linearAlloc.alloc(nBytesCwTreeLLRs));
                        // LLRs are copied only for the first decoder in the list.
                        CUDA_CHECK(cudaMemcpyAsync(cwTreeLLRsGpuAddrVec[nPolCws] + N_cw, dataset->refCwLLRsVec[datasetCwIdx].addr(), nBytesCwLLRs, cudaMemcpyHostToDevice, cuStrmMain.handle()));
                    }
                }
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        //----------------------------------------------------------------------
        // GPU output buffers

        uint8_t*               pCrcErrorFlags    = static_cast<uint8_t*>(linearAlloc.alloc(nPolCws));
        uint8_t*               pCrcStatusBuffer  = static_cast<uint8_t*>(linearAlloc.alloc(nPolCws));
        uint8_t*               pCrcStatus1Buffer = static_cast<uint8_t*>(linearAlloc.alloc(nPolCws));
        CU_CHECK(cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(pCrcErrorFlags), 0, nPolCws, static_cast<CUstream>(cuStrmMain.handle())));

        std::vector<uint32_t*> cbEstsGpuAddrVec(nPolCws);
        std::vector<uint32_t*> uciSegEstsGpuAddrVec(nPolUciSegs);

        uint32_t cbIdx = 0;
        for(int uciSegIdx = 0; uciSegIdx < nPolUciSegs; ++uciSegIdx)
        {
            uint32_t nUciSegBits  = pPolarUciSegPrmsCpu[uciSegIdx].nCbs * (pPolarUciSegPrmsCpu[uciSegIdx].K_cw - pPolarUciSegPrmsCpu[uciSegIdx].nCrcBits) - pPolarUciSegPrmsCpu[uciSegIdx].zeroInsertFlag;
            uint32_t nUciSegWords = div_round_up(nUciSegBits, static_cast<uint32_t>(32));

            uint32_t nDecodedCbBits  = pPolarUciSegPrmsCpu[uciSegIdx].K_cw - pPolarUciSegPrmsCpu[uciSegIdx].nCrcBits;
            uint32_t nDecodedCbWords = div_round_up(nDecodedCbBits, static_cast<uint32_t>(32));

            for(int cbIdxWithUciSeg = 0; cbIdxWithUciSeg < pPolarUciSegPrmsCpu[uciSegIdx].nCbs; ++cbIdxWithUciSeg)
            {
                cbEstsGpuAddrVec[cbIdx] = static_cast<uint32_t*>(linearAlloc.alloc(nDecodedCbWords * sizeof(uint32_t)));
                cbIdx++;
            }

            if(pPolarUciSegPrmsCpu[uciSegIdx].nCbs == 1)
            {
                uciSegEstsGpuAddrVec[uciSegIdx] = cbEstsGpuAddrVec[cbIdx - 1];
            }else
            {
                uciSegEstsGpuAddrVec[uciSegIdx] = static_cast<uint32_t*>(linearAlloc.alloc(nUciSegWords * sizeof(uint32_t)));
            }
        }

        // ----------------------------------------------------------------------
        // Codeword parameters

        std::vector<cuphyPolarCwPrm_t> cwPrmsCpuVec(nPolCws);
        uint16_t                       cwIdx = 0;

        for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
        {
            for(int i = 0; i < pPolarUciSegPrmsCpu[segIdx].nCbs; ++i)
            {
                cwPrmsCpuVec[cwIdx].N_cw         = pPolarUciSegPrmsCpu[segIdx].N_cw;
                cwPrmsCpuVec[cwIdx].nCrcBits     = pPolarUciSegPrmsCpu[segIdx].nCrcBits;
                cwPrmsCpuVec[cwIdx].A_cw         = pPolarUciSegPrmsCpu[segIdx].K_cw - cwPrmsCpuVec[cwIdx].nCrcBits;
                cwPrmsCpuVec[cwIdx].pCwTreeTypes = cwTreeTypesGpuAddrVec[segIdx];
                cwPrmsCpuVec[cwIdx].pCbEst       = cbEstsGpuAddrVec[cwIdx];
                cwPrmsCpuVec[cwIdx].pCrcStatus   = pCrcStatusBuffer + cwIdx;
                cwPrmsCpuVec[cwIdx].pCrcStatus1  = pCrcStatus1Buffer + cwIdx;

                cwPrmsCpuVec[cwIdx].nCbsInUciSeg       = pPolarUciSegPrmsCpu[segIdx].nCbs;
                cwPrmsCpuVec[cwIdx].cbIdxWithinUciSeg  = i;
                cwPrmsCpuVec[cwIdx].zeroInsertFlag     = pPolarUciSegPrmsCpu[segIdx].zeroInsertFlag;
                cwPrmsCpuVec[cwIdx].pUciSegEst         = uciSegEstsGpuAddrVec[segIdx];

                cwIdx += 1;
            }
        }

        cuphyPolarCwPrm_t* pCwPrmsGpu = static_cast<cuphyPolarCwPrm_t*>(linearAlloc.alloc(nPolCws * sizeof(cuphyPolarCwPrm_t)));
        cudaMemcpyAsync(pCwPrmsGpu, cwPrmsCpuVec.data(), nPolCws * sizeof(cuphyPolarCwPrm_t), cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle());

        //----------------------------------------------------------------------
        // GPU scratch buffers used in list decoder

        std::vector<bool*> listPolScratchGpuAddrVec;

        if (polarListSz > 1) {
            listPolScratchGpuAddrVec.resize(nPolCws);
            for(int cbIdx = 0; cbIdx < nPolCws; ++cbIdx)
            {
                size_t   nBytesScratch = sizeof(bool) * (2 * cwPrmsCpuVec[cbIdx].N_cw) * polarListSz;
                listPolScratchGpuAddrVec[cbIdx] = static_cast<bool*>(linearAlloc.alloc(nBytesScratch));
            }
        }

        runPolarDecoderTimed(nPolCws, cwTreeLLRsGpuAddrVec, pCwPrmsGpu, cwPrmsCpuVec, cbEstsGpuAddrVec,
                             listPolScratchGpuAddrVec, pCrcErrorFlags, polarListSz, numRuns, /*doTiming=*/true,
                             cuStrmMain.handle());

        // -------------------------------------------------------------------
        // Evaluate decoder output

        if(useFileInput)
        {
            for(size_t datasetIdx = 0; datasetIdx < uciPolarDatasets.size(); ++datasetIdx)
            {
                const auto& offset  = datasetOffsets[datasetIdx];
                const auto& dataset = uciPolarDatasets[datasetIdx];
                dataset->evalDecoderOutput(dataset->nPolCws,
                                           cwPrmsCpuVec.data() + offset.cw,
                                           cbEstsGpuAddrVec.data() + offset.cw,
                                           pCrcErrorFlags + offset.cw,
                                           dataset->nPolUciSegs,
                                           dataset->polUciSegPrmsVec.data(),
                                           uciSegEstsGpuAddrVec.data() + offset.seg,
                                           cuStrmMain.handle());
            }
        }
    }
    catch(std::exception& e)
    {
        if(nvlog_fmt) { NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what()); }
        else          { std::cerr << "EXCEPTION: " << e.what() << std::endl; }
        returnValue = 1;
    }
    catch(...)
    {
        if(nvlog_fmt) { NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION"); }
        else          { std::cerr << "UNKNOWN EXCEPTION" << std::endl; }
        returnValue = 2;
    }
    return returnValue;
}
