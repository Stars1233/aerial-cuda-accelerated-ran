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

#include "polar_decode_test_vec_gen.hpp"

#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "common_utils.hpp"
#include "polar_decoder/polar_cw_tree_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace
{
// Round a byte count up to the nearest multiple of 4 (32-bit alignment required
// by the polar encoder's packed-bit buffers).
inline uint32_t roundUp4(uint32_t nBytes)
{
    return ((nBytes + 3u) / 4u) * 4u;
}

// Derive the per-UCI-segment polar parameters, replicating
// PucchRx::expandUciCodingPrms. Handles both a single codeblock and 2-codeblock
// segmentation (38.212 6.3.1.2.1 / 5.2.1).
cuphyPolarUciSegPrm_t derivePolarSegPrm(uint16_t A, uint32_t E)
{
    cuphyPolarUciSegPrm_t seg{};

    const uint8_t nCrcBits = (A <= 19) ? 6 : 11;

    // Code-block segmentation: 2 codeblocks when (A >= 360 && E >= 1088) or
    // A >= 1013, else a single codeblock. Each CB carries ceil(A/2) info bits
    // (a zero is prepended to CB0 when A is odd, so both CBs are equal length).
    uint8_t  nCbs;
    uint32_t K_cw;
    uint32_t E_cw;
    uint8_t  zeroInsertFlag;
    if(((A >= 360) && (E >= 1088)) || (A >= 1013))
    {
        nCbs           = 2;
        K_cw           = static_cast<uint32_t>((A + 1u) / 2u) + nCrcBits; // ceil(A/2) + CRC
        E_cw           = E / 2u;                                          // floor(E/2)
        zeroInsertFlag = static_cast<uint8_t>(A & 1u);
    }
    else
    {
        nCbs           = 1;
        K_cw           = static_cast<uint32_t>(A) + nCrcBits;
        E_cw           = E;
        zeroInsertFlag = 0;
    }

    // Encoded CB size (38.212 5.3.1), from the per-CB E_cw and K_cw. n_temp =
    // ceil(log2(E_cw)) - 1; clamp at 0 so a tiny E_cw cannot underflow the shift.
    // Semantic validity (N in range, K <= N) is checked by the caller.
    const int32_t  n_temp_signed = static_cast<int32_t>(std::ceil(std::log2(static_cast<double>(E_cw)))) - 1;
    const uint32_t n_temp        = (n_temp_signed < 0) ? 0u : static_cast<uint32_t>(n_temp_signed);
    const uint32_t two_to_n_temp = 1u << n_temp;
    const uint32_t n_1           = ((8 * E_cw <= 9 * two_to_n_temp) && (16 * K_cw <= 9 * E_cw)) ? n_temp : n_temp + 1;
    const uint32_t n_2           = static_cast<uint32_t>(std::ceil(std::log2(static_cast<double>(K_cw * 8))));
    const uint32_t n_cw          = std::max(std::min(std::min(n_1, n_2), 10u), 5u);
    const uint32_t N_cw          = 1u << n_cw;

    seg.pUciSegLLRs    = nullptr;
    seg.exitFlag       = 0;
    seg.nCbs           = nCbs;
    seg.childCbIdxs[0] = 0;
    seg.childCbIdxs[1] = 0;
    seg.zeroInsertFlag = zeroInsertFlag;
    seg.E_seg          = E;
    seg.nCrcBits       = nCrcBits;
    seg.E_cw           = E_cw;
    seg.K_cw           = static_cast<uint16_t>(K_cw);
    seg.N_cw           = static_cast<uint16_t>(N_cw);
    seg.n_cw           = static_cast<uint8_t>(n_cw);
    return seg;
}

//----------------------------------------------------------------------------
// Host replica of the polar decoder's CRC check (validate_crc in
// polar_decoder.cu). The LUTs and byte-processing loops are copied verbatim so
// the generator can solve a real CRC field that makes the decoder's crcErrorFlag
// come out 0.

// CRC8 LUT based on polynomial = 388 (b110000100)  (copied from polar_decoder.cu)
const uint8_t CRC8_LUT[256] = {
    0,   132, 140, 8,   156, 24,  16,  148,
    188, 56,  48,  180, 32,  164, 172, 40,
    252, 120, 112, 244, 96,  228, 236, 104,
    64,  196, 204, 72,  220, 88,  80,  212,
    124, 248, 240, 116, 224, 100, 108, 232,
    192, 68,  76,  200, 92,  216, 208, 84,
    128, 4,   12,  136, 28,  152, 144, 20,
    60,  184, 176, 52,  160, 36,  44,  168,
    248, 124, 116, 240, 100, 224, 232, 108,
    68,  192, 200, 76,  216, 92,  84,  208,
    4,   128, 136, 12,  152, 28,  20,  144,
    184, 60,  52,  176, 36,  160, 168, 44,
    132, 0,   8,   140, 24,  156, 148, 16,
    56,  188, 180, 48,  164, 32,  40,  172,
    120, 252, 244, 112, 228, 96,  104, 236,
    196, 64,  72,  204, 88,  220, 212, 80,
    116, 240, 248, 124, 232, 108, 100, 224,
    200, 76,  68,  192, 84,  208, 216, 92,
    136, 12,  4,   128, 20,  144, 152, 28,
    52,  176, 184, 60,  168, 44,  36,  160,
    8,   140, 132, 0,   148, 16,  24,  156,
    180, 48,  56,  188, 40,  172, 164, 32,
    244, 112, 120, 252, 104, 236, 228, 96,
    72,  204, 196, 64,  212, 80,  88,  220,
    140, 8,   0,   132, 16,  148, 156, 24,
    48,  180, 188, 56,  172, 40,  32,  164,
    112, 244, 252, 120, 236, 104, 96,  228,
    204, 72,  64,  196, 80,  212, 220, 88,
    240, 116, 124, 248, 108, 232, 224, 100,
    76,  200, 192, 68,  208, 84,  92,  216,
    12,  136, 128, 4,   144, 20,  28,  152,
    176, 52,  60,  184, 44,  168, 160, 36
};

// CRC16 LUT based on polynomial = 50208 (b11000100 00100000)  (copied from polar_decoder.cu)
const uint16_t CRC16_LUT[256] = {
    0,     50208, 19552, 34880, 39104, 23776, 54432, 4224,
    62880, 12672, 47552, 32224, 28000, 43328, 8448,  58656,
    12128, 60224, 25344, 42784, 47008, 29568, 64448, 16352,
    56000, 7904,  38560, 21120, 16896, 34336, 3680,  51776,
    24256, 39648, 4768,  54912, 50688, 544,   35424, 20032,
    43872, 28480, 59136, 8992,  13216, 63360, 32704, 48096,
    29088, 46464, 15808, 63968, 59744, 11584, 42240, 24864,
    33792, 16416, 51296, 3136,  7360,  55520, 20640, 38016,
    48512, 31136, 61920, 13760, 9536,  57696, 26912, 44288,
    18464, 35840, 1088,  49248, 53472, 5312,  40064, 22688,
    37600, 22208, 56960, 6816,  2592,  52736, 17984, 33376,
    26432, 41824, 11040, 61184, 65408, 15264, 46048, 30656,
    58176, 10080, 44832, 27392, 31616, 49056, 14304, 62400,
    5856,  53952, 23168, 40608, 36384, 18944, 49728, 1632,
    52256, 2048,  32832, 17504, 21728, 37056, 6272,  56480,
    14720, 64928, 30176, 45504, 41280, 25952, 60704, 10496,
    48928, 31488, 62272, 14176, 10208, 58304, 27520, 44960,
    19072, 36512, 1760,  49856, 53824, 5728,  40480, 23040,
    36928, 21600, 56352, 6144,  2176,  52384, 17632, 32960,
    26080, 41408, 10624, 60832, 64800, 14592, 45376, 30048,
    57824, 9664,  44416, 27040, 31008, 48384, 13632, 61792,
    5184,  53344, 22560, 39936, 35968, 18592, 49376, 1216,
    52864, 2720,  33504, 18112, 22080, 37472, 6688,  56832,
    15136, 65280, 30528, 45920, 41952, 26560, 61312, 11168,
    672,   50816, 20160, 35552, 39520, 24128, 54784, 4640,
    63232, 13088, 47968, 32576, 28608, 44000, 9120,  59264,
    11712, 59872, 24992, 42368, 46336, 28960, 63840, 15680,
    55392, 7232,  37888, 20512, 16544, 33920, 3264,  51424,
    23648, 38976, 4096,  54304, 50336, 128,   35008, 19680,
    43456, 28128, 58784, 8576,  12544, 62752, 32096, 47424,
    29440, 46880, 16224, 64320, 60352, 12256, 42912, 25472,
    34464, 17024, 51904, 3808,  7776,  55872, 20992, 38432
};

// Reverse the 8 bits of a byte (host equivalent of (__brev(byte)) >> 24).
inline uint8_t reverseByte(uint8_t byte)
{
    uint8_t rev = 0;
    for(int i = 0; i < 8; ++i)
    {
        rev = static_cast<uint8_t>((rev << 1) | ((byte >> i) & 1u));
    }
    return rev;
}

// Host replica of ComputeCRC8LUT (polar_decoder.cu:1070).
uint8_t computeCRC8LUT(const uint8_t* bytes, int nBytes)
{
    uint8_t crc = 0;
    for(int b = 0; b < nBytes; ++b)
    {
        const uint8_t revByte = reverseByte(bytes[b]);
        const uint8_t pos     = static_cast<uint8_t>(crc ^ revByte);
        crc                   = CRC8_LUT[pos];
    }
    return crc;
}

// Host replica of ComputeCRC16LUT (polar_decoder.cu:1083).
uint16_t computeCRC16LUT(const uint8_t* bytes, int nBytes)
{
    uint16_t crc = 0;
    for(int b = 0; b < nBytes; ++b)
    {
        const uint16_t revByte = reverseByte(bytes[b]);
        const uint16_t pos     = static_cast<uint16_t>((crc >> 8) ^ revByte);
        crc                    = static_cast<uint16_t>((crc << 8) ^ CRC16_LUT[pos]);
    }
    return crc;
}

// CRC remainder exactly as validate_crc computes it: the whole LUT register.
//
// Solving for a zero register is solving for a genuine CRC, and it has exactly
// one solution. Do NOT reintroduce a mask here: the decoder's checkers used to
// test only part of the register, and solving against that produced codewords
// that are not valid CRC codewords at all (any of 2^4 / 2^5 fields would do).
uint32_t computeCrcRemainder(const uint8_t* bytes, int nBytes, uint8_t nCrcBits)
{
    if(nCrcBits == 6)
    {
        return static_cast<uint32_t>(computeCRC8LUT(bytes, nBytes));
    }
    return static_cast<uint32_t>(computeCRC16LUT(bytes, nBytes)); // nCrcBits == 11
}

// Solve for the nCrcBits CRC-field bits (placed at info-bit positions [A, K_cw))
// that make the decoder's CRC remainder of the payload||crc buffer equal 0.
//
// The masked CRC is linear over GF(2) with zero initial value, so:
//   crc(payload||crc) = crc(payload) XOR ( XOR_k crcBit[k] * basis[k] )
// where basis[k] = the remainder of a buffer with only info-bit (A+k) set. We
// want the total to be 0, i.e. XOR_k crcBit[k]*basis[k] == crc(payload).
// crcBits is written into hInfoBits (byte-packed, LSB-first) at positions
// [A, K_cw). Throws on inconsistency (would mean the host model is wrong).
void solveAndAppendCrc(uint8_t* hInfoBits, int nCrcBytes, uint16_t A, uint16_t K_cw, uint8_t nCrcBits)
{
    // Payload-only target (CRC field currently zero in hInfoBits).
    const uint32_t target = computeCrcRemainder(hInfoBits, nCrcBytes, nCrcBits);

    // Per-CRC-field-bit basis vectors.
    std::vector<uint32_t> basis(nCrcBits, 0);
    std::vector<uint8_t>  single(nCrcBytes, 0);
    for(uint8_t k = 0; k < nCrcBits; ++k)
    {
        const uint16_t p = static_cast<uint16_t>(A + k);
        std::fill(single.begin(), single.end(), 0);
        single[p >> 3] = static_cast<uint8_t>(1u << (p & 7u));
        basis[k]       = computeCrcRemainder(single.data(), nCrcBytes, nCrcBits);
    }

    // GF(2) linear-basis reduction with combination tracking (register <= 16 bits).
    constexpr int                          kMaxBits = 16;
    std::array<uint32_t, kMaxBits>         pivotVal{};
    std::array<uint32_t, kMaxBits>         pivotComb{};
    std::array<bool, kMaxBits>             pivotSet{};
    pivotSet.fill(false);

    for(uint8_t k = 0; k < nCrcBits; ++k)
    {
        uint32_t cur  = basis[k];
        uint32_t comb = (1u << k);
        for(int bit = kMaxBits - 1; bit >= 0; --bit)
        {
            if(!((cur >> bit) & 1u))
            {
                continue;
            }
            if(pivotSet[bit])
            {
                cur ^= pivotVal[bit];
                comb ^= pivotComb[bit];
            }
            else
            {
                pivotVal[bit]  = cur;
                pivotComb[bit] = comb;
                pivotSet[bit]  = true;
                break;
            }
        }
    }

    // Reduce target, accumulating which basis vectors are needed.
    uint32_t cur  = target;
    uint32_t comb = 0;
    for(int bit = kMaxBits - 1; bit >= 0; --bit)
    {
        if((cur >> bit) & 1u)
        {
            if(!pivotSet[bit])
            {
                throw std::runtime_error("polar generator: CRC solve is inconsistent (host CRC model mismatch)");
            }
            cur ^= pivotVal[bit];
            comb ^= pivotComb[bit];
        }
    }
    if(cur != 0)
    {
        throw std::runtime_error("polar generator: CRC solve failed to zero the residual (host CRC model mismatch)");
    }

    // Write the solved CRC-field bits into hInfoBits at positions [A, K_cw).
    for(uint8_t k = 0; k < nCrcBits; ++k)
    {
        if((comb >> k) & 1u)
        {
            const uint16_t p = static_cast<uint16_t>(A + k);
            hInfoBits[p >> 3] |= static_cast<uint8_t>(1u << (p & 7u));
        }
    }

    // Verify the assembled payload||crc buffer passes the decoder's check.
    if(computeCrcRemainder(hInfoBits, nCrcBytes, nCrcBits) != 0)
    {
        throw std::runtime_error("polar generator: constructed CRC field does not validate (host CRC model mismatch)");
    }
    (void)K_cw;
}

//----------------------------------------------------------------------------
// Host replica of the polar channel (triangular) interleaver used by the
// polSegDeRmDeItl kernel. The polar encoder's pTxBits are rate-matched but NOT
// channel-interleaved (PDCCH/PBCH style); the RX de-rate-match de-interleaves,
// so the generator applies the forward interleave to place each rate-matched bit
// e[rmIdx] at its transmission index. Mirrors compute_polDeItlDesc /
// computeColumnRowIndices in polar_seg_deRm_deItl.cu.
struct PolDeItlDesc
{
    int32_t nItlMat;
    int32_t nRowsRegion1;
    int32_t nColsRegion1;
    int32_t nBitsRegion1;
    int32_t nRowsRegion2;
    int32_t nBitsRegion1And2;
};

PolDeItlDesc computePolDeItlDesc(uint32_t nTxBits)
{
    PolDeItlDesc d{};
    const int32_t T   = static_cast<int32_t>(std::ceil((-1.f + std::sqrt(static_cast<float>(1 + 8 * nTxBits))) / 2.f));
    d.nItlMat         = T;
    const float   b                 = static_cast<float>(-(1 + 2 * T));
    const int32_t lastRmIdx         = static_cast<int32_t>(nTxBits) - 1;
    const int32_t lastRowIdxRegion1 = static_cast<int32_t>(std::floor((-b - std::sqrt(b * b - 8.f * lastRmIdx)) / 2.f));
    const int32_t lastColIdxRegion1 = lastRmIdx - lastRowIdxRegion1 * T + (lastRowIdxRegion1 - 1) * lastRowIdxRegion1 / 2;
    d.nBitsRegion1                  = (lastRowIdxRegion1 + 1) * (lastColIdxRegion1 + 1);
    d.nRowsRegion1                  = lastRowIdxRegion1 + 1;
    d.nColsRegion1                  = lastColIdxRegion1 + 1;
    d.nRowsRegion2                  = d.nRowsRegion1 - 1;
    const int32_t nColsRegion2      = ((T - d.nRowsRegion2 + 1) - d.nColsRegion1);
    d.nBitsRegion1And2              = d.nBitsRegion1 + nColsRegion2 * d.nRowsRegion2;
    return d;
}

// Map a transmission-order index to the rate-matched-bit index (rmIdx).
uint32_t chanItlToRmIdx(uint32_t chanItlIdx, uint32_t nTxBits, const PolDeItlDesc& d)
{
    uint32_t colIdx;
    uint32_t rowIdx;
    if(chanItlIdx < static_cast<uint32_t>(d.nBitsRegion1))
    {
        colIdx = chanItlIdx / d.nRowsRegion1;
        rowIdx = chanItlIdx % d.nRowsRegion1;
    }
    else if(chanItlIdx < static_cast<uint32_t>(d.nBitsRegion1And2))
    {
        colIdx = (chanItlIdx - d.nBitsRegion1) / d.nRowsRegion2 + d.nColsRegion1;
        rowIdx = (chanItlIdx - d.nBitsRegion1) % d.nRowsRegion2;
    }
    else
    {
        const uint32_t flippedIdx    = nTxBits - 1 - chanItlIdx;
        const uint32_t flippedColIdx = static_cast<uint32_t>(std::floor((-1.f + std::sqrt(static_cast<float>(1 + 8 * flippedIdx))) / 2.f));
        const uint32_t flippedRowIdx = flippedIdx - flippedColIdx * (flippedColIdx + 1) / 2;
        colIdx                       = d.nItlMat - 1 - flippedColIdx;
        rowIdx                       = flippedColIdx - flippedRowIdx;
    }
    return colIdx + d.nItlMat * rowIdx - rowIdx * (rowIdx - 1) / 2;
}

// Sub-block interleaver pattern P (38.212 Table 5.4.1.1-1), 32 entries.
constexpr uint8_t POLAR_SUBBLOCK_P[32] = {
    0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
    12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31};

//----------------------------------------------------------------------------
// Run the compCwTreeTypes kernel for one segment and return its 4*N_cw
// tree-types buffer ([2 header bytes][tree types][SC operation list][fast-SSC operation list]).
// This is the same frozen /
// info / parity assignment the decoder consumes; the host encoder reads the
// leaf entries (bit j at [N_cw + j]) from it. Boilerplate mirrors
// cuphy_ex_comp_cwTreeTypes, specialized to a single segment.
std::vector<uint8_t> computeSegTreeTypes(const cuphyPolarUciSegPrm_t& seg, cudaStream_t strm)
{
    const uint16_t                     N_cw = seg.N_cw;
    std::vector<cuphyPolarUciSegPrm_t> segVec(1, seg);

    size_t max_mem = 0;
    {
        constexpr size_t alignment = 128;
        auto reserve               = [&](size_t nBytes) { max_mem += (nBytes + alignment - 1) & ~(alignment - 1); };
        reserve(sizeof(cuphyPolarUciSegPrm_t));
        reserve(cuphy::polar::PolarCwTreeLayout::sizeBytes(N_cw));
    }
    cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

    cuphyPolarUciSegPrm_t* pSegPrmsGpu = static_cast<cuphyPolarUciSegPrm_t*>(linearAlloc.alloc(sizeof(cuphyPolarUciSegPrm_t)));
    CUDA_CHECK(cudaMemcpyAsync(pSegPrmsGpu, segVec.data(), sizeof(cuphyPolarUciSegPrm_t), cudaMemcpyHostToDevice, strm));
    CUDA_CHECK(cudaStreamSynchronize(strm));

    std::vector<uint8_t*> cwTreeTypesAddrVec(1);
    cwTreeTypesAddrVec[0] = static_cast<uint8_t*>(linearAlloc.alloc(cuphy::polar::PolarCwTreeLayout::sizeBytes(N_cw)));

    size_t        dynDescrSizeBytes, dynDescrAlignBytes;
    cuphyStatus_t statusGetWs = cuphyCompCwTreeTypesGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);
    if(CUPHY_STATUS_SUCCESS != statusGetWs) throw cuphy::cuphy_exception(statusGetWs);

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpuTreeAddrs(sizeof(uint8_t**));

    cuphyCompCwTreeTypesHndl_t hndl;
    cuphyStatus_t              statusCreate = cuphyCreateCompCwTreeTypes(&hndl);
    if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

    cuphyCompCwTreeTypesLaunchCfg_t launchCfg;
    const bool                      enableAsyncCpy = false;
    cuphyStatus_t                   setupStatus    = cuphySetupCompCwTreeTypes(hndl,
                                                            1,
                                                            segVec.data(),
                                                            pSegPrmsGpu,
                                                            cwTreeTypesAddrVec.data(),
                                                            dynDescrBufCpu.addr(),
                                                            dynDescrBufGpu.addr(),
                                                            dynDescrBufCpuTreeAddrs.addr(),
                                                            enableAsyncCpy,
                                                            &launchCfg,
                                                            strm);
    if(CUPHY_STATUS_SUCCESS != setupStatus) throw cuphy::cuphy_exception(setupStatus);

    CUDA_CHECK(cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, strm));
    CUDA_CHECK(cudaStreamSynchronize(strm));

    const CUDA_KERNEL_NODE_PARAMS& k = launchCfg.kernelNodeParamsDriver;
    CUresult                       runStatus = cuLaunchKernel(k.func, k.gridDimX, k.gridDimY, k.gridDimZ,
                                        k.blockDimX, k.blockDimY, k.blockDimZ, k.sharedMemBytes,
                                        static_cast<CUstream>(strm), k.kernelParams, k.extra);
    if(CUDA_SUCCESS != runStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
    CUDA_CHECK(cudaStreamSynchronize(strm));

    std::vector<uint8_t> treeTypes(cuphy::polar::PolarCwTreeLayout::sizeBytes(N_cw));
    CUDA_CHECK(cudaMemcpy(treeTypes.data(), cwTreeTypesAddrVec[0], treeTypes.size(), cudaMemcpyDeviceToHost));

    cuphyStatus_t statusDestroy = cuphyDestroyCompCwTreeTypes(hndl);
    if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

    return treeTypes;
}

//----------------------------------------------------------------------------
// Host uplink polar codeblock encoder + sub-block interleaving + rate matching,
// mirroring 5GModel uplinkPolarCbEncoder.m and uplinkPolarRmItl.m steps 1-2
// (38.212 sections 5.3.1.2, 5.4.1.1, 5.4.1.2). Unlike cuphyPolarEncRateMatch
// (the DL encoder, capped at N <= 512) this handles N up to the decoder limit
// (1024). cwBitTypes[j] are the per-bit leaf types (0 frozen, 1 info, 2 parity)
// from compCwTreeTypes; infoCrcBits holds the K_cw info+CRC bits (0/1). Fills
// packed (LSB-first) hCodedBits (N_cw coded bits, natural order) and hTxBits
// (E_cw rate-matched bits) in the same "rm" order cuphyPolarEncRateMatch emits,
// i.e. before the channel interleaver the caller applies.
void hostUplinkPolarEncodeRateMatch(const cuphyPolarUciSegPrm_t& seg,
                                    const std::vector<uint8_t>&  cwBitTypes,
                                    const std::vector<uint8_t>&  infoCrcBits,
                                    std::vector<uint8_t>&        hCodedBits,
                                    std::vector<uint8_t>&        hTxBits)
{
    const uint16_t N_cw = seg.N_cw;
    const uint8_t  n_cw = seg.n_cw;
    const uint32_t E_cw = seg.E_cw;
    const uint16_t K_cw = seg.K_cw;

    // Butterfly input: place info/CRC (type 1) and parity (type 2) bits, tracking
    // the length-5 shift register used for parity-check bits (38.212 5.3.1.2).
    std::vector<uint8_t> polCw(N_cw, 0);
    uint32_t             inputBitIdx = 0;
    uint8_t              y0 = 0, y1 = 0, y2 = 0, y3 = 0, y4 = 0;
    for(uint16_t i = 0; i < N_cw; ++i)
    {
        const uint8_t yt = y0;
        y0 = y1; y1 = y2; y2 = y3; y3 = y4; y4 = yt;
        if(cwBitTypes[i] == 1)
        {
            const uint8_t bit = infoCrcBits[inputBitIdx++];
            polCw[i]          = bit;
            y0                = static_cast<uint8_t>((y0 + bit) & 1u);
        }
        else if(cwBitTypes[i] == 2)
        {
            polCw[i] = y0;
        }
    }

    // Butterfly transform: n_cw stages of strided pairwise XOR.
    for(uint8_t s = 0; s < n_cw; ++s)
    {
        const uint32_t step = 1u << s;
        const uint32_t nGrp = N_cw / (2u * step);
        for(uint32_t g = 0; g < nGrp; ++g)
        {
            const uint32_t base = 2u * step * g;
            for(uint32_t kk = 0; kk < step; ++kk)
            {
                polCw[base + kk] ^= polCw[base + kk + step];
            }
        }
    }

    // Sub-block interleaving (38.212 5.4.1.1).
    std::vector<uint8_t> subItl(N_cw, 0);
    const uint32_t       subBlockSize = N_cw / 32u;
    for(uint32_t i = 0; i < N_cw; ++i)
    {
        const uint32_t sb  = i / subBlockSize;
        const uint32_t src = POLAR_SUBBLOCK_P[sb] * subBlockSize + (i % subBlockSize);
        subItl[i]          = polCw[src];
    }

    // Rate matching (38.212 5.4.1.2): repetition, puncturing, or shortening.
    std::vector<uint8_t> polCwRm(E_cw, 0);
    if(E_cw >= N_cw)
    {
        for(uint32_t kk = 0; kk < E_cw; ++kk) { polCwRm[kk] = subItl[kk % N_cw]; }
    }
    else if(16u * static_cast<uint32_t>(K_cw) <= 7u * E_cw)
    {
        for(uint32_t kk = 0; kk < E_cw; ++kk) { polCwRm[kk] = subItl[kk + N_cw - E_cw]; }
    }
    else
    {
        for(uint32_t kk = 0; kk < E_cw; ++kk) { polCwRm[kk] = subItl[kk]; }
    }

    // Pack LSB-first into the caller's byte buffers.
    std::fill(hCodedBits.begin(), hCodedBits.end(), 0);
    for(uint16_t i = 0; i < N_cw; ++i)
    {
        if(polCw[i]) { hCodedBits[i >> 3] |= static_cast<uint8_t>(1u << (i & 7u)); }
    }
    std::fill(hTxBits.begin(), hTxBits.end(), 0);
    for(uint32_t i = 0; i < E_cw; ++i)
    {
        if(polCwRm[i]) { hTxBits[i >> 3] |= static_cast<uint8_t>(1u << (i & 7u)); }
    }
}

//----------------------------------------------------------------------------
// checkGeneratorSupport()
// Throw a clear message if (A, E) fall outside the range the generator supports.
// The host encoder is not bound by the cuPHY *DL* polar-encoder caps, so N is
// limited only by the UL decoder (N <= CUPHY_POLAR_DECODER_MAX_BITS = 1024;
// 38.212 n_max = 10 for UCI). NR UCI polar starts at A = 12 (A in [3,11] is
// Reed-Muller, A in [1,2] simplex/repetition). N bounds are per codeblock
// (32 = 2^5 min; K <= N invariant). childCbIdxs is uint8_t[2] without ENABLE_64C
// and the de-rate-match kernel indexes the per-launch codeword array through it,
// so a launch is capped at 256 codewords (the decoder itself has no such limit).
void checkGeneratorSupport(const PolarGenParams& params, const cuphyPolarUciSegPrm_t& seg, uint32_t totalCbs)
{
    constexpr uint16_t kMinN            = 32;
    constexpr uint32_t kMaxCwsPerLaunch = 256;
    const uint16_t     N_cw             = seg.N_cw;
    const uint16_t     K_cw             = seg.K_cw;

    if(params.A < 12)
    {
        throw std::runtime_error("polar generator: A=" + std::to_string(params.A) +
                                 " is not polar-coded in NR UCI (A in [3,11] uses Reed-Muller, "
                                 "A in [1,2] uses simplex/repetition); use A >= 12");
    }
    if(N_cw < kMinN || N_cw > CUPHY_POLAR_DECODER_MAX_BITS)
    {
        throw std::runtime_error("polar generator: derived N_cw (" + std::to_string(N_cw) +
                                 ") is out of the supported [" + std::to_string(kMinN) + "," +
                                 std::to_string(CUPHY_POLAR_DECODER_MAX_BITS) + "] range for A=" +
                                 std::to_string(params.A) + ", E=" + std::to_string(params.E));
    }
    if(K_cw > N_cw)
    {
        throw std::runtime_error("polar generator: info block K_cw (" + std::to_string(K_cw) +
                                 ") exceeds derived mother code N_cw (" + std::to_string(N_cw) +
                                 "); increase -E or reduce -A");
    }
    if(totalCbs > kMaxCwsPerLaunch)
    {
        throw std::runtime_error("polar generator: " + std::to_string(totalCbs) +
                                 " codewords exceeds the 256 codewords/launch limit "
                                 "(childCbIdxs is uint8_t without ENABLE_64C); reduce -w");
    }
}

//----------------------------------------------------------------------------
// extractLeafBitTypes()
// Pull the per-bit leaf types (0 frozen, 1 info, 2 parity) out of the 2*N_cw
// compCwTreeTypes buffer, where leaf j lives at [N_cw + j].
std::vector<uint8_t> extractLeafBitTypes(const std::vector<uint8_t>& segTreeTypes, uint16_t N_cw)
{
    std::vector<uint8_t> cwBitTypes(N_cw, 0);
    for(uint16_t j = 0; j < N_cw; ++j)
    {
        cwBitTypes[j] = segTreeTypes[N_cw + j];
    }
    return cwBitTypes;
}

//----------------------------------------------------------------------------
// buildCodeblockInfoBits()
// Split a UCI segment payload into one codeblock's nInfoCb info bits (38.212
// 6.3.1.2). Single CB: the whole payload. Two CBs: CB0 = [zeros(zeroInsertFlag),
// payload[0:floorHalf]], CB1 = payload[floorHalf:A]; each carries ceil(A/2) bits.
std::vector<uint8_t> buildCodeblockInfoBits(const std::vector<uint8_t>& segPayload, uint8_t cb, uint8_t nCbs,
                                            uint16_t nInfoCb, uint8_t zeroInsertFlag, uint16_t floorHalf)
{
    std::vector<uint8_t> cbInfo(nInfoCb, 0);
    if(nCbs == 1)
    {
        for(uint16_t i = 0; i < nInfoCb; ++i) { cbInfo[i] = segPayload[i]; }
    }
    else if(cb == 0)
    {
        for(uint16_t i = zeroInsertFlag; i < nInfoCb; ++i) { cbInfo[i] = segPayload[i - zeroInsertFlag]; }
    }
    else
    {
        for(uint16_t i = 0; i < nInfoCb; ++i) { cbInfo[i] = segPayload[floorHalf + i]; }
    }
    return cbInfo;
}

//----------------------------------------------------------------------------
// modulateAwgn()
// BPSK-modulate nOut packed bits (bit i, or idxMap[i] when idxMap != nullptr),
// add AWGN, and return clamped fp16 channel LLRs. idxMap applies the forward
// channel interleaver for the rate-matched path; nullptr is the identity map.
std::vector<__half> modulateAwgn(const uint8_t* packedBits, uint32_t nOut, const uint32_t* idxMap,
                                 float invNoiseVar, float noiseStd, uint64_t noiseSeed)
{
    std::mt19937                    noiseRng(static_cast<uint32_t>(noiseSeed));
    std::normal_distribution<float> noiseDist(0.0f, noiseStd);
    std::vector<__half>             llrs(nOut);
    for(uint32_t i = 0; i < nOut; ++i)
    {
        const uint32_t bitIdx = (idxMap != nullptr) ? idxMap[i] : i;
        const uint8_t  b      = static_cast<uint8_t>((packedBits[bitIdx >> 3] >> (bitIdx & 7u)) & 1u);
        const float    x      = 1.0f - 2.0f * static_cast<float>(b);
        float          llr    = invNoiseVar * (x + noiseDist(noiseRng));
        llr                   = std::min(30.0f, std::max(-30.0f, llr));
        llrs[i]               = __float2half(llr);
    }
    return llrs;
}

//----------------------------------------------------------------------------
// deRateMatchToCwLLRs()
// Full-chain step: convert each codeblock's E_cw transmission-order LLRs into
// N_cw codeword LLRs via the polSegDeRmDeItl kernel. Per-segment received LLRs
// (E_seg = concatenation of the segment's CB tx LLRs) go in; per-codeblock N_cw
// LLRs are written to out.llrN. Boilerplate mirrors cuphy_ex_polar_seg_deRm_deItl.
void deRateMatchToCwLLRs(PolarGenResult& out, const std::vector<std::vector<__half>>& txLlrE,
                         uint8_t nCbs, uint16_t N_cw, uint32_t E_cw, cudaStream_t strm)
{
    const uint16_t nSegsDrDi = static_cast<uint16_t>(out.segPrms.size());
    const uint32_t nCwsDrDi  = static_cast<uint32_t>(nCbs) * nSegsDrDi;

    size_t drdiMem = 0;
    {
        constexpr size_t alignment = 128;
        auto reserve               = [&](size_t nBytes) { drdiMem += (nBytes + alignment - 1) & ~(alignment - 1); };
        reserve(sizeof(cuphyPolarUciSegPrm_t) * nSegsDrDi);
        reserve(sizeof(cuphyPolarCwPrm_t) * nCwsDrDi);
        for(uint16_t segIdx = 0; segIdx < nSegsDrDi; ++segIdx)
        {
            reserve(sizeof(__half) * out.segPrms[segIdx].E_seg); // received seg LLRs
        }
        for(uint32_t cbG = 0; cbG < nCwsDrDi; ++cbG)
        {
            reserve(sizeof(__half) * N_cw); // output cw LLRs
        }
    }
    cuphy::linear_alloc<128, cuphy::device_alloc> drdiAlloc(drdiMem);

    std::vector<cuphyPolarCwPrm_t> drdiCwPrms(nCwsDrDi);
    std::vector<__half*>           uciSegLLRsAddrVec(nSegsDrDi);
    std::vector<__half*>           cwLLRsAddrVec(nCwsDrDi);

    // Per-segment received LLR buffer: E_seg long, filled with each CB's E_cw
    // channel LLRs at offset cb*E_cw (zeroed first so any E_seg > nCbs*E_cw tail
    // is defined).
    for(uint16_t segIdx = 0; segIdx < nSegsDrDi; ++segIdx)
    {
        const uint32_t E_seg      = out.segPrms[segIdx].E_seg;
        uciSegLLRsAddrVec[segIdx] = static_cast<__half*>(drdiAlloc.alloc(sizeof(__half) * E_seg));
        CUDA_CHECK(cudaMemsetAsync(uciSegLLRsAddrVec[segIdx], 0, sizeof(__half) * E_seg, strm));
        for(uint8_t cb = 0; cb < nCbs; ++cb)
        {
            const uint32_t cbGlobal = static_cast<uint32_t>(nCbs) * segIdx + cb;
            CUDA_CHECK(cudaMemcpyAsync(uciSegLLRsAddrVec[segIdx] + cb * E_cw, txLlrE[cbGlobal].data(), sizeof(__half) * E_cw, cudaMemcpyHostToDevice, strm));
        }
        out.segPrms[segIdx].pUciSegLLRs = uciSegLLRsAddrVec[segIdx];
    }
    // Per-codeblock output LLR buffer (N_cw long).
    for(uint32_t cbG = 0; cbG < nCwsDrDi; ++cbG)
    {
        cwLLRsAddrVec[cbG]      = static_cast<__half*>(drdiAlloc.alloc(sizeof(__half) * N_cw));
        drdiCwPrms[cbG].N_cw    = N_cw;
        drdiCwPrms[cbG].pCwLLRs = cwLLRsAddrVec[cbG];
    }
    CUDA_CHECK(cudaStreamSynchronize(strm));

    cuphyPolarUciSegPrm_t* pSegPrmsGpuDrDi = static_cast<cuphyPolarUciSegPrm_t*>(drdiAlloc.alloc(sizeof(cuphyPolarUciSegPrm_t) * nSegsDrDi));
    cuphyPolarCwPrm_t*     pCwPrmsGpuDrDi  = static_cast<cuphyPolarCwPrm_t*>(drdiAlloc.alloc(sizeof(cuphyPolarCwPrm_t) * nCwsDrDi));
    CUDA_CHECK(cudaMemcpyAsync(pSegPrmsGpuDrDi, out.segPrms.data(), sizeof(cuphyPolarUciSegPrm_t) * nSegsDrDi, cudaMemcpyHostToDevice, strm));
    CUDA_CHECK(cudaMemcpyAsync(pCwPrmsGpuDrDi, drdiCwPrms.data(), sizeof(cuphyPolarCwPrm_t) * nCwsDrDi, cudaMemcpyHostToDevice, strm));
    CUDA_CHECK(cudaStreamSynchronize(strm));

    size_t        drdiDescrSizeBytes, drdiDescrAlignBytes;
    cuphyStatus_t drdiGetWs = cuphyPolSegDeRmDeItlGetDescrInfo(&drdiDescrSizeBytes, &drdiDescrAlignBytes);
    if(CUPHY_STATUS_SUCCESS != drdiGetWs) throw cuphy::cuphy_exception(drdiGetWs);

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> drdiDescrCpu(drdiDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> drdiDescrGpu(drdiDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> drdiUciAddrsCpu(sizeof(__half*) * nSegsDrDi);
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> drdiCwAddrsCpu(sizeof(__half*) * nCwsDrDi);

    cuphyPolSegDeRmDeItlHndl_t drdiHndl;
    cuphyStatus_t              drdiCreate = cuphyCreatePolSegDeRmDeItl(&drdiHndl);
    if(CUPHY_STATUS_SUCCESS != drdiCreate) throw cuphy::cuphy_exception(drdiCreate);

    cuphyPolSegDeRmDeItlLaunchCfg_t drdiLaunchCfg;
    bool                            drdiAsyncCpy = false;
    cuphyStatus_t                   drdiSetup    = cuphySetupPolSegDeRmDeItl(drdiHndl,
                                                            nSegsDrDi,
                                                            nCwsDrDi,
                                                            out.segPrms.data(),
                                                            pSegPrmsGpuDrDi,
                                                            drdiCwPrms.data(),
                                                            pCwPrmsGpuDrDi,
                                                            uciSegLLRsAddrVec.data(),
                                                            cwLLRsAddrVec.data(),
                                                            drdiDescrCpu.addr(),
                                                            drdiDescrGpu.addr(),
                                                            drdiCwAddrsCpu.addr(),
                                                            drdiUciAddrsCpu.addr(),
                                                            static_cast<uint8_t>(drdiAsyncCpy),
                                                            &drdiLaunchCfg,
                                                            strm);
    if(CUPHY_STATUS_SUCCESS != drdiSetup) throw cuphy::cuphy_exception(drdiSetup);

    if(!drdiAsyncCpy)
    {
        CUDA_CHECK(cudaMemcpyAsync(drdiDescrGpu.addr(), drdiDescrCpu.addr(), drdiDescrSizeBytes, cudaMemcpyHostToDevice, strm));
        CUDA_CHECK(cudaStreamSynchronize(strm));
    }

    const CUDA_KERNEL_NODE_PARAMS& drdiKernel = drdiLaunchCfg.kernelNodeParamsDriver;
    CUresult                       drdiRun    = cuLaunchKernel(drdiKernel.func,
                                         drdiKernel.gridDimX,
                                         drdiKernel.gridDimY,
                                         drdiKernel.gridDimZ,
                                         drdiKernel.blockDimX,
                                         drdiKernel.blockDimY,
                                         drdiKernel.blockDimZ,
                                         drdiKernel.sharedMemBytes,
                                         static_cast<CUstream>(strm),
                                         drdiKernel.kernelParams,
                                         drdiKernel.extra);
    if(CUDA_SUCCESS != drdiRun) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
    CUDA_CHECK(cudaStreamSynchronize(strm));

    for(uint32_t cbG = 0; cbG < nCwsDrDi; ++cbG)
    {
        out.llrN[cbG].resize(N_cw);
        CUDA_CHECK(cudaMemcpy(out.llrN[cbG].data(), cwLLRsAddrVec[cbG], sizeof(__half) * N_cw, cudaMemcpyDeviceToHost));
    }

    cuphyStatus_t drdiDestroy = cuphyDestroyPolSegDeRmDeItl(drdiHndl);
    if(CUPHY_STATUS_SUCCESS != drdiDestroy) throw cuphy::cuphy_exception(drdiDestroy);

    // The de-rate-match scratch is freed on return; drop the now-dangling device
    // pointers from the returned segment params.
    for(uint16_t segIdx = 0; segIdx < nSegsDrDi; ++segIdx)
    {
        out.segPrms[segIdx].pUciSegLLRs = nullptr;
    }
}
} // namespace

PolarGenResult generatePolarTestVectors(const PolarGenParams& params, cudaStream_t strm)
{
    if(params.nCws == 0)
    {
        throw std::runtime_error("polar generator requires at least one segment (-w)");
    }
    if(params.E == 0)
    {
        throw std::runtime_error("polar generator requires E >= 1 (-E)");
    }

    // All codeblocks share identical dimensions (same A and E).
    const cuphyPolarUciSegPrm_t segTemplate = derivePolarSegPrm(params.A, params.E);
    const uint16_t N_cw      = segTemplate.N_cw;
    const uint16_t K_cw      = segTemplate.K_cw;
    const uint8_t  nCbs      = segTemplate.nCbs;                       // 1 or 2 codeblocks per segment
    const uint16_t nSegs     = params.nCws;                            // one UCI segment per -w
    const uint32_t totalCbs  = static_cast<uint32_t>(nSegs) * nCbs;
    const uint16_t nInfoCb   = static_cast<uint16_t>(K_cw - segTemplate.nCrcBits); // info bits per CB
    const uint32_t E_cw      = segTemplate.E_cw;                       // rate-matched bits per CB
    const uint16_t floorHalf = static_cast<uint16_t>(params.A / 2u);

    checkGeneratorSupport(params, segTemplate, totalCbs);

    PolarGenResult out;
    out.segPrms.resize(nSegs);
    out.treeTypes4N.resize(nSegs);       // per segment (both CBs share)
    out.llrN.resize(totalCbs);           // per codeblock
    out.srcPayloadBits.resize(totalCbs); // per codeblock (the CB's nInfoCb info bits)
    for(uint16_t segIdx = 0; segIdx < nSegs; ++segIdx)
    {
        out.segPrms[segIdx] = segTemplate;
        for(uint8_t cb = 0; cb < nCbs; ++cb)
        {
            out.segPrms[segIdx].childCbIdxs[cb] = static_cast<uint16_t>(nCbs * segIdx + cb);
        }
    }

    // Frozen / info / parity bit types (shared by both CBs) from compCwTreeTypes.
    const std::vector<uint8_t> segTreeTypes = computeSegTreeTypes(segTemplate, strm);
    const std::vector<uint8_t> cwBitTypes   = extractLeafBitTypes(segTreeTypes, N_cw);
    for(uint16_t segIdx = 0; segIdx < nSegs; ++segIdx)
    {
        out.treeTypes4N[segIdx] = segTreeTypes;
    }

    // Host-encoder scratch buffers (packed bits, LSB-first, 32-bit aligned).
    std::vector<uint8_t> hInfoBits(roundUp4((K_cw + 7u) / 8u), 0);
    std::vector<uint8_t> hCodedBits(roundUp4((N_cw + 7u) / 8u), 0);
    std::vector<uint8_t> hTxBits(roundUp4((E_cw + 7u) / 8u), 0);

    // Full-chain path collects each codeblock's E_cw received LLRs to feed the
    // de-rate-match / de-interleave kernel. rmIdxOfChanItl[i] maps a per-CB
    // transmission index to the encoder's rate-matched bit index (the forward
    // channel interleaver; E_cw is constant across codeblocks).
    std::vector<std::vector<__half>> txLlrE;
    std::vector<uint32_t>            rmIdxOfChanItl;
    if(params.fullChain)
    {
        txLlrE.resize(totalCbs);
        const PolDeItlDesc deItlDesc = computePolDeItlDesc(E_cw);
        rmIdxOfChanItl.resize(E_cw);
        for(uint32_t i = 0; i < E_cw; ++i)
        {
            rmIdxOfChanItl[i] = chanItlToRmIdx(i, E_cw, deItlDesc);
        }
    }

    const double noiseVar    = std::pow(10.0, -static_cast<double>(params.snrDb) / 10.0);
    const float  noiseStd    = static_cast<float>(std::sqrt(noiseVar));
    const float  invNoiseVar = static_cast<float>(2.0 / noiseVar);

    // CRC is computed over nWords*4 bytes (nWords = ceil(K_cw/32)), matching
    // validate_crc; this equals infoBytes for the sizes we generate.
    const int nCrcBytes = static_cast<int>(((K_cw + 31u) / 32u) * 4u);

    for(uint16_t segIdx = 0; segIdx < nSegs; ++segIdx)
    {
        // Draw the UCI segment's A payload bits.
        std::mt19937_64      payloadRng(params.seed + segIdx);
        std::vector<uint8_t> segPayload(params.A);
        for(uint16_t i = 0; i < params.A; ++i)
        {
            segPayload[i] = static_cast<uint8_t>(payloadRng() & 1u);
        }

        for(uint8_t cb = 0; cb < nCbs; ++cb)
        {
            const uint32_t             cbGlobal = static_cast<uint32_t>(nCbs) * segIdx + cb;
            const std::vector<uint8_t> cbInfo   = buildCodeblockInfoBits(segPayload, cb, nCbs, nInfoCb,
                                                                         segTemplate.zeroInsertFlag, floorHalf);

            // Pack the info bits and solve the real CRC field (CRC occupies
            // [nInfoCb, K_cw)) so the decoder's per-CB crcErrorFlag comes out 0.
            std::fill(hInfoBits.begin(), hInfoBits.end(), 0);
            for(uint16_t i = 0; i < nInfoCb; ++i)
            {
                if(cbInfo[i]) { hInfoBits[i >> 3] |= static_cast<uint8_t>(1u << (i & 7u)); }
            }
            solveAndAppendCrc(hInfoBits.data(), nCrcBytes, nInfoCb, K_cw, segTemplate.nCrcBits);

            // Host-encode: K_cw info+CRC bits -> N_cw coded + E_cw rate-matched bits.
            std::vector<uint8_t> infoCrcBits(K_cw, 0);
            for(uint16_t j = 0; j < K_cw; ++j)
            {
                infoCrcBits[j] = static_cast<uint8_t>((hInfoBits[j >> 3] >> (j & 7u)) & 1u);
            }
            hostUplinkPolarEncodeRateMatch(segTemplate, cwBitTypes, infoCrcBits, hCodedBits, hTxBits);
            out.srcPayloadBits[cbGlobal] = cbInfo; // decoder compares each CB against these

            // Channel: the simplified path modulates the N coded bits directly
            // (E == N model); the full-chain path modulates the E_cw rate-matched
            // bits through the forward channel interleaver and de-rate-matches after.
            const uint64_t noiseSeed = params.seed * 1000003ull + cbGlobal;
            if(!params.fullChain)
            {
                out.llrN[cbGlobal] = modulateAwgn(hCodedBits.data(), N_cw, nullptr, invNoiseVar, noiseStd, noiseSeed);
            }
            else
            {
                txLlrE[cbGlobal] = modulateAwgn(hTxBits.data(), E_cw, rmIdxOfChanItl.data(), invNoiseVar, noiseStd, noiseSeed);
            }
        }
    }

    if(params.fullChain)
    {
        deRateMatchToCwLLRs(out, txLlrE, nCbs, N_cw, E_cw, strm);
    }

    return out;
}
