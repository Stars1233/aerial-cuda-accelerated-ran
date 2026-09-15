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

#if !defined(PUSCH_UTILS_HPP_INCLUDED_)
#define PUSCH_UTILS_HPP_INCLUDED_

#include "cuphy_api.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "utils.cuh"
#include "tensor_desc.hpp"
#include "ldpc/ldpc_params.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <type_traits>

//--------------------------------------------------------------------------------
// cuphyDerivedPuschCmnPrms contains derived 3gpp user group parameters

struct cuphyDerivedPuschCmnPrms
{
    uint16_t chEst0DmrsSymLocBmsk;
    uint16_t chEst1DmrsSymLocBmsk;
    uint16_t rssiSymLocBmsk;
    uint32_t nMaxPrb;
};

//--------------------------------------------------------------------------------
// cuphyDerivedPuschUeGrpPrms contains derived 3gpp user group parameters

struct cuphyDerivedPuschUeGrpPrms
{
    uint32_t activeDMRSGridBmsk;
    uint32_t nDMRSSyms;
    uint32_t nDMRSGridsPerPRB;

    uint32_t nLayers;
    uint16_t nPrb;
    uint32_t nDataSymbols;

    // TODO: remove tensors
    cuphy::tensor_pinned tDmrsSymLocCpu, tDataSymLocCpu, tStartPrbCpu, tNumPrbCpu, tDmrsScIdCpu;
    using typed_tensor_pinned_R_8U = cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>;
    typed_tensor_pinned_R_8U tQamInfoCpu;

    cuphy::tensor_device tDmrsSymLocGpu, tDataSymLocGpu, tQamInfoGpu, tStartPrbGpu, tNumPrbGpu, tDmrsScIdGpu;
    cuphyTensorPrm_t     tPrmStartPrbGpu, tPrmNumPrbGpu, tPrmDmrsScIdGpu;

    cuphyDerivedPuschUeGrpPrms(cudaStream_t cuStream)
    {
        tDmrsSymLocCpu = std::move(cuphy::tensor_pinned(CUPHY_R_8U, N_MAX_DMRS_SYMS));
        tDmrsSymLocGpu = std::move(cuphy::tensor_device(CUPHY_R_8U, N_MAX_DMRS_SYMS));

        tDataSymLocCpu = std::move(cuphy::tensor_pinned(CUPHY_R_8U, 14));
        tDataSymLocGpu = std::move(cuphy::tensor_device(CUPHY_R_8U, 14));

        tQamInfoCpu = std::move(typed_tensor_pinned_R_8U(8, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight));
        tQamInfoGpu = std::move(cuphy::tensor_device(CUPHY_R_8U, 8, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight));

        tStartPrbCpu = std::move(cuphy::tensor_pinned(CUPHY_R_16U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight));
        tStartPrbGpu = std::move(cuphy::tensor_device(CUPHY_R_16U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight));

        tNumPrbCpu = std::move(cuphy::tensor_pinned(CUPHY_R_16U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight));
        tNumPrbGpu = std::move(cuphy::tensor_device(CUPHY_R_16U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight));

        tDmrsScIdCpu = std::move(cuphy::tensor_pinned(CUPHY_R_16U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight));
        tDmrsScIdGpu = std::move(cuphy::tensor_device(CUPHY_R_16U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight));

        tPrmStartPrbGpu.desc  = tStartPrbGpu.desc().handle();
        tPrmStartPrbGpu.pAddr = tStartPrbGpu.addr();

        tPrmNumPrbGpu.desc  = tNumPrbGpu.desc().handle();
        tPrmNumPrbGpu.pAddr = tNumPrbGpu.addr();

        tPrmDmrsScIdGpu.desc  = tDmrsScIdGpu.desc().handle();
        tPrmDmrsScIdGpu.pAddr = tDmrsScIdGpu.addr();
    }
};

struct cuphyChEqParams
{
    uint8_t nTimeChEq = 1; // number of channel equalizer coefficient updates in time
};

//--------------------------------------------------------------------------------
// cuphyLDPCParams contains derived API parameters along with reciver LDPC settings

struct cuphyLDPCParams
{
    std::vector<uint32_t> KbArray;          // ""
    uint32_t              fixedMaxNumItrs;  // number of max iterations for LDPC, used when ldpcMaxNumItrAlgo = FIXED),
    bool                  earlyTermination; // LDPC early termination
    uint32_t              algoIndex;        // LDPC algoIndex
    std::vector<uint32_t> parityNodesArray; // LDPC parity nodes
    uint32_t              flags;            // LDPC flags (default = 0)
    bool                  useHalf;          // LDPC flag for half precision

    cuphyLDPCParams() {}

    cuphyLDPCParams(const cuphyPuschStatPrms_t* pStatPrms) :
        fixedMaxNumItrs(pStatPrms->fixedMaxNumLdpcItrs),
        earlyTermination(static_cast<bool>(pStatPrms->ldpcEarlyTermination)),
        algoIndex(pStatPrms->ldpcAlgoIndex),
        flags(pStatPrms->ldpcFlags),
        useHalf(static_cast<bool>(pStatPrms->ldpcUseHalf))
    {
        KbArray.resize(pStatPrms->nMaxTbs == 0 ? MAX_N_TBS_PER_CELL_GROUP_SUPPORTED : pStatPrms->nMaxTbs);
        parityNodesArray.resize(pStatPrms->nMaxTbs == 0 ? MAX_N_TBS_PER_CELL_GROUP_SUPPORTED : pStatPrms->nMaxTbs);
    }
};

inline uint32_t getMaxNCbsPerTb(cuphyPuschStatPrms_t const* pStatPrms)
{
    uint32_t maxNCbsPerTb = pStatPrms->nMaxCbsPerTb == 0 ? MAX_N_CBS_PER_TB_SUPPORTED : pStatPrms->nMaxCbsPerTb;
    return maxNCbsPerTb;
}

inline uint32_t getMaxNTbs(cuphyPuschStatPrms_t const* pStatPrms)
{
    uint32_t maxNTbs = pStatPrms->nMaxTbs == 0 ? (pStatPrms->nMaxCellsPerSlot > 1 ? MAX_N_TBS_PER_CELL_GROUP_SUPPORTED : MAX_N_TBS_SUPPORTED) : pStatPrms->nMaxTbs;
    return maxNTbs;
}

inline uint32_t getMaxNCbs(cuphyPuschStatPrms_t const* pStatPrms)
{
    uint32_t maxNCbs = pStatPrms->nMaxTotCbs == 0 ? getMaxNTbs(pStatPrms) * getMaxNCbsPerTb(pStatPrms) : pStatPrms->nMaxTotCbs;
    return maxNCbs;
}

inline uint32_t getMaxNPrbAlloc(cuphyPuschStatPrms_t const* pStatPrms)
{
    uint32_t maxNPrbAlloc = pStatPrms->nMaxPrb == 0 ? MAX_N_PRBS_SUPPORTED : pStatPrms->nMaxPrb;
    return maxNPrbAlloc;
}

struct PuschRateMatchDescrInfo
{
    size_t sizeBytes;
    size_t alignBytes;
};

/**
 * Returns the descriptor allocation required for early-SCH rate matching.
 *
 * Full-TB decoding does not use the early rate-matching path, so its descriptor
 * is omitted from the dynamic descriptor allocation and bulk H2D copy.
 *
 * @param[in] useCbLdpc Whether the codeblock-centric LDPC path is enabled.
 * @param[in] rateMatchDescrSizeBytes Size of the regular rate-matching descriptor.
 * @param[in] rateMatchDescrAlignBytes Alignment of the regular rate-matching descriptor.
 * @return Size and alignment required for the early rate-matching descriptor.
 */
[[nodiscard]] constexpr PuschRateMatchDescrInfo getEarlyRateMatchDescrInfo(
    bool useCbLdpc, size_t rateMatchDescrSizeBytes, size_t rateMatchDescrAlignBytes) noexcept
{
    return useCbLdpc ? PuschRateMatchDescrInfo{rateMatchDescrSizeBytes, rateMatchDescrAlignBytes}
                     : PuschRateMatchDescrInfo{0, 0};
}

/**
 * Returns the number of complete leading codeblocks represented by early-SCH bits.
 *
 * The UCI path partitions the rate-matched budget in @c G; the non-UCI path uses
 * @c encodedSize.
 *
 * @param[in] tbPrms Per-transport-block rate-matching parameters.
 * @param[in] availableSchBits Number of early shared-channel bits available.
 * @return The count of complete leading codeblocks available for early decode.
 */
[[nodiscard]] inline uint16_t computeLeadingEarlySchCbCount(const PerTbParams& tbPrms, uint32_t availableSchBits)
{
    const uint32_t rateMatchedBits = tbPrms.uciOnPuschFlag ? tbPrms.G : tbPrms.encodedSize;
    if (availableSchBits == 0 || tbPrms.num_CBs == 0 || tbPrms.Nl == 0 || tbPrms.Qm == 0 ||
        rateMatchedBits == 0)
    {
        return 0;
    }

    const uint32_t bitsPerQamBlock = tbPrms.Nl * tbPrms.Qm;
    if (bitsPerQamBlock == 0)
    {
        return 0;
    }

    const uint32_t nQamBlocks          = rateMatchedBits / bitsPerQamBlock;
    const uint32_t qamBlocksPerCb      = nQamBlocks / tbPrms.num_CBs;
    const uint32_t longerCbRemainder   = nQamBlocks - qamBlocksPerCb * tbPrms.num_CBs;
    const uint32_t shorterCbCount      = tbPrms.num_CBs - longerCbRemainder;
    const bool     hasLongerCodeblocks =
        qamBlocksPerCb * bitsPerQamBlock * tbPrms.num_CBs < rateMatchedBits;
    uint64_t       consumedSchBits     = 0;
    uint16_t       completeLeadingCbs  = 0;

    for (uint32_t cbIdx = 0; cbIdx < tbPrms.num_CBs; ++cbIdx)
    {
        const uint32_t cbQamBlocks = qamBlocksPerCb +
            ((cbIdx >= shorterCbCount && longerCbRemainder > 0 && hasLongerCodeblocks) ? 1U : 0U);
        const uint64_t cbSchBits   = static_cast<uint64_t>(cbQamBlocks) * bitsPerQamBlock;
        if (cbSchBits == 0 || consumedSchBits + cbSchBits > availableSchBits)
        {
            break;
        }
        consumedSchBits += cbSchBits;
        ++completeLeadingCbs;
    }

    return completeLeadingCbs;
}

/**
 * Determines whether the early-SCH decode batch requires preparation.
 *
 * @param[in] nEarlyDecodedSchCbs Number of codeblocks scheduled for early decode.
 * @return @c true when at least one codeblock is scheduled for early decode.
 */
[[nodiscard]] inline bool needsEarlyCbBatchPreparation(uint32_t nEarlyDecodedSchCbs)
{
    return nEarlyDecodedSchCbs != 0;
}

/**
 * Determines whether the full-slot path has rate-matching and LDPC work.
 *
 * Full-TB LDPC always uses the full-slot backend. Codeblock-centric LDPC can
 * omit that work when every scheduled codeblock was decoded in the subslot
 * phase. CRC processing remains separate and may still be required.
 *
 * @param[in] useCbLdpc Whether the codeblock-centric LDPC path is enabled.
 * @param[in] nFullSlotRemainingSchCbs Number of SCH codeblocks left for the full-slot phase.
 * @return @c true when full-slot rate matching and LDPC decoding are required.
 *         @c false when codeblock-centric LDPC is enabled and no SCH codeblocks
 *         remain for full-slot decoding.
 */
[[nodiscard]] constexpr bool needsFullSlotSchDecode(
    bool useCbLdpc, uint32_t nFullSlotRemainingSchCbs) noexcept
{
    return !useCbLdpc || nFullSlotRemainingSchCbs != 0;
}

// Expand Parameters Helpers: UCI on PUSCH
inline uint8_t crcLength(uint32_t nBits)
{
    if(nBits <= 11)
        return 0;
    else if((nBits >= 12) && (nBits <= 19))
        return 6;
    else
        return 11;
}

inline uint32_t firstTerm(bool isDataPresent, uint32_t oUci, uint32_t lUci, float betaOffsetPusch, uint32_t mScUciSum, uint32_t codeBlockSizeSum, double codeRate, uint32_t Qm)
{
    const uint32_t sharedChannelUciSum = isDataPresent ? mScUciSum : 1u;

    const float numerator   = (oUci + lUci) * betaOffsetPusch * sharedChannelUciSum;
    const float denominator = isDataPresent ? codeBlockSizeSum : codeRate * Qm;

    return static_cast<uint32_t>(std::ceil(numerator / denominator));
}

inline std::tuple<uint32_t, uint32_t> rateMatchAck(bool isDataPresent, uint32_t oAck, float betaOffsetPusch, uint32_t mScUciSum, uint32_t codedBitsSum, float alpha, uint32_t mScUciSumFroml0, uint32_t nl, uint32_t qam, double codeRate)
{
    const uint8_t  lAck          = crcLength(oAck); // Ref. 38.212 Sec. 6.3.1.2.1 and 6.3.1.2.2
    const uint32_t firstTermAck  = firstTerm(isDataPresent, oAck, lAck, betaOffsetPusch, mScUciSum, codedBitsSum, codeRate, qam);
    const uint32_t secondTermAck = static_cast<uint32_t>(std::ceil(alpha * mScUciSumFroml0));

    const auto Qack = std::min(firstTermAck, secondTermAck);
    const auto Eack = nl * Qack * qam;

    return {Qack, Eack};
}

inline float alphaScalingToAlphaMapping(uint32_t alphaScaling)
{
    constexpr float alphaScalingMap[] = {
        0.5, 0.65, 0.8, 1.0};
    constexpr size_t max = std::extent<decltype(alphaScalingMap)>::value;

    if(alphaScaling >= max)
        throw std::runtime_error("alphaScaling invalid");

    return alphaScalingMap[alphaScaling];
}

inline float betaOffsetHarqMapping(uint32_t betaOffsetHarqAck)
{
    constexpr float betaOffsetHarqAckMap[] = {
        1.000, 2.000, 2.500, 3.125, 4.000, 5.000, 6.250, 8.000, 10.000, 12.625, 15.875, 20.000, 31.000, 50.000, 80.000, 126.000};
    constexpr size_t max = std::extent<decltype(betaOffsetHarqAckMap)>::value;

    if(betaOffsetHarqAck >= max)
        throw std::runtime_error("betaOffsetHarqAck reserved or invalid");

    return betaOffsetHarqAckMap[betaOffsetHarqAck];
}

inline float betaOffsetCsiMapping(uint32_t betaOffsetCsi)
{
    constexpr float betaOffsetCsiMap[] = {
        1.125, 1.250, 1.375, 1.625, 1.750, 2.000, 2.250, 2.500, 2.875, 3.125, 3.500, 4.000, 5.000, 6.250, 8.000, 10.000, 12.625, 15.875, 20.000};
    constexpr size_t max = std::extent<decltype(betaOffsetCsiMap)>::value;

    if(betaOffsetCsi >= max)
        throw std::runtime_error("betaOffsetCsi reserved or invalid");

    return betaOffsetCsiMap[betaOffsetCsi];
}

/*--------------------------------------------------------------------------------
 * Parameters:
 *   Outputs:
 *     PerTbParams::G_harq
 *     PerTbParams::G_harq_rvd
 *     PerTbParams::G_csi1
 *     PerTbParams::G
 *   Inputs:
 *     cuphyPuschUePrm_t::pduBitmap
 *     cuphyPuschCellDynPrm_t::nPuschSym
 *     puschUeGrpPrms::nPrb
 *     C
 *     codeRate
 *     pDmrsSymLoc // DMRS symbol location array
 *     pDataSymLoc // Data symbol location array
 * Functionality:
 *   Calculates rate matching output sequence lengths for HARQ, CSI-PART1, and ULSCH
 *--------------------------------------------------------------------------------*/
inline void rate_match_seq_len(PerTbParams& tbPrms, const cuphyPuschUePrm_t& puschUePrms, const cuphyPuschCellDynPrm_t& cellDynPrms, const cuphyPuschUeGrpPrm_t& puschUeGrpPrms, double codeRate, uint8_t dataCnt, const uint8_t* pDataSymLoc, uint8_t dmrsCnt, const uint8_t* pDmrsSymLoc, uint8_t nDmrsCdmGrpsNoData, uint8_t enableCsiP2Fapiv3)
{
    // UCI on PUSCH is not configured, exit early
    const cuphyUciOnPuschPrm_t uciPrms =
        (puschUePrms.pUciPrms) ?
            (*puschUePrms.pUciPrms) :
            (cuphyUciOnPuschPrm_t{0, 0, 3, 0, 0, 0, 0, 0, 0});

    const auto Nl = tbPrms.Nl;
    const auto Qm = tbPrms.Qm;
    const auto K  = tbPrms.K;
    const auto C  = tbPrms.num_CBs;
    // codeRate = coderate calculated earlier

    auto       nBitsHarq         = uciPrms.nBitsHarq;
    const auto nBitsCsi1         = uciPrms.nBitsCsi1;
    const auto alphaScaling      = uciPrms.alphaScaling;
    const auto betaOffsetHarqAck = uciPrms.betaOffsetHarqAck;
    const auto betaOffsetCsi1    = uciPrms.betaOffsetCsi1;
    uint16_t   nCsi2Reports      = uciPrms.nCsi2Reports;

    tbPrms.betaOffsetCsi2 = uciPrms.betaOffsetCsi2;
    tbPrms.codeRate       = codeRate;

    const auto pduBitmap     = puschUePrms.pduBitmap;
    const auto nPrb          = puschUeGrpPrms.nPrb;
    const auto nPuschSym     = puschUeGrpPrms.nPuschSym;
    const auto puschStartSym = puschUeGrpPrms.puschStartSym;
    // dataSymLoc_array = pDataSymLoc
    // dmrsSymLoc_array = pDmrsSymLoc

    const auto alpha           = alphaScalingToAlphaMapping(alphaScaling); // Ref.: SCF FAPI Table 3-48 (optional puschUci information)
    const auto betaOffsetPusch = betaOffsetHarqMapping(betaOffsetHarqAck); // Ref.: Table 9.3-1, TS 38.213

    tbPrms.alpha = alpha;

    std::array<uint32_t, OFDM_SYMBOLS_PER_SLOT> mScUlsch{};
    std::array<uint32_t, OFDM_SYMBOLS_PER_SLOT> mScUci{};

    std::fill(mScUlsch.begin(), mScUlsch.begin() + nPuschSym, nPrb * 12);
    std::fill(mScUci.begin(), mScUci.begin() + nPuschSym, nPrb * 12);

    // Set correct number of ULSCH and UCI resources on DMRS symbols
    for(int i = 0; i < dmrsCnt; ++i)
    {
        uint8_t dmrsSymIdx_within_puschSymbols = pDmrsSymLoc[i] - puschStartSym;

        if(nDmrsCdmGrpsNoData == 1)
        {
            mScUlsch[dmrsSymIdx_within_puschSymbols] = mScUlsch[dmrsSymIdx_within_puschSymbols] >> 1;
        }else{
            mScUlsch[dmrsSymIdx_within_puschSymbols] = 0;
        }
        mScUci[dmrsSymIdx_within_puschSymbols] = 0;
    }

    // Identify first dataSym after DMRS
    uint8_t firstDataSymIdx_after_dmrs_within_puschSymbols = pDmrsSymLoc[0] + 1 - puschStartSym;
    for(int i = 1; i < dmrsCnt; ++i)
    {
        if(firstDataSymIdx_after_dmrs_within_puschSymbols == (pDmrsSymLoc[i] - puschStartSym))
        {
            firstDataSymIdx_after_dmrs_within_puschSymbols = pDmrsSymLoc[i] - puschStartSym + 1;
        }else
        {
            break;
        }
    }

    const auto mScUlschSum = std::accumulate(std::begin(mScUlsch), std::end(mScUlsch), 0);

    // Total number of REs available for UCI transmission
    const auto mScUciSum = std::accumulate(std::begin(mScUci), std::end(mScUci), 0);

    // Take sum from first index at firstDataSymIdx_after_dmrs_within_puschSymbols to end of mScUci
    const auto mScUciSumFroml0 = std::accumulate(std::begin(mScUci) + firstDataSymIdx_after_dmrs_within_puschSymbols, std::begin(mScUci) + nPuschSym, 0);

    const auto mUlsch = mScUlschSum;

    tbPrms.mScUciSum = mScUciSum;

    // Asssumption: all code blocks are transmitted, C_ULsch is the number of code blocks for UL-SCH of the PUSCH transmission
    // sum(ones(C, 1)*Kr)
    // Summation of K_r, r=0,..,(C_ULSCH-1), the denominator of first term in Q'_ACK [Ref. TS 38.212 Sec. 6.3.2.4.1.1]
    const auto codedBitsSum = K * C;

    tbPrms.codedBitsSum = codedBitsSum;

    const auto isDataPresent = pduBitmap & 0b00001; // Bit0 = 1 in pduBitmap, if data is present
                                                    // const auto isCsi2Present = pduBitmap & 0b10000; // Bit5 = 1 in pduBitmap, if CSI2 is present (NOT a FAPI defined field)

    tbPrms.isDataPresent = isDataPresent;


    uint32_t isCsi2Present = 0;
    if(enableCsiP2Fapiv3)
    {
        if(nCsi2Reports > 0)
        {
            isCsi2Present = 1;
        }
    }else
    {
        isCsi2Present = (pduBitmap >> 5) & 1;
    }

    if(!isDataPresent && !isCsi2Present && nBitsCsi1)
    {
        if(nBitsHarq == 0 || nBitsHarq == 1)
        {
            nBitsHarq = 2;
        }
    }

    // Number of reserved ACK bits, Ref. 38.212 Sec. 6.2.7 (Step 1)
    const auto nBitsAckRvd = (nBitsHarq <= 2) ? 2 : 0;

    // Bit capacity, and modulation symbol capacity (per layer) and rate matched
    // output sequence length for HARQ-ACK payload
    uint32_t qPrimeAck_rvd                     = 0;
    uint32_t qPrimeAck                         = 0;
    std::tie(qPrimeAck_rvd, tbPrms.G_harq_rvd) = rateMatchAck(isDataPresent, nBitsAckRvd, betaOffsetPusch, mScUciSum, codedBitsSum, alpha, mScUciSumFroml0, Nl, Qm, codeRate);
    std::tie(qPrimeAck, tbPrms.G_harq)         = rateMatchAck(isDataPresent, nBitsHarq, betaOffsetPusch, mScUciSum, codedBitsSum, alpha, mScUciSumFroml0, Nl, Qm, codeRate);

    tbPrms.qPrimeAck = qPrimeAck;
    /****Calculation of ECsi1****/
    if(nBitsCsi1)
    {
        const auto betaOffsetPusch = betaOffsetCsiMapping(betaOffsetCsi1); // Ref.: Table 9.3-2, TS 38.213
        const auto lCsi1           = crcLength(nBitsCsi1);

        const auto QPrimeAckCsi1 = (nBitsHarq > 2) ? qPrimeAck : qPrimeAck_rvd;

        const auto firstTermCsi1 = firstTerm(isDataPresent, nBitsCsi1, lCsi1, betaOffsetPusch, mScUciSum, codedBitsSum, codeRate, Qm);

        const auto qPrimeCsi1 =
            (isDataPresent) ?
                std::min(firstTermCsi1, static_cast<uint32_t>(std::ceil(alpha * mScUciSum)) - QPrimeAckCsi1) :
                (isCsi2Present) ?
                std::min(firstTermCsi1, mScUciSum - QPrimeAckCsi1) :
                mScUciSum - QPrimeAckCsi1;

        // Output: // CSI-1 bit capacity
        tbPrms.G_csi1     = Nl * qPrimeCsi1 * Qm;
        tbPrms.qPrimeCsi1 = qPrimeCsi1;
    }
    else
    {
        tbPrms.G_csi1     = 0;
        tbPrms.qPrimeCsi1 = 0;
    }

    /****Calculation of EUlsch****/
    const auto G_Ulsch = mUlsch * Qm * Nl; // bit capacity of UL-SCH without UCI

    // tbPrms.G =
    //     (isDataPresent) ?
    //         G_Ulsch - tbPrms.G_harq * (nBitsAckRvd == 0) - tbPrms.G_csi1 :
    //         0;
    if((isDataPresent > 0) ||  (isCsi2Present > 0))
    {
        tbPrms.G = G_Ulsch - tbPrms.G_harq * (nBitsAckRvd == 0) - tbPrms.G_csi1;
    }else{
        tbPrms.G = 0;
    }
    tbPrms.G_schAndCsi2 = tbPrms.G;


    // save nBitsHarq
    tbPrms.nBitsHarq     = nBitsHarq;
    tbPrms.nCsiReports   = uciPrms.nCsiReports;
    tbPrms.rankBitOffset = uciPrms.rankBitOffset;
    tbPrms.nRanksBits    = uciPrms.nRanksBits;
    tbPrms.csi2Flag      = (isCsi2Present > 0) ? 1 : 0;
}

//--------------------------------------------------------------------------------
// function takes as input pusch API parameters and derives them
inline void expandParameters(PerTbParams* pPerTbPrms, cuphyPuschStatPrms_t const* pStatPrm,
                             const cuphyPuschDynPrms_t* pDynPrm,
                             cuphyDerivedPuschCmnPrms& cmnPrms, cuphyDerivedPuschUeGrpPrms& ueGrpPrmsPrime,
                             cuphyLDPCParams& ldpcPrms, const uint8_t enableCsiP2Fapiv3)
{
    //extract cell, UE group, and UE parameters
    const cuphyPuschCellDynPrm_t    cellDynPrm    = pDynPrm->pCellGrpDynPrm->pCellPrms[0];
    const cuphyPuschCellGrpDynPrm_t cellGrpDynPrm = *(pDynPrm->pCellGrpDynPrm);
    const cuphyPuschUePrm_t*        uePrmsArray   = pDynPrm->pCellGrpDynPrm->pUePrms;

    cuphyPuschUeGrpPrm_t const* pUeGrpPrms = pDynPrm->pCellGrpDynPrm->pUeGrpPrms;

    // @todo: include these in per UE group processing
    cuphyPuschDmrsPrm_t const& dmrsPrm = *(pUeGrpPrms[0].pDmrsDynPrm);
    ueGrpPrmsPrime.nPrb                = pUeGrpPrms[0].nPrb;
    const uint8_t nDmrsCdmGrpsNoData         = dmrsPrm.nDmrsCdmGrpsNoData;

    // time allocation
    if(0 == pUeGrpPrms[0].dmrsSymLocBmsk) CUPHY_CHECK(CUPHY_STATUS_NOT_SUPPORTED);
    const uint32_t nTotalDmrsSyms = __builtin_popcount(pUeGrpPrms[0].dmrsSymLocBmsk);

    ueGrpPrmsPrime.nDataSymbols = static_cast<uint32_t>(pUeGrpPrms[0].nPuschSym) - nTotalDmrsSyms;
    ueGrpPrmsPrime.nDMRSSyms    = static_cast<uint32_t>(dmrsPrm.dmrsMaxLen); //does not include additional DMRS

    cmnPrms.chEst0DmrsSymLocBmsk = pUeGrpPrms[0].dmrsSymLocBmsk;
    cmnPrms.chEst1DmrsSymLocBmsk = 0;

    // The 2nd channel estimation and subsequent CFO estimation is enabled iff
    // - There is atleast one and only one additional DMRS position
    // - DMRS max length is 1
    // @todo: dmrsMaxLen > 1 to be supported

    // Using only the first and last symbols for the two channel estimates and subsequent CFO estimation
    // @todo: channel estimations with more than 2 DMRS additional positions to be supported
    if((1 == dmrsPrm.dmrsMaxLen) && (nTotalDmrsSyms > 1))
    {
        // Index of first pilot symbol - Find least significant set bit
        const int32_t firstPilotSymbPosIdx = __builtin_ctz(static_cast<uint32_t>(pUeGrpPrms[0].dmrsSymLocBmsk));
        cmnPrms.chEst0DmrsSymLocBmsk = static_cast<decltype(cmnPrms.chEst1DmrsSymLocBmsk)>(1) << firstPilotSymbPosIdx;

        // Index of last pilot symbol - Find most significant set bit
        // __builtin_clz expects a non-zero input, bitor with 1 gaurantees this result without affecting the result
        // chEst0DmrsSymLocBmsk is 16bit. __builtin_clz works with a min of 32bit inputs. So type convert input to 32bits and subtract 16bits.
        const int32_t nLeadingZeros = __builtin_clz(static_cast<uint32_t>(pUeGrpPrms[0].dmrsSymLocBmsk | 0x1)) - 16;
        // 16 bit number, -1 for zero indexing
        int32_t lastPilotSymbPosIdx  = 16 - nLeadingZeros - 1;
        cmnPrms.chEst1DmrsSymLocBmsk = static_cast<decltype(cmnPrms.chEst0DmrsSymLocBmsk)>(1) << lastPilotSymbPosIdx;
    }

    static constexpr uint16_t SLOT_SYMB_BMSK = (static_cast<decltype(pUeGrpPrms[0].rssiSymLocBmsk)>(1) << MAX_ND_SUPPORTED) - 1;
    cmnPrms.rssiSymLocBmsk                   = pStatPrm->enableRssiMeasurement ? (pUeGrpPrms[0].rssiSymLocBmsk & SLOT_SYMB_BMSK) : 0;

    // printf("nDataSymbols %d nDMRSSyms %d nTotalDmrsSyms %d chEst0DmrsSymLocBmsk 0x%0x chEst1DmrsSymLocBmsk 0x%0x rssiSymLocBmsk 0x%0x\n", ueGrpPrmsPrime.nDataSymbols, ueGrpPrmsPrime.nDMRSSyms, nTotalDmrsSyms, cmnPrms.chEst0DmrsSymLocBmsk, cmnPrms.chEst1DmrsSymLocBmsk, cmnPrms.rssiSymLocBmsk);

    auto* pDataSymLoc = static_cast<uint8_t*>(ueGrpPrmsPrime.tDataSymLocCpu.addr());
    auto* pDmrsSymLoc = static_cast<uint8_t*>(ueGrpPrmsPrime.tDmrsSymLocCpu.addr());
    uint8_t  dmrsCnt = 0, dataCnt = 0;
    for(uint8_t i = pUeGrpPrms[0].puschStartSym; i < (pUeGrpPrms[0].puschStartSym + pUeGrpPrms[0].nPuschSym); ++i)
    {
        if(1 & (pUeGrpPrms[0].dmrsSymLocBmsk >> i))
        {
            pDmrsSymLoc[dmrsCnt++] = i;
        }
        else
        {
            pDataSymLoc[dataCnt++] = i;
        }
    }

    // initialization/constants
    ueGrpPrmsPrime.nLayers                      = 0;
    uint16_t                  groupDmrsPortBmsk = 0;
    static constexpr uint16_t GRP_DMRS_PORT_MSK = (1U << 12) - 1; // DMRS port info in bit 0 to bit 11

    // @todo: this loop needs to be per UE group
    for(int i = 0; i < pUeGrpPrms[0].nUes; ++i)
    {
        // number of layers
        const uint16_t ueIdx = pUeGrpPrms[0].pUePrmIdxs[i];
        ueGrpPrmsPrime.nLayers += static_cast<uint32_t>(uePrmsArray[ueIdx].nUeLayers);

        // derive active dmrs grids
        groupDmrsPortBmsk = groupDmrsPortBmsk | (uePrmsArray[i].dmrsPortBmsk & GRP_DMRS_PORT_MSK);
    }

    // finish derivation of active dmrs grids
    const auto gridBmsk0                = static_cast<uint32_t>(((0x33 & groupDmrsPortBmsk) != 0) ? 1 : 0);
    const auto gridBmsk1                = static_cast<uint32_t>(((0xCC & groupDmrsPortBmsk) != 0) ? 1 : 0);
    ueGrpPrmsPrime.nDMRSGridsPerPRB   = 2;
    ueGrpPrmsPrime.activeDMRSGridBmsk = gridBmsk0 | (gridBmsk1 << 1);

    //
    // Loop over per-UE items
    //

    static constexpr uint32_t TBS_table[93] = {
        24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 208, 224,
        240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704,
        736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544,
        1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792,
        2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824
    };

    uint32_t totBitSize = 0;

    uint32_t layerCount[MAX_N_USER_GROUPS_SUPPORTED] = {0};

    for(int i = 0; i < cellGrpDynPrm.nUes; i++)
    {
        // HARQ parameters
        pPerTbPrms[i].ndi                      = uePrmsArray[i].ndi;
        pPerTbPrms[i].rv                       = uePrmsArray[i].rv;
        pPerTbPrms[i].debug_d_derateCbsIndices = uePrmsArray[i].debug_d_derateCbsIndices;

        // compute cinit seeds for descrambling
        pPerTbPrms[i].cinit = (static_cast<uint32_t>(((uePrmsArray[i].rnti << 15) + uePrmsArray[i].dataScramId)) & (0x7FFFFFFF));

        // find Qm and target code rate
        uint32_t tbSize{};

        pPerTbPrms[i].Qm = uePrmsArray[i].qamModOrder;
        const double codeRate         = uePrmsArray[i].targetCodeRate / (10.0f * 1024.0f);

        // Derive TB size and number of code blocks C(from derive_TB_size.m)

        // Compute number of REs
        const auto nUeLayers        = static_cast<uint32_t>(uePrmsArray[i].nUeLayers);
        const auto nPrb             = static_cast<uint32_t>(uePrmsArray[i].pUeGrpPrm->nPrb);
        const uint32_t Ndata            = 12 * nPrb * ueGrpPrmsPrime.nDataSymbols;
        const uint32_t Nre              = std::min(static_cast<uint32_t>(156), Ndata / nPrb) * nPrb;
        pPerTbPrms[i].encodedSize = Nre * pPerTbPrms[i].Qm * nUeLayers;

        // Compute number of info bits
        const float    Ninfo = Nre * codeRate * pPerTbPrms[i].Qm * nUeLayers;
        uint32_t Ninfo_prime{};

        if(Ninfo <= 3824)
        {
            // For "small" sizes, look up TBS in a table. First round the
            // number of information bits.
            const uint32_t n  = std::max(3, int(floor(log2(Ninfo)) - 6));
            Ninfo_prime = std::max(24, int(pow(2, n) * floor(Ninfo / pow(2, n))));

            // Pick the smallest TB from TBS_table which is larger than or equal to Ninfo_prime
            for(int j = 0; j < 93; j++)
            {
                if(Ninfo_prime <= TBS_table[j])
                {
                    tbSize = TBS_table[j];
                    break;
                }
            }
            pPerTbPrms[i].num_CBs = 1;
        }
        else
        {
            // For "large" sizes, compute TBS. First round the number of
            // information bits to a power of two.
            const uint32_t n  = floor(log2(static_cast<float>(Ninfo - 24))) - 5;
            Ninfo_prime = std::max(3840, int(pow(2, n) * round((double(Ninfo - 24.0) / pow(2, n)))));
            // printf("%d nre n %d, ninfo %d Ninfo_prime %d\n",Nre, n,Ninfo,Ninfo_prime);
            // Next, compute the number of code words. For large code rates,
            // use base-graph 1. For small code rate use base-graph 2.

            if(codeRate < 0.25)
            {
                const uint32_t C = div_round_up((Ninfo_prime + 24), static_cast<uint32_t>(3816));
                tbSize     = 8 * C * div_round_up((Ninfo_prime + 24), (8 * C)) - 24;
            }
            else
            {
                if(Ninfo_prime > 8424)
                {
                    const uint32_t C = div_round_up((Ninfo_prime + 24), static_cast<uint32_t>(8424));
                    tbSize     = 8 * C * div_round_up((Ninfo_prime + 24), (8 * C)) - 24;
                }
                else
                {
                    const uint32_t C = 1;
                    tbSize     = 8 * C * div_round_up((Ninfo_prime + 24), (8 * C)) - 24;
                }
            }
        }
        pPerTbPrms[i].tbSize = tbSize;

        // Derive BG (from derive_BGN.m) - 3GPP TS 38.212 Section 6.2.2
        pPerTbPrms[i].bg = cuphy::ldpc::derive_base_graph(tbSize, codeRate);

        // Derive codeblock size and number of filler bits

        // uint32_t polyBitSize = 24;
        // Max number of bits per codeblock
        uint32_t K_cb{};
        if(pPerTbPrms[i].bg == 1)
        {
            K_cb = 8448;
        }
        else
        {
            K_cb = 3840;
        }
        // Number of codeblocks
        uint32_t crcPolyByteSize{}, B{}, B_prime{};
        if(tbSize <= K_cb)
        {
            crcPolyByteSize       = tbSize <= 3824 ? 2 : 3;                     // CRC-16
            B                     = tbSize <= 3824 ? tbSize + 16 : tbSize + 24; // size of TB + TB-CRC
            pPerTbPrms[i].num_CBs = 1;                                          // number of CBs
            B_prime               = B;                                          // size of TB + TB-CRC + CB-CRCs
        }
        else
        {
            crcPolyByteSize       = 3;                              // CRC-24
            B                     = tbSize + 24;                    // size of TB + TB-CRC
            pPerTbPrms[i].num_CBs = div_round_up(B, (K_cb - 24));   // number of CBs
            B_prime               = B + pPerTbPrms[i].num_CBs * 24; // size of TB + TB-CRC + CB-CRCs
        }

        // Bits per code block
        uint32_t K_prime = B_prime / pPerTbPrms[i].num_CBs;

        // Derive Kb and Zc - 3GPP TS 38.212 Section 5.2.2
        ldpcPrms.KbArray[i] = cuphy::ldpc::derive_Kb(pPerTbPrms[i].bg, B);
        pPerTbPrms[i].Zc    = cuphy::ldpc::derive_Zc(ldpcPrms.KbArray[i], K_prime);

        // Derive K (codeblock size) and F (number of filler bits)

        if(pPerTbPrms[i].bg == 1)
        {
            pPerTbPrms[i].K = pPerTbPrms[i].Zc * 22;
        }
        else
        {
            pPerTbPrms[i].K = pPerTbPrms[i].Zc * 10;
        }

        pPerTbPrms[i].F = pPerTbPrms[i].K - K_prime;

        // Derive startIdx

        // Derive E_vec - rate-matched code block sizes

        // Fill out output parameter structure TB-specific
        // Back-end
        const uint16_t ueGrpIdx = uePrmsArray[i].ueGrpIdx;

        // Calculate Ncb
        uint32_t Ncb = pPerTbPrms[i].bg == 1 ? pPerTbPrms[i].Zc * 66 : pPerTbPrms[i].Zc * 50;
        if(uePrmsArray[i].i_lbrm == 1)
        {
            static constexpr float R_LBRM      = 2. / 3.;
            static constexpr float maxRate     = 948. / 1024.;
            static constexpr int   num_symbols = 156 / 12;
            uint32_t    TBS_LBRM{}, num_CBs_unused{};
            get_TB_size_and_num_CBs(
                num_symbols,               // int num_symbols,
                uePrmsArray[i].n_PRB_LBRM, // int num_prbs,
                uePrmsArray[i].maxLayers,  // int num_layers,
                maxRate,                   // float code_rate,
                uePrmsArray[i].maxQm,      // uint32_t Qm,
                num_CBs_unused,            // uint32_t &num_cBS
                TBS_LBRM,                  // uint32_t &tb_size
                0);                        // int num_dmrs_cdmGrpsNoData1_symbols; set to 0 for PUSCH

            const uint32_t Nref = floor(TBS_LBRM / (pPerTbPrms[i].num_CBs * R_LBRM));
            if(Nref < Ncb) Ncb = Nref;
        }

        pPerTbPrms[i].Ncb                 = Ncb;
        pPerTbPrms[i].firstCodeBlockIndex = (0); //NEEDS FIX, input tbStructs will have to contain symbol-by-symbol processing info
        pPerTbPrms[i].userGroupIndex      = static_cast<uint32_t>(ueGrpIdx);
        pPerTbPrms[i].nBBULayers          = ueGrpPrmsPrime.nLayers;
        pPerTbPrms[i].Nl                  = nUeLayers;
        pPerTbPrms[i].startLLR            = static_cast<uint32_t>(cellGrpDynPrm.pUeGrpPrms[ueGrpIdx].startPrb * 12 * QAM_STRIDE * ueGrpPrmsPrime.nLayers * ueGrpPrmsPrime.nDataSymbols);

        // Calculate padding for codeblock LLRs
        uint32_t Ncb_padded = pPerTbPrms[i].Ncb + 2 * pPerTbPrms[i].Zc;
        Ncb_padded          = (Ncb_padded + 7) / 8;
        Ncb_padded *= 8;
        pPerTbPrms[i].Ncb_padded = Ncb_padded;

        for(int l = 0; l < nUeLayers; l++)
        {
            pPerTbPrms[i].layer_map_array[l] = layerCount[ueGrpIdx];
            layerCount[ueGrpIdx]++;
        }

        const uint32_t Kd = pPerTbPrms[i].K - pPerTbPrms[i].F - 2 * pPerTbPrms[i].Zc;

        if(pPerTbPrms[i].bg == 1)
        {
            const uint32_t rv  = pPerTbPrms[i].rv;
            const uint32_t Zc  = pPerTbPrms[i].Zc;
            const uint32_t Ncb = pPerTbPrms[i].Ncb;
            uint32_t k0{};
            if(rv == 0)
            {
                k0 = 0;
            }
            else if(rv == 1)
            {
                k0 = (17 * Ncb / (66 * Zc)) * Zc;
            }
            else if(rv == 2)
            {
                k0 = (33 * Ncb / (66 * Zc)) * Zc;
            }
            else if(rv == 3)
            {
                k0 = (56 * Ncb / (66 * Zc)) * Zc;
            }
            const uint32_t Ncb_forparity       = std::min<uint32_t>(pPerTbPrms[i].encodedSize / pPerTbPrms[i].num_CBs + k0, Ncb);
            ldpcPrms.parityNodesArray[i] = std::max<uint32_t>(4, std::min<uint32_t>(CUPHY_LDPC_MAX_BG1_PARITY_NODES, (Ncb_forparity - Kd + 1) / Zc));
            //printf("BG=%u encodedSize=%u Zc=%u Ncb=%u k0=%u num_CBs=%u Kd=%u parityNodes=%u\n",pPerTbPrms[i].bg,pPerTbPrms[i].encodedSize,Zc,Ncb,k0,pPerTbPrms[i].num_CBs,Kd,ldpcPrms.parityNodesArray[i]);
        }
        else
        {
            const uint32_t rv  = pPerTbPrms[i].rv;
            const uint32_t Zc  = pPerTbPrms[i].Zc;
            const uint32_t Ncb = pPerTbPrms[i].Ncb;
            uint32_t k0{};
            if(rv == 0)
            {
                k0 = 0;
            }
            else if(rv == 1)
            {
                k0 = (13 * Ncb / (50 * Zc)) * Zc;
            }
            else if(rv == 2)
            {
                k0 = (25 * Ncb / (50 * Zc)) * Zc;
            }
            else if(rv == 3)
            {
                k0 = (43 * Ncb / (50 * Zc)) * Zc;
            }
            const uint32_t Ncb_forparity       = std::min<uint32_t>(pPerTbPrms[i].encodedSize / pPerTbPrms[i].num_CBs + k0, Ncb);
            ldpcPrms.parityNodesArray[i] = std::max<uint32_t>(4, std::min<uint32_t>(CUPHY_LDPC_MAX_BG2_PARITY_NODES, (Ncb_forparity - Kd + 1) / Zc));
        }

        pPerTbPrms[i].nZpBitsPerCb = (ldpcPrms.parityNodesArray[i] + ldpcPrms.KbArray[i]) * pPerTbPrms[i].Zc;

        const uint32_t codeBlockDataByteSize = (pPerTbPrms[i].K - crcPolyByteSize * 8 - pPerTbPrms[i].F + 8 - 1) / 8;

        const uint32_t decodedTbSize   = codeBlockDataByteSize * pPerTbPrms[i].num_CBs;
        pPerTbPrms[i].nDataBytes = decodedTbSize;

        totBitSize += decodedTbSize * 8;
        rate_match_seq_len(pPerTbPrms[i], uePrmsArray[i], cellDynPrm, pUeGrpPrms[ueGrpIdx], codeRate, dataCnt, pDataSymLoc, dmrsCnt, pDmrsSymLoc, nDmrsCdmGrpsNoData, enableCsiP2Fapiv3);

        // flag if uci on pusch present
        pPerTbPrms[i].uciOnPuschFlag = static_cast<uint8_t>((uePrmsArray[i].pduBitmap >> 1) & 1);
    }

    //
    // Loop over per-UE-group items
    //
    // frequency allocations
    auto* pStartPrbCpu = static_cast<uint16_t*>(ueGrpPrmsPrime.tStartPrbCpu.addr());
    auto* pNumPrbCpu   = static_cast<uint16_t*>(ueGrpPrmsPrime.tNumPrbCpu.addr());
    auto* pDmrsScIdCpu = static_cast<uint16_t*>(ueGrpPrmsPrime.tDmrsScIdCpu.addr());
    uint32_t  nMaxPrb      = 0;
    for(int i = 0; i < cellGrpDynPrm.nUeGrps; ++i)
    {
        pStartPrbCpu[i] = pUeGrpPrms[i].startPrb;
        pNumPrbCpu[i]   = pUeGrpPrms[i].nPrb;
        pDmrsScIdCpu[i] = pUeGrpPrms[i].pDmrsDynPrm->dmrsScrmId;

        if(nMaxPrb < pNumPrbCpu[i]) nMaxPrb = pNumPrbCpu[i];

        for(int j = 0; j < pUeGrpPrms[i].nUes; ++j)
        {
            const uint16_t ueIdx = pUeGrpPrms[i].pUePrmIdxs[j];
            const auto  Qm    = static_cast<uint8_t>(pPerTbPrms[ueIdx].Qm);
            for(int k = 0; k < pPerTbPrms[ueIdx].Nl; ++k)
            {
                ueGrpPrmsPrime.tQamInfoCpu(pPerTbPrms[ueIdx].layer_map_array[k], i) = Qm;
            }
        }
    }
    cmnPrms.nMaxPrb = nMaxPrb;
}

#endif //PUSCH_UTILS_HPP_INCLUDED_
