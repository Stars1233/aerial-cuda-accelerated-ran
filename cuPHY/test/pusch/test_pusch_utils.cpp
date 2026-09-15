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

#include "cuphy.h"
#include "ldpc/ldpc_params.hpp"
#include "pusch_utils.hpp"

namespace
{

TEST(PuschDescriptorSizing, OmitsEarlyRateMatchDescriptorForFullTbDecoder)
{
    constexpr size_t rateMatchDescrSizeBytes  = 4240;
    constexpr size_t rateMatchDescrAlignBytes = 8;

    const auto earlyDescrInfo = getEarlyRateMatchDescrInfo(
        false, rateMatchDescrSizeBytes, rateMatchDescrAlignBytes);

    EXPECT_EQ(earlyDescrInfo.sizeBytes, 0);
    EXPECT_EQ(earlyDescrInfo.alignBytes, 0);
}

TEST(PuschDescriptorSizing, RetainsEarlyRateMatchDescriptorForCbDecoder)
{
    constexpr size_t rateMatchDescrSizeBytes  = 4240;
    constexpr size_t rateMatchDescrAlignBytes = 8;

    const auto earlyDescrInfo = getEarlyRateMatchDescrInfo(
        true, rateMatchDescrSizeBytes, rateMatchDescrAlignBytes);

    EXPECT_EQ(earlyDescrInfo.sizeBytes, rateMatchDescrSizeBytes);
    EXPECT_EQ(earlyDescrInfo.alignBytes, rateMatchDescrAlignBytes);
}

TEST(EarlySchCbDecodePlan, CountsOnlyCompleteLeadingCodeBlocks)
{
    PerTbParams tb{};
    tb.num_CBs        = 3;
    tb.Nl             = 1;
    tb.Qm             = 2;
    tb.encodedSize    = 20;
    tb.uciOnPuschFlag = 0;

    // Per-CB rate-matched lengths are [6, 6, 8]. Only leading CBs that are
    // completely covered by early symbols are eligible for early decode.
    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 5), 0);
    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 6), 1);
    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 12), 2);
    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 19), 2);
    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 20), 3);
}

TEST(EarlySchCbDecodePlan, CountsCompleteLeadingUciCodeBlocksUsingRateMatchPartition)
{
    PerTbParams tb{};
    tb.num_CBs        = 3;
    tb.Nl             = 1;
    tb.Qm             = 2;
    tb.G              = 20;
    tb.encodedSize    = 18;
    tb.uciOnPuschFlag = 1;

    // The UCI path uses G, not encodedSize, to derive q1. Its rate-match E partition is [6, 6, 8].
    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 18), 2);
}

TEST(EarlySchCbDecodePlan, UsesUciRateMatchBudgetWhenEncodedSizeIsZero)
{
    PerTbParams tb{};
    tb.num_CBs        = 3;
    tb.Nl             = 1;
    tb.Qm             = 2;
    tb.G              = 20;
    tb.uciOnPuschFlag = 1;

    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 20), 3);
}

TEST(EarlySchCbDecodePlan, ReturnsZeroWithoutSchedulableEarlyBits)
{
    PerTbParams tb{};
    tb.num_CBs     = 3;
    tb.Nl          = 1;
    tb.Qm          = 2;
    tb.encodedSize = 20;

    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 0), 0);

    tb.num_CBs = 0;
    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 20), 0);

    tb.num_CBs = 3;
    tb.Qm      = 0;
    EXPECT_EQ(computeLeadingEarlySchCbCount(tb, 20), 0);
}

TEST(EarlySchCbDecodePlan, SkipsEarlyBatchPreparationWithoutEarlyCodeblocks)
{
    EXPECT_FALSE(needsEarlyCbBatchPreparation(0));
    EXPECT_TRUE(needsEarlyCbBatchPreparation(1));
}

TEST(EarlySchCbDecodePlan, ExposesDedicatedSubslotSchedulingSignal)
{
    cuphyPuschDataOut_t out{};

    out.isEarlySchCbDecodePresent = 1;

    EXPECT_EQ(out.isEarlySchCbDecodePresent, 1);
    EXPECT_EQ(out.isEarlyHarqPresent, 0);
}

TEST(FullSlotSchDecodePlan, SkipsRateMatchAndLdpcWhenAllCbWorkCompletedEarly)
{
    EXPECT_TRUE(needsFullSlotSchDecode(false, 0));
    EXPECT_TRUE(needsFullSlotSchDecode(true, 1));
    EXPECT_FALSE(needsFullSlotSchDecode(true, 0));
}

TEST(PuschStatus, ExposesLdpcSetupFailure)
{
    EXPECT_NE(cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_LDPC_SETUP_ERROR,
              cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE);
}

TEST(LdpcParams, Bg2SmallBlockSystematicSizeUsesTenColumns)
{
    const auto kb6 = cuphy::ldpc::derive_ldpc_params(100, 0.5);
    EXPECT_EQ(kb6.bg, 2);
    EXPECT_EQ(kb6.Kb, 6);
    EXPECT_EQ(kb6.Zc, 20);
    EXPECT_EQ(kb6.K, 200);
    EXPECT_EQ(kb6.F, 84U);

    const auto kb8 = cuphy::ldpc::derive_ldpc_params(250, 0.5);
    EXPECT_EQ(kb8.bg, 2);
    EXPECT_EQ(kb8.Kb, 8);
    EXPECT_EQ(kb8.Zc, 36);
    EXPECT_EQ(kb8.K, 360);
    EXPECT_EQ(kb8.F, 94U);

    const auto kb9 = cuphy::ldpc::derive_ldpc_params(550, 0.5);
    EXPECT_EQ(kb9.bg, 2);
    EXPECT_EQ(kb9.Kb, 9);
    EXPECT_EQ(kb9.Zc, 64);
    EXPECT_EQ(kb9.K, 640);
    EXPECT_EQ(kb9.F, 74U);
}

TEST(LdpcParams, LargeBg1TransportBlockSegmentsAtCodeblockLimit)
{
    const auto params = cuphy::ldpc::derive_ldpc_params(8440, 0.9);

    EXPECT_EQ(params.bg, 1);
    EXPECT_EQ(params.nCb, 2U);
}

} // namespace
