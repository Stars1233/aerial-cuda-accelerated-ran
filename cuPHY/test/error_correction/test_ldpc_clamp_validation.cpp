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

#include "ldpc_clamp_validation.hpp"

TEST(LdpcClampValidation, FiniteLimitMatchesTheSelectedLlrType)
{
    EXPECT_EQ(CUPHY_FP8_E4M3_MAX_FINITE, 448.0f);
    EXPECT_EQ(CUPHY_FP8_E5M2_MAX_FINITE, 57344.0f);
    EXPECT_EQ(ldpc_example::max_finite_clamp_value(CUPHY_R_8F_E4M3), CUPHY_FP8_E4M3_MAX_FINITE);
    EXPECT_EQ(ldpc_example::max_finite_clamp_value(CUPHY_R_8F_E5M2), CUPHY_FP8_E5M2_MAX_FINITE);
    EXPECT_EQ(ldpc_example::max_finite_clamp_value(CUPHY_R_16F), 65504.0f);
    EXPECT_EQ(ldpc_example::max_finite_clamp_value(CUPHY_R_32F), 65504.0f);
}

TEST(LdpcClampValidation, RejectsE4M3AtItsFiniteLimit)
{
    EXPECT_FALSE(ldpc_example::is_valid_clamp_value(CUPHY_R_8F_E4M3, 448.0f));
}

TEST(LdpcClampValidation, AcceptsE4M3BelowItsFiniteLimit)
{
    EXPECT_TRUE(ldpc_example::is_valid_clamp_value(CUPHY_R_8F_E4M3, 447.0f));
}

TEST(LdpcClampValidation, RejectsE5M2AtItsFiniteLimit)
{
    EXPECT_FALSE(ldpc_example::is_valid_clamp_value(CUPHY_R_8F_E5M2, 57344.0f));
}

TEST(LdpcClampValidation, AcceptsE5M2BelowItsFiniteLimit)
{
    EXPECT_TRUE(ldpc_example::is_valid_clamp_value(CUPHY_R_8F_E5M2, 57343.0f));
}

TEST(LdpcClampValidation, UsesFp16RangeForFp16AndFp32)
{
    EXPECT_TRUE(ldpc_example::is_valid_clamp_value(CUPHY_R_16F, 65503.0f));
    EXPECT_TRUE(ldpc_example::is_valid_clamp_value(CUPHY_R_32F, 65503.0f));
    EXPECT_FALSE(ldpc_example::is_valid_clamp_value(CUPHY_R_16F, 65504.0f));
    EXPECT_FALSE(ldpc_example::is_valid_clamp_value(CUPHY_R_32F, 65504.0f));
}

TEST(LdpcClampValidation, RejectsNonPositiveAndUnsupportedValues)
{
    EXPECT_FALSE(ldpc_example::is_valid_clamp_value(CUPHY_R_8F_E4M3, 0.0f));
    EXPECT_FALSE(ldpc_example::is_valid_clamp_value(CUPHY_R_8F_E5M2, -1.0f));
    EXPECT_FALSE(ldpc_example::is_valid_clamp_value(CUPHY_BIT, 1.0f));
}
