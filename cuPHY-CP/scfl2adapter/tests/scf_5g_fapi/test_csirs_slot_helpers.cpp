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

#include "scf_5g_csirs_slot_command_helpers.hpp"
#include "scf_5g_fapi.h"
#include "cuphy.h"

TEST(CsirsSlotHelpers, ZpPolicyUsesPdschPduCountNotLegacyFlag)
{
    EXPECT_TRUE(scf_5g_fapi::should_skip_zp_csirs_without_pdsch(cuphyCsiType_t::ZP_CSI_RS, false));
    EXPECT_FALSE(scf_5g_fapi::should_skip_zp_csirs_without_pdsch(cuphyCsiType_t::ZP_CSI_RS, true));
    EXPECT_FALSE(scf_5g_fapi::should_skip_zp_csirs_without_pdsch(cuphyCsiType_t::NZP_CSI_RS, false));
}

// NOTE: The DL_TTI fast-path skip predicate (for_each_tti_msg in
// nv_fapi_pdu_utils.hpp) lives in the cuphyl2adapter/nvphy library, which
// this test executable does not link against (it only links scf_5g_csirs_helpers).
// The previous tautology test that wrote/read nPDUsOfEachType[CSI_RS] was
// removed; the real assertion lives in the nvphy-side test:
//   cuPHY-CP/cuphyl2adapter/lib/nvPHY/tests/test_csirs_slot_helpers.cpp
//     -> ForEachTtiMsgSkipsDlTtiWithoutCsirsPdus
// and additionally:
//   cuPHY-CP/cuphyl2adapter/lib/nvPHY/tests/test_fapi_message_storage.cpp
//     -> ForEachTtiMsgSkipsDlTtiWithZeroPduCount
