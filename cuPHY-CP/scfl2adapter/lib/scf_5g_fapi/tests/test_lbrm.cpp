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

/**
 * @file test_lbrm.cpp
 * @brief Unit tests for the shared LBRM helpers in scf_5g_fapi_lbrm.hpp.
 *
 * Covers:
 *   Group NPRB — compute_n_prb_lbrm(): N_PRB_LBRM bucket lookup, every
 *                boundary of TS 38.212 Table 5.4.2.1-2 (expected values taken
 *                directly from the spec table, not recomputed by the test).
 *   Group QM   — compute_max_qm(): 256-QAM vs 64-QAM maxQm derivation.
 *   Group LAYERS — compute_max_layers(): antenna-count clamp to k_max_layers.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "scf_5g_fapi_lbrm.hpp"

namespace {

// ---------------------------------------------------------------------------
// Group NPRB — compute_n_prb_lbrm()
// ---------------------------------------------------------------------------

// TS 38.212 Table 5.4.2.1-2 maps a BWP size (PRBs) to one of the seven
// N_PRB_LBRM buckets {32,66,107,135,162,217,273}: the smallest table entry
// that is >= bwp_size.  bwp_size beyond the largest entry (273) saturates at
// 273.  Expected values below are read straight off the spec table.
struct NPrbCase
{
    const char* name;                 //!< Test case name (used as the GTest suffix).
    uint16_t    bwp_size;             //!< BWP bandwidth in PRBs (input to compute_n_prb_lbrm).
    uint16_t    expected_n_prb_lbrm;  //!< Expected N_PRB_LBRM bucket (TS 38.212 Table 5.4.2.1-2).
};

class LbrmNPrbTest : public ::testing::TestWithParam<NPrbCase> {};

TEST_P(LbrmNPrbTest, MatchesSpecTable)
{
    const auto& c = GetParam();
    EXPECT_EQ(scf_5g_fapi::lbrm::compute_n_prb_lbrm(c.bwp_size), c.expected_n_prb_lbrm);
}

INSTANTIATE_TEST_SUITE_P(
    Boundaries, LbrmNPrbTest,
    ::testing::Values(
        //        name                  bwp_size  expected
        NPrbCase{"Zero_Bucket32",            0u,   32u},
        NPrbCase{"Edge32_Bucket32",         32u,   32u},
        NPrbCase{"Above32_Bucket66",        33u,   66u},
        NPrbCase{"Edge66_Bucket66",         66u,   66u},
        NPrbCase{"Above66_Bucket107",       67u,  107u},
        NPrbCase{"Edge107_Bucket107",      107u,  107u},
        NPrbCase{"Above107_Bucket135",     108u,  135u},
        NPrbCase{"Edge135_Bucket135",      135u,  135u},
        NPrbCase{"Above135_Bucket162",     136u,  162u},
        NPrbCase{"Edge162_Bucket162",      162u,  162u},
        NPrbCase{"Above162_Bucket217",     163u,  217u},
        NPrbCase{"Edge217_Bucket217",      217u,  217u},
        NPrbCase{"Above217_Bucket273",     218u,  273u},
        NPrbCase{"Edge273_Bucket273",      273u,  273u},
        NPrbCase{"Above273_Saturates273",  274u,  273u},
        NPrbCase{"Max_Saturates273",     65535u,  273u}),
    [](const ::testing::TestParamInfo<NPrbCase>& info) {
        return std::string(info.param.name);
    });

// ---------------------------------------------------------------------------
// Group QM — compute_max_qm()
// ---------------------------------------------------------------------------

// mcs_table == 1 selects the 256-QAM table (maxQm 8); every other table index
// is 64-QAM or lower (maxQm 6).  See TS 38.214 Table 5.1.3.1-2 / 6.1.4.1-2.
struct MaxQmCase
{
    const char* name;             //!< Test case name (used as the GTest suffix).
    uint8_t     mcs_table;        //!< SCF FAPI mcs_table index (input to compute_max_qm).
    uint8_t     expected_max_qm;  //!< Expected maxQm modulation order (8 for 256-QAM, else 6).
};

class LbrmMaxQmTest : public ::testing::TestWithParam<MaxQmCase> {};

TEST_P(LbrmMaxQmTest, MatchesModulationOrder)
{
    const auto& c = GetParam();
    EXPECT_EQ(scf_5g_fapi::lbrm::compute_max_qm(c.mcs_table), c.expected_max_qm);
}

INSTANTIATE_TEST_SUITE_P(
    Tables, LbrmMaxQmTest,
    ::testing::Values(
        //         name             mcs_table  expected
        MaxQmCase{"Table0_64qam",        0u,    6u},
        MaxQmCase{"Table1_256qam",       1u,    8u},
        MaxQmCase{"Table2_64qam",        2u,    6u},
        MaxQmCase{"TableMax_64qam",    255u,    6u}),
    [](const ::testing::TestParamInfo<MaxQmCase>& info) {
        return std::string(info.param.name);
    });

// ---------------------------------------------------------------------------
// Group LAYERS — compute_max_layers()
// ---------------------------------------------------------------------------

struct MaxLayersCase
{
    std::string_view name;
    uint16_t         num_ant;
    uint8_t          expected_max_layers;
};

class LbrmMaxLayersTest : public ::testing::TestWithParam<MaxLayersCase> {};

TEST_P(LbrmMaxLayersTest, ClampsToSpecMaximum)
{
    const auto& c = GetParam();
    EXPECT_EQ(scf_5g_fapi::lbrm::compute_max_layers(c.num_ant), c.expected_max_layers);
}

INSTANTIATE_TEST_SUITE_P(
    AntennaCounts, LbrmMaxLayersTest,
    ::testing::Values(
        //               name              num_ant  expected
        MaxLayersCase{"TwoTR",                  2u, 2u},
        MaxLayersCase{"FourTR",                 4u, 4u},
        MaxLayersCase{"EightTRClamped",         8u, 4u},
        MaxLayersCase{"SixtyFourTRClamped",    64u, 4u}),
    [](const ::testing::TestParamInfo<MaxLayersCase>& info) {
        return std::string(info.param.name);
    });

// ---------------------------------------------------------------------------
// Compile-time contract: both helpers are inline constexpr and must be usable
// in a constant-evaluated context (mirrors their use in cell_dyn_info init).
// ---------------------------------------------------------------------------

static_assert(scf_5g_fapi::lbrm::compute_n_prb_lbrm(0u) == 32u);
static_assert(scf_5g_fapi::lbrm::compute_n_prb_lbrm(59u) == 66u);
static_assert(scf_5g_fapi::lbrm::compute_n_prb_lbrm(273u) == 273u);
static_assert(scf_5g_fapi::lbrm::compute_n_prb_lbrm(300u) == 273u);
static_assert(scf_5g_fapi::lbrm::compute_max_qm(scf_5g_fapi::lbrm::k_mcs_table_256qam)
              == scf_5g_fapi::lbrm::k_max_qm_256qam);
static_assert(scf_5g_fapi::lbrm::compute_max_qm(0u) == scf_5g_fapi::lbrm::k_max_qm_64qam);
static_assert(scf_5g_fapi::lbrm::compute_max_layers(2u) == 2u);
static_assert(scf_5g_fapi::lbrm::compute_max_layers(64u) == scf_5g_fapi::lbrm::k_max_layers);

} // namespace
