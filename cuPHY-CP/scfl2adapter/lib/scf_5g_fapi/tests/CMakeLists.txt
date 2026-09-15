# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# scf_5g_fapi unit tests — single executable covering all test suites.
# ---------------------------------------------------------------------------

# These tests use FAPI-10.04-only fields (nPDUsOfEachType, DL_TTI_NPDUS_IDX_*),
# so they only compile when SCF_FAPI_10_04 is enabled.
if(ENABLE_TESTS AND SCF_FAPI_10_04)

add_executable(test_scfl2adapter
    test_tti_dispatch.cpp
    test_pdsch_pdu_parser.cpp
    test_ssb_pdu_parser.cpp
    test_pucch_pdu_parser.cpp
    test_prach_pdu_parser.cpp
    test_prach_dop.cpp
    test_prach_module_view.cpp
    test_srs_pdu_parser.cpp
    test_pdcch_pdu_parser.cpp
    test_uci_config_parser.cpp
    test_pusch_ue_grouping.cpp
    test_lbrm.cpp
    test_stubs.cpp
    test_slot_config_tlv_parser.cpp
    # Parser under test compiled directly: it lives in the heavyweight scf_5g_fapi
    # sources, so pull just this translation unit in rather than linking that target.
    ${CMAKE_CURRENT_SOURCE_DIR}/../scf_5g_fapi_slot_config_tlv_parser.cpp)

target_include_directories(test_scfl2adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    # nvPHY public headers (nv_phy_mac_transport.hpp, nv_phy_config_option.hpp, etc.)
    # Not linking against the full nvphy target (avoids CUDA/gRPC) — just the include path.
    ${PROJECT_SOURCE_DIR}/../cuphyl2adapter/lib/nvPHY)

target_link_libraries(test_scfl2adapter PRIVATE
    scf_5g_fapi_uci_config_parser
    scf::scf_5g_fapi_h
    slot_command
    nvlog_static)

target_link_system_libraries(test_scfl2adapter PRIVATE
    GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_scfl2adapter PROPERTIES LABELS cuphy-cp)

endif()
