# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set (SOURCES test_scf_5g_fapi.cpp ${PROJECT_SOURCE_DIR}/dummy.cu)
add_executable(test_scf_5g_fapi ${SOURCES})
target_link_libraries(test_scf_5g_fapi PRIVATE nvphy)

# decode_reconfig_tlvs — shared reconfig TLV decoder used by both the serial
# (on_config_request) and offload (on_config_request_offload) paths. The decoder
# lives in the scf_5g_fapi lib; nv::cell_update_config (its output type) pulls in
# nvPHY + cuphy headers, so we link the same propagating deps as the CSI-RS test.
add_executable(test_config_tlv_decode test_config_tlv_decode.cpp)
# sdk::warnings intentionally omitted: scf_5g_fapi.h uses GNU zero-length arrays
# that trigger -Werror=pedantic (same rationale as test_scfl2_csirs_slot_helpers).
target_link_libraries(test_config_tlv_decode PRIVATE
    scf_5g_fapi
    nvphy_generic
    cuphydriver
    cuphy_cpp_header_only
    slot_command
    aerial_common
    sdk::options)
target_link_system_libraries(test_config_tlv_decode PRIVATE nvlog_static GTest::gtest_main)
target_compile_options(test_config_tlv_decode PRIVATE -Wno-unknown-pragmas)
target_include_directories(test_config_tlv_decode PRIVATE
    ${PROJECT_SOURCE_DIR}/cuphyl2adapter/lib/nvPHY)
add_test(
    NAME scfl2adapter_config_tlv_decode.unit_tests
    COMMAND test_config_tlv_decode --gtest_output=xml:test_config_tlv_decode.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(scfl2adapter_config_tlv_decode.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")

# CSI-RS slot helpers — standalone GTests. Depend on scf_5g_csirs_helpers,
# which is only built under ENABLE_FAPI_STORE_REPLAY.
if(ENABLE_FAPI_STORE_REPLAY)
    add_executable(test_scfl2_csirs_slot_helpers test_csirs_slot_helpers.cpp)
    # sdk::warnings intentionally omitted: transitively includes scf_5g_fapi.h
    # (GNU zero-length arrays trigger -Werror=pedantic). See scf_5g_csirs_helpers.
    target_link_libraries(test_scfl2_csirs_slot_helpers PRIVATE
        scf_5g_csirs_helpers
        sdk::options)
    target_link_system_libraries(test_scfl2_csirs_slot_helpers PRIVATE GTest::gtest_main)

    add_test(
        NAME scfl2adapter_csirs_helpers.unit_tests
        COMMAND test_scfl2_csirs_slot_helpers --gtest_output=xml:test_scfl2_csirs_slot_helpers.xml
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(scfl2adapter_csirs_helpers.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")

    # SSB precoding-matrix cache tests. Same shape as the CSI-RS block above, and a
    # separate executable for the same reason: the code under test
    # (update_pm_weights_ssb_cuphy, in scf_5g_slot_commands_ssb_pbch.cpp) needs the real
    # scf_5g_ssb_helpers archive, whereas test_scfl2adapter is stub-based by design --
    # test_stubs.cpp supplies its own update_cell_command / update_beam_list /
    # phy_mac_transport precisely to avoid that dependency chain, so linking the real
    # archive there produces multiple-definition and undefined-reference errors.
    # Unlike scf_5g_csirs_helpers, the SSB archive is NOT self-contained:
    # update_pm_weights_ssb_cuphy() reaches nv::PHYDriverProxy (nv_phy_driver_proxy.cpp,
    # in nvphy) and update_fh_params_ssb() reaches check_prb_info_size() /
    # update_beam_list() (scf_5g_slot_commands_common.cpp, in scf_5g_fapi).
    #
    # Those three archives are mutually dependent, so a single left-to-right pass
    # cannot resolve them: CMake orders scf_5g_fapi before scf_5g_ssb_helpers (the
    # former links the latter), but it is scf_5g_ssb_helpers that *introduces* the
    # undefined references, by which point the defining archives are already behind
    # the linker's cursor. Wrap them in a rescanning group (ld --start-group) so the
    # linker re-walks the set until it reaches a fixed point. Same hazard the
    # scf_5g_ssb_helpers comment in lib/scf_5g_fapi/CMakeLists.txt calls out.
    add_executable(test_scfl2_ssb_pm_weights test_ssb_pm_weights.cpp)
    # sdk::warnings intentionally omitted: transitively includes scf_5g_fapi.h
    # (GNU zero-length arrays trigger -Werror=pedantic). See scf_5g_ssb_helpers.
    target_link_libraries(test_scfl2_ssb_pm_weights PRIVATE
        "$<LINK_GROUP:RESCAN,scf_5g_ssb_helpers,scf_5g_fapi,nvphy>"
        sdk::options)
    target_link_system_libraries(test_scfl2_ssb_pm_weights PRIVATE GTest::gtest_main)
    # No extra target_include_directories / -Wno-unknown-pragmas: nvPHY headers arrive
    # via PUBLIC deps of the RESCAN archives (same as test_scfl2_csirs_slot_helpers).

    add_test(
        NAME scfl2adapter_ssb_pm_weights.unit_tests
        COMMAND test_scfl2_ssb_pm_weights --gtest_output=xml:test_scfl2_ssb_pm_weights.xml
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(scfl2adapter_ssb_pm_weights.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")
endif()
