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

if (NOT TARGET GTest::gtest)
    message(STATUS "testmac_core tests: GTest not found, skipping test_testmac_core")
    return()
endif ()

add_executable(test_testmac_core
    test_testmac_core.cpp)

# testmac_core propagates all required includes and libs via its PUBLIC links.
# sdk::warnings (–Werror) is intentionally omitted: test_testmac_core.cpp
# directly includes cuphy_hdf5.hpp / cuphy.hpp which carry pre-existing
# sign-conversion and unused-parameter warnings. Re-add sdk::warnings once
# those upstream headers are clean.
target_link_libraries(test_testmac_core PRIVATE
    testmac_core
    sdk::options)

target_include_directories(test_testmac_core PRIVATE
    ${CMAKE_SOURCE_DIR}/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi
    ${CMAKE_SOURCE_DIR}/cuPHY-CP/gt_common_libs/slot_command/include
    ${CMAKE_SOURCE_DIR}/cuPHY-CP/aerial-fh-driver/include)

# GTest via target_link_system_libraries so its headers are treated as system
# includes and don't trigger project warning flags.
target_link_system_libraries(test_testmac_core PRIVATE GTest::gtest)

if (DEFINED ENV{cuBB_SDK})
    set(CUBB_HOME "$ENV{cuBB_SDK}")
else ()
    get_filename_component(CUBB_HOME "${CMAKE_SOURCE_DIR}" ABSOLUTE)
endif ()

add_test(
    NAME testmac_core.copy_tv
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/copy_testmac_ctest_tv.sh"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test(
    NAME testmac_core.unit_tests
    COMMAND test_testmac_core
        --gtest_catch_exceptions=0
        --gtest_output=xml:test_testmac_core.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_tests_properties(testmac_core.copy_tv PROPERTIES
    FIXTURES_SETUP testmac_core_tv
    LABELS "cuphy-cp;requires-tvs"
    ENVIRONMENT "cuBB_SDK=${CUBB_HOME}")

set_tests_properties(testmac_core.unit_tests PROPERTIES
    FIXTURES_REQUIRED testmac_core_tv
    LABELS "cuphy-cp;requires-tvs"
    ENVIRONMENT "cuBB_SDK=${CUBB_HOME}")
