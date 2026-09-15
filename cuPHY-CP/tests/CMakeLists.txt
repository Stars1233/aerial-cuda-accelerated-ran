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

cmake_minimum_required (VERSION 3.25)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_cuphycp
   test_cuphycp.cpp
   test_utils.cpp
   ../testMAC/testMAC/launch_pattern.cpp
   ../testMAC/testMAC/test_mac_configs.cpp
   ../testMAC/testMAC/tests_launch_pattern.cpp
   ../scfl2adapter/lib/scf_5g_fapi/tests_scf_5g_slot_commands.cpp
)
target_compile_definitions(test_cuphycp PRIVATE CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM=3)
target_include_directories(test_cuphycp PRIVATE . ../testMAC/testMAC ../scfl2adapter/lib/scf_5g_fapi)
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
target_link_libraries(test_cuphycp
        # scf_5g_fapi and nvphy have circular static-archive refs; rescan resolves both.
        "$<LINK_GROUP:RESCAN,scf_5g_fapi,nvphy>"
        nvlog_static nvipc cuphy_cpp_header_only cuphy_hdf5
        yaml GTest::gmock GTest::gtest benchmark::benchmark aerial_sdk_version)

if (AERIAL_METRICS)
    target_link_libraries(test_cuphycp aerial_metrics)
endif()

message(STATUS "tests includes order kernel test bench")
add_subdirectory(order_kernel_test_bench)
message(STATUS "tests includes DL C test bench")
add_subdirectory(dlc_test_bench)
message(STATUS "tests includes C-Plane test bench")
add_subdirectory(cplane_test_bench)
message(STATUS "tests includes parser equivalence tests")
add_subdirectory(parser_equivalence)
