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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# cplane_test_bench requires the aerial-framework CPlaneGenerator.
if(NOT AERIAL_FRAMEWORK_AVAILABLE)
    message(STATUS "cplane_test_bench: skipped (aerial-framework not available)")
    return()
endif()

# Check for cuBB_SDK environment variable
if (DEFINED ENV{cuBB_SDK})
    set(CUBB_HOME $ENV{cuBB_SDK})
    message(STATUS "cplane_test_bench: CUBB_HOME = ${CUBB_HOME}")
else()
    message(FATAL_ERROR "You must define cuBB_SDK environment variable")
endif()

# Find HDF5 (needed by RU emulator)
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)

# Add test executable
add_executable(cplane_test_bench
    # C-Plane test bench sources
    cplane_test_bench.cpp
    cplane_test_fixture.cpp
    fapi_source_testmac.cpp
    ru_emulator_wrapper.cpp
    pcap_writer.cpp
)

target_compile_definitions(cplane_test_bench
    PRIVATE
        ${HDF5_DEFINITIONS}
)

# Dependencies — ensure libraries are built first.
add_dependencies(cplane_test_bench oran app_config nvlog aerial-fh perf_metrics)

find_package(CLI11 REQUIRED)

# Link libraries
target_link_libraries(cplane_test_bench
    PRIVATE
        gtest             # Google Test (custom main, no gtest_main)
        oran              # Framework CPlaneGenerator, VecBuf
        scf::scf_5g_fapi_h # FAPI struct definitions only
        nvlog             # Logging
        yaml              # YAML config parsing
        app_config        # Config utilities
        aerial_sdk_version
        aerial-fh         # Fronthaul driver (MsgReceiveInfo struct for RU emulator wrapper)
        perf_metrics      # Performance metrics (RU emulator dependency)
        slot_command       # Slot command types
        ${HDF5_C_LIBRARIES}  # HDF5 (RU emulator test vector loading)
        cuphy_hdf5           # cuPHY HDF5 utilities
        cuphy_cpp_header_only
        ${DPDK_LIBRARIES}    # DPDK (RU emulator internals)
        sdk::options
        sdk::warnings
)

# Link third-party libraries as SYSTEM so their headers don't trip the strict
# warning set from sdk::warnings. CLI11 in particular triggers a GCC 12
# -Werror=null-dereference false positive inside CLI::detail::split when
# vector<std::string>::emplace_back expands the basic_string move constructor
# (surfaces at -O2, i.e. RelWithDebInfo; -O3 folds past it).
target_link_system_libraries(cplane_test_bench PRIVATE CLI11::CLI11 testmac_core ru_emulator_common)

# DPDK: SYSTEM includes (suppress warnings) + required compile flags/link directories.
target_compile_options(cplane_test_bench PRIVATE ${DPDK_CFLAGS_OTHER})
target_include_directories(cplane_test_bench SYSTEM PRIVATE ${DPDK_INCLUDE_DIRS})
target_link_directories(cplane_test_bench PRIVATE ${DPDK_LIBRARY_DIRS})

# VectorCAST instrumentation pragmas leak through NVLOG macro expansion;
# harmless in test bench code.
target_compile_options(cplane_test_bench PRIVATE -Wno-unknown-pragmas)

# Include directories
# Non-owned headers — SYSTEM to suppress warnings
target_include_directories(cplane_test_bench SYSTEM PRIVATE
    ${CUBB_HOME}/cuPHY-CP/tests/
    ${CUBB_HOME}/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi
    ${CUBB_HOME}/cuPHY-CP/testMAC/testMAC
    ${CUBB_HOME}/cuPHY-CP/ru-emulator/ru_emulator
    ${CUBB_HOME}/cuPHY-CP/cuphydriver/include
    ${CUBB_HOME}/cuPHY-CP/cuphycontroller/include
    ${CUBB_HOME}/cuPHY-CP/compression_decompression/comp_decomp_lib/include
    ${CUBB_HOME}/cuPHY/src
    ${CUBB_HOME}/cuPHY/src/cuphy_channels
    ${HDF5_INCLUDE_DIRS}
)

# Own headers — NOT SYSTEM
target_include_directories(cplane_test_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


# ctest registration (per-pattern tests, phase fixtures, CSV auto-discovery)
# lives in a sibling file to keep this CMakeLists.txt focused on the build.
include("${CMAKE_CURRENT_SOURCE_DIR}/register_tests.cmake")

# Install
install(TARGETS cplane_test_bench RUNTIME DESTINATION bin/tests)
