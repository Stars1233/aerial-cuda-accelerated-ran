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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Check for cuBB_SDK environment variable
if (DEFINED ENV{cuBB_SDK})
    set(CUBB_HOME $ENV{cuBB_SDK})
    message(STATUS "Setting CUBB_HOME to ${CUBB_HOME}")
else()
    message(FATAL_ERROR "You must define cuBB_SDK environment variable")
endif()

# Find HDF5 package
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)

# Add test executable
add_executable(dlc_test_bench
    dlc_test_bench.cpp
    dlc_test_bench.hpp
    sendCplaneUnitTest.cpp
    sendCplaneUnitTest.hpp
    testmac_setup_service.cpp
    testmac_setup_service.hpp
    ru_emulator_wrapper.cpp
    ru_emulator_wrapper.hpp
    pcap_writer.cpp
    pcap_writer.h
    ${CUBB_HOME}/cuPHY-CP/cuphycontroller/src/yamlparser.cpp
    ${CUBB_HOME}/cuPHY-CP/cuphycontroller/src/cuphydriver.cpp
)

# Add dependencies to ensure libraries are built before the test
add_dependencies(dlc_test_bench cuphydriver scf_5g_fapi nvphy cuphyoamlib aerial_metrics app_config slot_command nvlog aerial-fh perf_metrics)

find_package(CLI11 REQUIRED)
# Add RU emulator specific libraries (from helper.cmake link_target function)
target_link_libraries(dlc_test_bench PRIVATE slot_command nvlog yaml CUDA::cuda_driver aerial-fh cuphyoamlib app_config perf_metrics CLI11::CLI11)

# DPDK configuration (following project pattern)
target_link_directories(dlc_test_bench PRIVATE ${DPDK_LIBRARY_DIRS})
target_compile_options(dlc_test_bench PRIVATE ${DPDK_CFLAGS})

# Link dependencies
target_link_libraries(dlc_test_bench
    PRIVATE
        gtest          # Only gtest library (NO gtest_main - we provide our own main())
        # gtest_main   # REMOVED: We have custom main() in dlc_test_bench.cpp
        benchmark
        cuphydriver
        cuda_driver_utils
        ${HDF5_C_LIBRARIES}
        cuphy_hdf5
        cuphy_cpp_header_only
        nvphy
        aerial_sdk_version
        gdrapi
        ${DPDK_LIBRARIES}  # DPDK libraries including mbuf for rte_pktmbuf_clone
        # Add testMAC dependencies
        scf_5g_fapi  # SCF FAPI library for scf_fapi_handler
        nvipc        # NVIPC library for transport
        testmac_core
        ru_emulator_common
        sdk::options
)

# Include directories
target_include_directories(dlc_test_bench
    PRIVATE
        ${CUBB_HOME}/cuPHY-CP/tests/
        ${CUBB_HOME}/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi
        ${CUBB_HOME}/cuPHY-CP/cuphydriver/include
        ${CUBB_HOME}/cuPHY-CP/cuphycontroller/include
        ${CUBB_HOME}/cuPHY-CP/compression_decompression/comp_decomp_lib/include
        ${CUBB_HOME}/cuPHY-CP/aerial-fh-driver/lib
        ${CUBB_HOME}/cuPHY-CP/aerial-fh-driver/include
        ${CUBB_HOME}/cuPHY-CP/aerial-fh-driver/include/aerial-fh-driver
        ${CUBB_HOME}/cuPHY-CP/testMAC/testMAC
        ${CUBB_HOME}/cuPHY-CP/ru-emulator/ru_emulator
        ${CUBB_HOME}/cuPHY/src
        ${CUBB_HOME}/cuPHY/src/cuphy_channels
        ${CUBB_HOME}/cuPHY-CP/data_lake
        ${CUBB_HOME}/cuPHY-CP/external/clickhouse-cpp
        ${HDF5_INCLUDE_DIRS}
        /usr/local/include
        /usr/local/gdrcopy/include
)

# Additional include directories from cuphycontroller example
target_include_directories(dlc_test_bench PUBLIC ${CUBB_HOME}/cuPHY-CP/cuphycontroller/include)
target_include_directories(dlc_test_bench PUBLIC ${CUBB_HOME}/cuPHY-CP/cuphycontroller/lib/cuphydriver/include)


# Add compile definitions if needed
target_compile_definitions(dlc_test_bench
    PRIVATE
        ${HDF5_DEFINITIONS}
)

# Add compile options
if (ENABLE_LINE_INFO)
    target_compile_options(dlc_test_bench PRIVATE -lineinfo)
endif()


# Install test
install(TARGETS dlc_test_bench
    RUNTIME DESTINATION bin/tests
)

# Copy test data files
install(FILES 
    ${CMAKE_SOURCE_DIR}/testVectors/multi-cell/launch_pattern_F08_20C_59c.yaml
    DESTINATION share/testVectors/multi-cell
)

