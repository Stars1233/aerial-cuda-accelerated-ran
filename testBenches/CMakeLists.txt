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

# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)

if (DEFINED CUMAC_GENCODE_ARCH_LIST)
    message(WARNING "CUMAC_GENCODE_ARCH_LIST is deprecated. Please use CMAKE_CUDA_ARCHITECTURES instead")
    string(REPLACE "," ";" CUMAC_GENCODE_ARCH_LIST_ ${CUMAC_GENCODE_ARCH_LIST})
    set(CMAKE_CUDA_ARCHITECTURES ${CUMAC_GENCODE_ARCH_LIST_})
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80-real 90-real)
endif()

# Get system ARCH
execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
string(STRIP ${ARCH} ARCH)

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if ("${ARCH}" STREQUAL "aarch64")
        set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../cuPHY/cmake/toolchains/grace-cross)
    else ()
        set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../cuPHY/cmake/toolchains/devkit)
    endif()
endif()
message(STATUS "testBenches is using toolchain ${CMAKE_TOOLCHAIN_FILE}")

message(STATUS "testBenches is using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

# ----------------------------------------------------------------------
# Project options
project(testBenches LANGUAGES C CXX CUDA ASM) 

include(../cmake/cubb-utils.cmake)
read_aerial_sdk_version_file("${CMAKE_CURRENT_SOURCE_DIR}/..") # the location of aerial-sdk-version file is in ".."
if (NOT AERIAL_SDK_VERSION)
    message(FATAL_ERROR "AERIAL_SDK_VERSION is NULL or Undefined, halting")
endif ()

OPTION(ENABLE_64C "Enable 64C" OFF)
if ("${ARCH}" STREQUAL "aarch64")
    OPTION(ENABLE_20C "Enable 20C" ON)
else ()
    OPTION(ENABLE_20C "Enable 20C" OFF)
endif()
option(ENABLE_NVTX  "Enable use of NVTX" OFF)
option(BUILD_SHARED_LIBS "Build libraries as shared" ON)

if(ENABLE_20C)
    add_definitions(-DENABLE_20C)
endif()

if(ENABLE_64C)
    add_definitions(-DENABLE_64C)
endif()
# ----------------------------------------------------------------------
# Additional packages
find_package(Threads REQUIRED)
# https://cmake.org/cmake/help/latest/module/FindHDF5.html
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(CUDAToolkit REQUIRED)

if (ENABLE_NVTX)
    add_definitions(-DUSE_NVTX=1)
endif (ENABLE_NVTX)

# ----------------------------------------------------------------------
# choose cmake toolchain
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # Get system ARCH
    execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
    string(STRIP ${ARCH} ARCH)

    # Run the command to get correct arch
    execute_process(
        COMMAND bash -c "gcc -march=native -Q --help=target | grep march"
        OUTPUT_VARIABLE GCC_MARCH_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    string(FIND "${GCC_MARCH_OUTPUT}" "znver" FIND_ZNVER_ARCH)

    if ("${ARCH}" STREQUAL "aarch64")
        set(CMAKE_TOOLCHAIN_FILE cmake/toolchains/grace-cross)
    elseif (${FIND_ZNVER_ARCH})
        set(CMAKE_TOOLCHAIN_FILE cmake/toolchains/x86-64) # use -march=x86-64 
    else()
        set(CMAKE_TOOLCHAIN_FILE cmake/toolchains/devkit) # use -march=cascadelake
    endif()
endif()

message(STATUS "Using toolchain ${CMAKE_TOOLCHAIN_FILE}")

#----------------------------------------------------------------------
# Subdirectories
if (NOT TARGET nvlog)
    message(STATUS "including nvlog")
    if(NOT NVIPC_FMTLOG_ENABLE)
        message(STATUS "Setting NVIPC_FMTLOG_ENABLE ON")
        set(NVIPC_FMTLOG_ENABLE ON)
        add_definitions(-DNVIPC_FMTLOG_ENABLE)
    endif()
    add_subdirectory(../cuPHY/nvlog cuPHY/nvlog)
endif()

# ----------------------------------------------------------------------
# cuMAC compilation option (if not already defined at top level)
if(NOT DEFINED ENABLE_CUMAC)
    option(ENABLE_CUMAC "Enable cuMAC support in test benches" ON)
endif()

# if this is top level CMakeLists.txt, add cuPHY and cuMAC dependencies
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    add_subdirectory(../cuPHY/src cuPHY/src)

    if(ENABLE_CUMAC)
        add_subdirectory(../cuMAC/src cuMAC/src cuMAC/lib/cumac_msg)
    endif()
endif()

# Add a target to build all testbench examples with a single command.
add_custom_target(testbenches_examples
    DEPENDS
        cubb_gpu_test_bench
        cdl_chan_ex
        fading_chan_ex
        gau_noise_adder_ex
        ofdm_mod_demod
        sls_chan_ex
        tdl_chan_ex
    COMMENT "Builds all specified testbench examples"
)

# test benches 
add_subdirectory(cubb_gpu_test_bench)  # phase-3 test bench (supports conditional cuMAC)
add_subdirectory(chanModels)
