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

#################################################################################################
# Notes:
# 1. This file is used both in Aerial SDK build and NVIPC standalone build (for L2 integration).
# 2. For NVIPC standalone build, this is the top CMakeLists.txt and GCC version may be different.
#################################################################################################

# Minimum CMake version required.
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(gt_common_libs LANGUAGES C CXX)

# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------
# Below options doesn't take effect if had been defined in cuBB_SDK top CMakeLists.txt
OPTION(BUILD_NVIPC_ONLY "Build libnvipc.so only" ON)
OPTION(NVIPC_CUDA_ENABLE "Build libnvipc.so with CUDA" ON)
OPTION(NVIPC_GDRCPY_ENABLE "Build libnvipc.so with GDRCOPY" OFF)
OPTION(NVIPC_DPDK_ENABLE "Build libnvipc.so with DPDK" OFF)
OPTION(NVIPC_DOCA_ENABLE "Build libnvipc.so with DOCA" OFF)
OPTION(NVIPC_DOCA_GPUNETIO "Use DOCA gpunetio in DOCA GPU DMA" OFF)
OPTION(NVIPC_FMTLOG_ENABLE "Enable nvIPC logging with fmtlog" ON)
OPTION(ENABLE_SCTP_CHECK "Check for SCTP library availability" ON)

# FMTLOG requires C++17. When L2 partner uses GCC version < 7.0 which doesn't support C++17, disable FMTLOG
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
   set(NVIPC_FMTLOG_ENABLE OFF)
endif()

# For NVIPC standalone build. If CMAKE_CXX_STANDARD was not defined in cmake options, set a default version
if (NOT CMAKE_CXX_STANDARD)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    set(CMAKE_CXX_STANDARD 11)
  else()
    set(CMAKE_CXX_STANDARD 17)
  endif()
endif()

# Set default CMAKE_BUILD_TYPE=Release, can be overwritten by cmake .. -DCMAKE_BUILD_TYPE=Debug
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    SET(CMAKE_BUILD_TYPE Release)
endif()

find_package(PkgConfig REQUIRED)

# Disable NVIPC_DOCA_ENABLE if DOCA version < 2.5
pkg_search_module(DOCA QUIET doca)
if("${DOCA_VERSION}" STREQUAL "" OR DOCA_VERSION LESS 2.5)
    message("DOCA_VERSION=${DOCA_VERSION}, set NVIPC_DOCA_ENABLE=OFF")
    set(NVIPC_DOCA_ENABLE OFF)
endif()

# Enable NVIPC_DPDK_ENABLE if DPDK exist
pkg_search_module(DPDK QUIET libdpdk)
if("${DPDK_VERSION}" STREQUAL "")
    message("DPDK_VERSION not defined, set NVIPC_DPDK_ENABLE=OFF")
    set(NVIPC_DPDK_ENABLE OFF)
endif()

if(NVIPC_DPDK_ENABLE)
    add_definitions(-DNVIPC_DPDK_ENABLE)
    if(DPDK_VERSION LESS 22.11)
        message("DPDK_VERSION=${DPDK_VERSION}, set DOCA_GPU_DPDK_OLD=1")
        add_definitions(-DDOCA_GPU_DPDK_OLD=1)
    else()
        message("DPDK_VERSION=${DPDK_VERSION}, set DOCA_GPU_DPDK_OLD=0")
        add_definitions(-DDOCA_GPU_DPDK_OLD=0)
    endif()
endif()

find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND OR CUDAToolkit_VERSION LESS 11.3)
    message("CUDAToolkit_FOUND=${CUDAToolkit_FOUND} CUDAToolkit_VERSION=${CUDAToolkit_VERSION}: Disable CUDA")
    set(NVIPC_CUDA_ENABLE OFF)
endif()

if(NVIPC_CUDA_ENABLE)
    enable_language(CUDA)
    add_definitions(-DNVIPC_CUDA_ENABLE)
endif()

# Get system ARCH
execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
string(STRIP ${ARCH} ARCH)

message("=========================================================")
message("Build nvipc for [${ARCH}] CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message("NVIPC_CUDA_ENABLE=${NVIPC_CUDA_ENABLE} CUDAToolkit_VERSION=${CUDAToolkit_VERSION}")
message("NVIPC_DPDK_ENABLE=${NVIPC_DPDK_ENABLE} DPDK_VERSION=${DPDK_VERSION}")
message("NVIPC_DOCA_ENABLE=${NVIPC_DOCA_ENABLE} DOCA_VERSION=${DOCA_VERSION}")
message("NVIPC_DOCA_GPUNETIO=${NVIPC_DOCA_GPUNETIO}")
message("NVIPC_GDRCPY_ENABLE=${NVIPC_GDRCPY_ENABLE}")
message("NVIPC_FMTLOG_ENABLE=${NVIPC_FMTLOG_ENABLE}")
message("=========================================================")

if(NVIPC_GDRCPY_ENABLE)
    add_definitions(-DNVIPC_GDRCPY_ENABLE)
endif()

# ====== Build libnvipc.so only for partner ============================
if(BUILD_NVIPC_ONLY)
    message("Build nvipc only for [${ARCH}]")

    # Please export PKG_CONFIG_PATH to the DPDK pkgconfig path
    message(STATUS "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}")
    add_definitions(-DBUILD_NVIPC_ONLY)

    # In-house yaml library
    if (NOT (TARGET yaml))
        message(STATUS "including libyaml")
        add_subdirectory(external/libyaml)
    endif()
    set_property(TARGET yaml PROPERTY POSITION_INDEPENDENT_CODE ON)

    # OSS yaml-cpp
    if (NOT (TARGET yaml-cpp))
        message(STATUS "including yaml-cpp")
        add_subdirectory(external/yaml-cpp)
    endif()
    set_property(TARGET yaml-cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

    if(NVIPC_FMTLOG_ENABLE)
        if (NOT (TARGET fmt))
            message(STATUS "including fmtlog")
            add_subdirectory(external/fmtlog)
            set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
            set_property(TARGET fmtlog-static PROPERTY POSITION_INDEPENDENT_CODE ON)
        endif()
        add_definitions(-DNVIPC_FMTLOG_ENABLE)
    endif()
    if (NOT TARGET nvlog)
        add_subdirectory(nvlog)
    endif()
    if (NVIPC_CUDA_ENABLE AND NOT TARGET cuda_driver_utils)
        add_subdirectory(cuda_driver_utils)
    endif()
    if (NOT TARGET nvipc)
        message(STATUS "including nvipc")
        add_subdirectory(nvIPC)
    endif()

    RETURN()
endif()

# ====== Build libnvipc.so inside cuBB_SDK =============================

# ----------------------------------------------------------------------
# Additional packages
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(HDF5_USE_STATIC_LIBRARIES ON)
if (CUPHY_HDF5_VERSION_CHECK)
    find_package(HDF5 1.10 REQUIRED COMPONENTS C)
else (CUPHY_HDF5_VERSION_CHECK)
    find_package(HDF5 REQUIRED COMPONENTS C)
endif (CUPHY_HDF5_VERSION_CHECK)

# check for CURL
find_package(CURL REQUIRED)

# check for SCTP
if(ENABLE_SCTP_CHECK)
    find_library(Sctp NAMES sctp)
    if(Sctp MATCHES Sctp-NOTFOUND)
        message(FATAL_ERROR "libsctp not found. RPC over SCTP support will not be compiled. Set -DENABLE_SCTP_CHECK=OFF to skip this check.")
    else()
        message(STATUS "Found libsctp: ${Sctp}")
    endif()
else()
    message(STATUS "SCTP check disabled (ENABLE_SCTP_CHECK=OFF)")
endif()


# ----------------------------------------------------------------------
# Subdirectories

if(AERIAL_METRICS)
    if(NOT TARGET aerial_metrics)
        add_subdirectory(aerial_metrics)
    endif()
else()
    message(STATUS "Aerial metrics disabled (AERIAL_METRICS=OFF)")
endif()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuPHY/CMakeLists.txt")
  add_subdirectory(cuPHY)
endif()

if (NOT TARGET nvipc)
    message(STATUS "including nvipc")
    add_subdirectory(nvIPC)
endif()

if (NOT TARGET oran_utils)
  message(STATUS "including oran_utils")
  add_subdirectory(oran_utils)
endif()

if (NOT TARGET slot_command)
    message(STATUS "including slot_comamnd")
    add_subdirectory(slot_command)
endif()

if (NOT TARGET cuda_driver_utils)
    message(STATUS "including cuda_driver_utils")
    add_subdirectory(cuda_driver_utils)
endif()

if (NOT TARGET perf_metrics)
    message(STATUS "including perf_metrics")
    add_subdirectory(perf_metrics)
endif()

if (NOT TARGET pmu_reader)
    message(STATUS "including pmu_reader")
    add_subdirectory(pmu_reader)
endif()

if (NOT TARGET task_instrumentation)
    message(STATUS "including task_instrumentation")
    add_subdirectory(task_instrumentation)
endif()

# Optional perf benchmark (separate executable; does not affect task_instrumentation link interface).
if(ENABLE_TESTS)
    add_subdirectory(task_instrumentation/benchmark)
endif()
