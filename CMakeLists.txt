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

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR [=[
No toolchain specified.
A toolchain needs to be specified
  Example:
  when building and running on the same system:
   -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native

  cross compilation is needed when building and running on different systems
  x86-64:
   -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/x86-64
  arm:
   -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/grace-cross


More information on toolchains can be found in the documentation
	]=]
    )
endif()
message(STATUS "Using toolchain ${CMAKE_TOOLCHAIN_FILE}")

include(cmake/cubb-utils.cmake)
read_aerial_sdk_version_file(${CMAKE_CURRENT_SOURCE_DIR})

# Make repo-local find modules (e.g. FindgRPC.cmake, used when gRPC comes from
# the apt packages that ship no gRPCConfig.cmake) visible to find_package().
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set default CMAKE_BUILD_TYPE=Release, can be overwritten by cmake .. -DCMAKE_BUILD_TYPE=Debug
if(NOT DEFINED CMAKE_BUILD_TYPE)
    message(WARNING "Setting CMAKE_BUILD_TYPE to 'Release' as none was specified.")
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug Release(Default) RelWithDebInfo MinSizeRel.")
endif()

# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
option(ENABLE_CCACHE "Enable compiler cache" ON)
if (ENABLE_CCACHE)
    find_program(CCACHE_FOUND "ccache")
    if (NOT CCACHE_FOUND)
        message(FATAL_ERROR "Failed to find program ccache.  You may need to install it.")
    endif()

    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
    message(STATUS "Enabled ccache")
endif()

# Get system ARCH
execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
string(STRIP ${ARCH} ARCH)

option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." FALSE)
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()


option(ENABLE_TESTS "Enable cuBB tests" ON)
if (ENABLE_TESTS)
    enable_testing()
endif()

if (NOT DEFINED ENV{cuBB_SDK})
    set(ENV{cuBB_SDK} ${CMAKE_CURRENT_SOURCE_DIR})
endif()
message(STATUS "Using ENV{cuBB_SDK} $ENV{cuBB_SDK}")


if (DEFINED CUPHY_GENCODE_ARCH_LIST)
    message(WARNING "CUPHY_GENCODE_ARCH_LIST is deprecated. Please use CMAKE_CUDA_ARCHITECTURES instead")
    string(REPLACE "," ";" CUPHY_GENCODE_ARCH_LIST_ ${CUPHY_GENCODE_ARCH_LIST})
    set(CMAKE_CUDA_ARCHITECTURES ${CUPHY_GENCODE_ARCH_LIST_})
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80-real 90-real)
endif()

#
#
project(cuphy_ctrl_plane LANGUAGES C CXX ASM CUDA)

include(cmake/CompileOptions.cmake)
include(cmake/Cuda.cmake)
include(cmake/ClangFormat.cmake)
include(cmake/CMakeFormat.cmake)

# clang-tidy is opt-in and target-scoped; ordinary builds do not find or run it.
option(ENABLE_CLANG_TIDY "Enable clang-tidy for explicitly opted-in C++ targets" OFF)
option(ENABLE_CLANG_TIDY_CACHE "Cache clang-tidy results" ON)
include(cmake/StaticAnalyzers.cmake)
if (ENABLE_TESTS)
    find_package(GTest     REQUIRED)
    find_package(benchmark REQUIRED)
endif()

message(STATUS "Using GPU architectures ${CMAKE_CUDA_ARCHITECTURES}")

# Now we have values in CMAKE_CUDA_ARCHITECTURES_NATIVE
# if we have "native" in CMAKE_CUDA_ARCHITECTURES, extract the native values.
if ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "native")
    string(FIND ${CMAKE_CUDA_ARCHITECTURES_NATIVE} "No CUDA devices found." RESULT_OUT)
    # If string exists, RESULT_OUT will have a valid non -1 index
    # and if it's value (non -1), then we need user to provide ARCH value
    # since we cannot extract it from CMAKE_CUDA_ARCHITECTURES_NATIVE
    if (NOT ${RESULT_OUT} EQUAL -1)
        message(FATAL_ERROR [=[
        "CMAKE_CUDA_ARCHITECTURES_NATIVE=${CMAKE_CUDA_ARCHITECTURES_NATIVE},
          You must provide -DCMAKE_CUDA_ARCHITECTURES=N, aborting generation"
        ]=]
        )
    endif ()
endif ()

if(ENABLE_IMPLICIT_CONVERSION_WARNINGS)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wconversion;-Wsign-conversion>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wconversion;-Wsign-conversion>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wconversion;-Wsign-conversion>")
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Werror>)
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-volatile>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-stringop-overread>)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wno-stringop-overread>)
    endif()
    
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-attributes>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wno-attributes>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
endif()

# ----------------------------------------------------------------------
# Project default options
OPTION(GTEST_POPULATE_CMAKE_TESTS "GoogleTest populates CTest" OFF)
OPTION(PHY_DRIVER_INTG "cuphydriver is integrated" ON)
OPTION(BUILD_DOCS "Build cuBB documentation" ON)
OPTION(BUILD_NVIPC_ONLY "Build libnvipc.so only" OFF)
OPTION(SCF_FAPI_10_04 "Set flag SCF_FAPI_10_04" OFF)
OPTION(ENABLE_L2_SLT_RSP "Enable L2 Slot Response" ON)
OPTION(ENABLE_FAPI_STORE_REPLAY "Enable FAPI store-replay. Requires ENABLE_L2_SLT_RSP." OFF)
OPTION(SCF_FAPI_10_04_SRS "Set flag SCF_FAPI_10_04_SRS for 10.04 SRS Only" OFF)
OPTION(CMAKE_CUDA_ARCHITECTURES "Set CUDA Architectures" OFF)
# Having BUILD_SHARED_LIBS set to OFF resulted in significant increase in cuPHY build directory in cuBB builds
OPTION(BUILD_SHARED_LIBS "Build Shared Libs" ON)
OPTION(AERIAL_METRICS "Enable Aerial Metrics" ON)
OPTION(ENABLE_PYAERIAL "Enable pyaerial Python bindings" ON)
OPTION(ENABLE_GPU3GPPCHAN_PYTHON "Enable gpu3gppchan Python bindings and wheel" ON)
OPTION(ENABLE_CELL_GROUP "Enable Cell Groups" ON)
OPTION(DOCA_ALLOW_EXPERIMENTAL_API "Enable DOCA Experimental API without compiler warnings" ON)
OPTION(NVIPC_FMTLOG_ENABLE "Enable nvIPC logging with fmtlog" ON)
OPTION(NVIPC_DOCA_ENABLE "Enable nvIPC on DOCA" OFF)
OPTION(ENABLE_BLOCKING_LOGGER "Logger blocks calling thread, instead of dropping message, if log queue is full" OFF)
OPTION(ENABLE_CUPHY_PTI_TRACING "Enable kernel tracing using cuPHY PTI" OFF)
OPTION(ENABLE_SCHED_FIFO_ALL_RT "Enable SCHED_FIFO on all high priority threads.  When OFF, SCHED_FIFO still used on some threads." ON)
OPTION(ENABLE_LINE_INFO "Add -lineinfo flag to compiler in release build" OFF)

OPTION(ENABLE_CONFORMANCE_TM_PDSCH_PDCCH "Enable Test Model 38.141-1 4.9.2.2" OFF)
OPTION(ENABLE_64C "Enable 64C" OFF)
OPTION(ENABLE_CUMAC "Enable cuMAC support in test benches and applications" ON)
OPTION(ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR "Enable PUSCH per-UE pre-EQ noise-interference power reporting" ON)

if ("${ARCH}" STREQUAL "aarch64")
    OPTION(FORCE_SLEEP_OF_L2A_MSG_THREAD "Enable WAR for system hang with SCHED_FIFO + 100% CPU polling" ON)
    OPTION(ENABLE_20C "Enable 20C" ON)
else ()
    OPTION(FORCE_SLEEP_OF_L2A_MSG_THREAD "Enable WAR for system hang with SCHED_FIFO + 100% CPU polling" OFF)
    OPTION(ENABLE_20C "Enable 20C" OFF)
endif()

OPTION(ENABLE_IMPLICIT_CONVERSION_WARNINGS "Enable '-Wconversion -Wsign-conversion' build flags" OFF)

if(ENABLE_20C)
    add_definitions(-DENABLE_20C)
endif()

if(ENABLE_64C)
    add_definitions(-DENABLE_64C)
endif()

if(SCF_FAPI_10_04)
    add_definitions(-DSCF_FAPI_10_04)
    add_definitions(-DSCF_FAPI_10_04_SRS)
endif()

if(SCF_FAPI_10_04_SRS)
    add_definitions(-DSCF_FAPI_10_04_SRS)
endif()

if(ENABLE_L2_SLT_RSP)
    add_definitions(-DENABLE_L2_SLT_RSP)
endif()

if(ENABLE_FAPI_STORE_REPLAY AND NOT ENABLE_L2_SLT_RSP)
    message(FATAL_ERROR
        "ENABLE_FAPI_STORE_REPLAY requires ENABLE_L2_SLT_RSP to be ON. "
        "Please enable ENABLE_L2_SLT_RSP or disable ENABLE_FAPI_STORE_REPLAY.")
endif()

# Store-replay supports both FAPI versions.

if(ENABLE_BLOCKING_LOGGER)
    add_definitions(-DFMTLOG_BLOCK)
endif()

if(NVIPC_FMTLOG_ENABLE)
    add_definitions(-DNVIPC_FMTLOG_ENABLE)
endif()

if(ENABLE_CUPHY_PTI_TRACING)
    add_definitions(-DCUPHY_PTI_ENABLE_TRACING)
endif()

if(ENABLE_SCHED_FIFO_ALL_RT)
    add_definitions(-DENABLE_SCHED_FIFO_ALL_RT)
endif()

if(EARLY_UCI_CUBB_CALLFLOW_TEST)
    add_definitions(-DEARLY_UCI_CUBB_CALLFLOW_TEST)
endif()

if(FORCE_SLEEP_OF_L2A_MSG_THREAD)
    add_definitions(-DFORCE_SLEEP_OF_L2A_MSG_THREAD)
endif()

if (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=1)
else (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=0)
endif (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_INSTALL_PREFIX ../install)

# ----------------------------------------------------------------------
if (AERIAL_METRICS)
    add_definitions(-DAERIAL_METRICS)
endif ()

find_package(CUDAToolkit)
if (NOT ${CUDAToolkit_FOUND})
    message("CUDA_TOOLKIT_ROOT_DIR NOT FOUND")
else ()
    # Legacy compatibility: CUDA_INCLUDE_DIRS used by cuphydriver, fh_generator, order_kernel_test_bench, etc.
    set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} CACHE INTERNAL "CUDA include dirs (from CUDAToolkit)")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    include_directories(
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}/cccl
    )

    # Pin CUDA::cuda_driver to the toolkit's own libcuda stub. FindCUDAToolkit's
    # find_library() will otherwise pick /usr/lib/x86_64-linux-gnu/libcuda.so when
    # a host NVIDIA driver is installed in the build container — and if that driver
    # predates the toolkit, it may be missing symbols and linking fails. The
    # toolkit stub exports every symbol this toolkit version uses; at runtime the
    # NVIDIA container runtime injects the real driver libcuda.
    if (TARGET CUDA::cuda_driver)
        find_library(_cubb_cuda_stub cuda
            HINTS "${CUDAToolkit_LIBRARY_DIR}/stubs"
            NO_DEFAULT_PATH)
        if (_cubb_cuda_stub)
            set_target_properties(CUDA::cuda_driver PROPERTIES
                IMPORTED_LOCATION "${_cubb_cuda_stub}")
            message(STATUS "CUDA::cuda_driver pinned to toolkit stub: ${_cubb_cuda_stub}")
        else()
            message(WARNING "Toolkit libcuda stub not found under ${CUDAToolkit_LIBRARY_DIR}/stubs; "
                "CUDA::cuda_driver may resolve to a system libcuda that lacks newer driver APIs.")
        endif()
        unset(_cubb_cuda_stub CACHE)
    endif()
endif ()

# ----------------------------------------------------------------------
# nvidia gputelecom repos
# ----------------------------------------------------------------------
add_subdirectory(aerial_common)
add_subdirectory(cuPHY)
if (ENABLE_PYAERIAL)
    # pyAerial consumes the gpu3gppchan wheel, so it always enables its producer.
    set(ENABLE_GPU3GPPCHAN_PYTHON ON CACHE BOOL
        "Enable gpu3gppchan Python bindings and wheel" FORCE)
endif()
if (ENABLE_GPU3GPPCHAN_PYTHON)
    # Build gpu3gppchan's Python wheel from the native target created by cuPHY.
    add_subdirectory(testBenches/gpu3GPPChan/python)
endif()
if (ENABLE_PYAERIAL)
    add_subdirectory(pyaerial)
endif()
if (ENABLE_TESTS)
    add_subdirectory(testBenches)
endif()

# Include cuMAC and cuMAC-CP if exist
if (ENABLE_CUMAC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuMAC)
    add_subdirectory(cuMAC)
    if (ENABLE_CUMAC AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuMAC-CP)
        add_subdirectory(cuMAC-CP)
    endif()
endif()

find_package(PkgConfig REQUIRED)
find_library(MLX5_LIB mlx5)
find_library(IBVERBS_LIB ibverbs)
SET(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/opt/mellanox/dpdk/lib/${ARCH}-linux-gnu/pkgconfig:/opt/mellanox/doca/lib/${ARCH}-linux-gnu/pkgconfig:/opt/mellanox/collectx/lib/${ARCH}-linux-gnu/pkgconfig")
SET(DPDK_PATH /opt/mellanox/dpdk)
pkg_search_module(DPDK REQUIRED libdpdk)
SET(DOCA_PATH /opt/mellanox/doca)
pkg_search_module(DOCA REQUIRED doca-gpunetio)


add_subdirectory(cuPHY-CP/app_config)
add_subdirectory(cuPHY-CP/app_utils)
add_subdirectory(cuPHY-CP/gt_common_libs)
add_subdirectory(cuPHY-CP/cuphyoam)
add_subdirectory(cuPHY-CP/ru-emulator)
add_subdirectory(cuPHY-CP/aerial-fh-driver)

# Set BEFORE cuphydriver is added (cuphydriver gates sources on this).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuPHY-CP/external/aerial-framework/CMakeLists.txt)
    set(AERIAL_FRAMEWORK_AVAILABLE ON CACHE INTERNAL "aerial-framework submodule present")
else()
    set(AERIAL_FRAMEWORK_AVAILABLE OFF CACHE INTERNAL "aerial-framework submodule present")
endif()

add_subdirectory(cuPHY-CP/cuphydriver)
add_subdirectory(cuPHY-CP/data_lake)
add_subdirectory(cuPHY-CP/cuphyl2adapter)
add_subdirectory(cuPHY-CP/scfl2adapter)
add_subdirectory(cuPHY-CP/cuphycontroller)
add_subdirectory(cuPHY-CP/compression_decompression)

# Add testMAC after l2adapter modules
set(SCF_FAPI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi)
set(CUPHY_COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/cuPHY/examples/common)
add_subdirectory(cuPHY-CP/testMAC)

# Optional aerial-framework submodule (opt in via: git submodule update --init --checkout -- cuPHY-CP/external/aerial-framework)
if(AERIAL_FRAMEWORK_AVAILABLE)
    set(FRAMEWORK_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(FRAMEWORK_MAINTAINER_MODE OFF CACHE BOOL "" FORCE)
    set(FRAMEWORK_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(FRAMEWORK_SKIP_ACAR_DOWNLOAD ON CACHE BOOL "" FORCE)
    set(FRAMEWORK_ENABLE_RAN_PY OFF CACHE BOOL "" FORCE)
    set(ACAR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "" FORCE)
    set(ACAR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "" FORCE)
    set(FRAMEWORK_USE_FMTLOG_BACKEND ON CACHE BOOL "" FORCE)
    add_subdirectory(cuPHY-CP/external/aerial-framework EXCLUDE_FROM_ALL)
else()
    message(STATUS "aerial-framework submodule not present; providing empty ran::fapi / ran::oran stubs")
    add_library(ran_fapi_stub INTERFACE)
    add_library(ran::fapi ALIAS ran_fapi_stub)
    add_library(ran_oran_stub INTERFACE)
    add_library(ran::oran ALIAS ran_oran_stub)
endif()

# cuPHY-CP unit tests
if (ENABLE_TESTS AND AERIAL_METRICS)
    add_subdirectory(cuPHY-CP/tests)
endif()
