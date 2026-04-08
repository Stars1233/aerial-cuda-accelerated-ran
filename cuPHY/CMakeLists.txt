# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Minimum CMake version required.
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
message(STATUS "cuPHY is using toolchain ${CMAKE_TOOLCHAIN_FILE}")

# Set default CMAKE_BUILD_TYPE=Release, can be overwritten by cmake .. -DCMAKE_BUILD_TYPE=Debug
if(NOT DEFINED CMAKE_BUILD_TYPE)
    message(WARNING "Setting CMAKE_BUILD_TYPE to 'Release' as none was specified.")
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug Release(Default) RelWithDebInfo MinSizeRel.")
endif()

# Get system ARCH
execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
string(STRIP ${ARCH} ARCH)

if (NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()


# ----------------------------------------------------------------------
# Build options
option(USE_MATLAB "Build MEX libraries to allow use from MATLAB" OFF)
option(BUILD_DOCS "Generate Doxygen documentation" ON)
option(CUPHY_BUILD_PTXAS_VERBOSE "Enable verbose mode for the PTX assembler to show local memory usage" OFF)
option(CUPHY_HDF5_VERSION_CHECK "Force HDF5 library version checking, necessary for interoperability with binary cuPHY packages" ON)
option(ENABLE_NVTX  "Enable use of NVTX" OFF)
option(BUILD_SHARED_LIBS "Build cuPHY as a shared library" ON)
option(ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR  "Enable PUSCH per-UE pre-EQ noise-interference power reporting" ON)
OPTION(ENABLE_CUPHY_PTI_TRACING "Enable kernel tracing using cuPHY PTI" OFF)
option(ASIM_CUPHY_FP32 "Enable ASIM cuPHY PUSCH FP32 input" OFF)
option(ENABLE_PUSCH_LDPC_OUTPUT_H5_DUMP "Enable PUSCH LDPC OUTPUT H5 DUMP" OFF)
option(ASIM_CUPHY_SRS_OUTPUT_FP32 "Enable ASIM cuPHY SRS FP32 output" OFF)
option(ENABLE_LINE_INFO "Add -lineinfo flag to CUPHY_NVCC_FLAGS" OFF)
OPTION(ENABLE_32DL "Enable 32-layer Downlink" OFF)
OPTION(ENABLE_64C "Enable 64C" OFF)

if(ENABLE_CUPHY_PTI_TRACING)
    add_definitions(-DCUPHY_PTI_ENABLE_TRACING)
endif()
if(ASIM_CUPHY_FP32)
    add_definitions(-DASIM_CUPHY_FP32)
endif()
if(ENABLE_PUSCH_LDPC_OUTPUT_H5_DUMP)
    add_definitions(-DENABLE_PUSCH_LDPC_OUTPUT_H5_DUMP)
endif()
if(ASIM_CUPHY_SRS_OUTPUT_FP32)
    add_definitions(-DASIM_CUPHY_SRS_OUTPUT_FP32)
endif()
if(BFW_BOTH_COMP_FLOAT)
    add_definitions(-DBFW_BOTH_COMP_FLOAT)
endif()
if(ENABLE_32DL)
    add_definitions(-DENABLE_32DL)
endif()
if(ENABLE_64C)
    add_definitions(-DENABLE_64C)
endif()
# ----------------------------------------------------------------------
# Optionally include internal cuPHY CMake commands
set(CUPHY_INTERNAL_CMAKE_FILE OFF)
include(../../internal/cuPHY/cuphy_internal.cmake OPTIONAL)

# ----------------------------------------------------------------------
# Set the version number for this project
set(cuPHY_VERSION_MAJOR 0)
set(cuPHY_VERSION_MINOR 8)

# ----------------------------------------------------------------------
# Enable testing support
if (ENABLE_TESTS)
    enable_testing()
endif()

# ----------------------------------------------------------------------
# Global C++ options
# Though this is mentioned in the root CMakeLists.txt, in some projects, this directory
# is being added with add_subdirectory() hence the root CMakeLists.txt is not used and
# these variables are not set.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if (DEFINED CUPHY_GENCODE_ARCH_LIST)
    message(WARNING "CUPHY_GENCODE_ARCH_LIST is deprecated. Please use CMAKE_CUDA_ARCHITECTURES instead")
    string(REPLACE "," ";" CUPHY_GENCODE_ARCH_LIST_ ${CUPHY_GENCODE_ARCH_LIST})
    set(CMAKE_CUDA_ARCHITECTURES ${CUPHY_GENCODE_ARCH_LIST_})
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80-real 90-real)
endif()

project(cuPHY LANGUAGES C CXX ASM CUDA)

include(../cmake/cubb-utils.cmake)
read_aerial_sdk_version_file("${CMAKE_CURRENT_SOURCE_DIR}/..") # the location of aerial-sdk-version file is in ".."
if (NOT AERIAL_SDK_VERSION)
    message(FATAL_ERROR "AERIAL_SDK_VERSION is NULL or Undefined, halting")
endif ()

find_package(CLI11 REQUIRED)

# ----------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND CUPHY_NVCC_FLAGS -Wreorder -lineinfo)
elseif(ENABLE_LINE_INFO)
    list(APPEND CUPHY_NVCC_FLAGS -lineinfo)
endif()


if(CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND CUPHY_NVCC_FLAGS -Wreorder -Werror all-warnings)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror=reorder>)
#add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)


message(STATUS "cuPHY using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

if(CUPHY_BUILD_PTXAS_VERBOSE)
    #set(CUPHY_PTXAS_OPTIONS "--ptxas-options=-v")
    set(CUPHY_PTXAS_OPTIONS ${CUPHY_PTXAS_OPTIONS} "--resource-usage")
endif(CUPHY_BUILD_PTXAS_VERBOSE)

# ----------------------------------------------------------------------
# Additional packages
set(THREADS_PREFER_PTHREAD_FLAG OFF)
find_package(Threads REQUIRED)

# https://cmake.org/cmake/help/latest/module/FindHDF5.html
# set(HDF5_USE_STATIC_LIBRARIES ON)
if (NOT DEFINED HDF5_C_LIBRARIES) # allow for external hdf5 project
    if (CUPHY_HDF5_VERSION_CHECK)
        find_package(HDF5 1.10 REQUIRED COMPONENTS C)
    else()
        find_package(HDF5 REQUIRED COMPONENTS C)
    endif()
endif()

if (USE_MATLAB)
    find_package(Matlab REQUIRED)
endif (USE_MATLAB)

if (ENABLE_NVTX)
    add_definitions(-DUSE_NVTX=1)
endif (ENABLE_NVTX)

if (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=1)
else (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=0)
endif (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)

# ----------------------------------------------------------------------
# Generation of documentation
if (BUILD_DOCS)
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        set(DOXYGEN_IN  ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        message(STATUS "Configuring ${DOXYGEN_OUT}")
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        add_custom_target(docs_doxygen ALL
          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          COMMENT "Generating API documentation (doxygen)"
          VERBATIM)
    else (DOXYGEN_FOUND)
        message(FATAL_ERROR "Documentation generation requested, but Doxygen package not found")
    endif (DOXYGEN_FOUND)
endif (BUILD_DOCS)

project(cuPHY)
# ----------------------------------------------------------------------
# Subdirectories
if (NOT TARGET nvlog)
    message(STATUS "including nvlog")
    if(NOT NVIPC_FMTLOG_ENABLE)
        message(STATUS "Setting NVIPC_FMTLOG_ENABLE ON")
        set(NVIPC_FMTLOG_ENABLE ON)
        add_definitions(-DNVIPC_FMTLOG_ENABLE)
    endif()
    add_subdirectory(nvlog)
endif()
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(util/aerial_utils)
if (ENABLE_TESTS)
    add_subdirectory(test)
endif()

# ----------------------------------------------------------------------
# build chanModels library from testBenches/chanModels/src if it does not exist
if (NOT TARGET chanModels)
    add_subdirectory(../testBenches/chanModels/src examples/chanModels/src)
endif()
set(CHAN_MODELS_SRC_INCLUDE_DIR ../testBenches/chanModels/src)

if (USE_MATLAB)
    add_subdirectory(util/matlab/mex)
endif (USE_MATLAB)

# Add a target to build all cuPHY examples with a single command.
# This assumes that targets corresponding to the example directories
# (e.g., 'bfc', 'ch_est', etc.) are defined within examples/CMakeLists.txt or similar.
add_custom_target(cuphy_examples ALL
    DEPENDS
        cuphy_ex_bfc
        cuphy_ex_ch_est
        cuphy_ex_channel_eq
        cuphy_ex_gen_prach_perf_curve_from_file
        cuphy_ex_ldpc
        cuphy_ex_pdsch_tx
        cuphy_ex_polar_encoder
        cuphy_ex_pucch_rx_pipeline
        cuphy_ex_pusch_rateMatch
        cuphy_ex_pusch_rx_multi_pipe
        cuphy_ex_nzp_csirs_rx_multi_cell
        cuphy_ex_simplex_decoder
        cuphy_ex_srs_bfw
        cuphy_ex_srs_chEst
        cuphy_ex_srs_rx_pipeline
        cuphy_ex_srs_tx
        embed_pdcch_tf_signal
        ldpc_encode
        nzp_csi_rs_test
        prach_receiver_multi_cell
        testSS
    COMMENT "Builds all specified cuPHY examples"
)

# ----------------------------------------------------------------------
# Installation
install(DIRECTORY examples/common DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/bfc DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/ch_est DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/channel_eq DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pusch_rx_multi_pipe DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/error_correction DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/ldpc_encode DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pdsch_tx DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pucch_rx_pipeline DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/prach_receiver_multi_cell DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/prach_test DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/ss DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/polar_encoder DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/srs_chEst DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(FILES CMakeLists.release.txt DESTINATION . RENAME CMakeLists.txt)
if (BUILD_DOCS)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs DESTINATION . FILES_MATCHING PATTERN "*")
endif (BUILD_DOCS)
