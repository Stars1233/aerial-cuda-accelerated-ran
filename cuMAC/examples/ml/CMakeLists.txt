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

cmake_minimum_required(VERSION 3.25)
# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Though this is mentioned in the root CMakeLists.txt, in some projects, this directory
# is being added with add_subdirectory() hence the root CMakeLists.txt is not used and
# these variables are not set.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE MATCHES "Release" OR ENABLE_LINE_INFO)
    list(APPEND CUMAC_NVCC_FLAGS -lineinfo)
endif()

if (DEFINED CUMAC_GENCODE_ARCH_LIST)
    message(WARNING "CUMAC_GENCODE_ARCH_LIST is deprecated. Please use CMAKE_CUDA_ARCHITECTURES instead")
    string(REPLACE "," ";" CUMAC_GENCODE_ARCH_LIST_ ${CUMAC_GENCODE_ARCH_LIST})
    set(CMAKE_CUDA_ARCHITECTURES ${CUMAC_GENCODE_ARCH_LIST_})
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80-real 90-real)
endif()

block()
    # Remove -real and -virtual extensions from the native variable,
    # which includes compile time native values
    # strip "-real" and "-virtual" and rewrite it to CMAKE_CUDA_ARCHITECTURES_NO_EXT
    # First take the list (semicolon separated values) and make it a string, replacing ';' -> ' '
    # This is why "${CMAKE_CUDA_ARCHITECTURES}" is in quotes!
    # strip virtual and real
    # place back the ';' so it is a string that we can use IN_LIST
    macro(cuda_arch_no_extensions)
        string(REPLACE ";" " " CMAKE_CUDA_ARCHITECTURES_NO_EXT "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
        string(REPLACE "-real" "" CMAKE_CUDA_ARCHITECTURES_NO_EXT ${CMAKE_CUDA_ARCHITECTURES_NO_EXT})
        string(REPLACE "-virtual" "" CMAKE_CUDA_ARCHITECTURES_NO_EXT ${CMAKE_CUDA_ARCHITECTURES_NO_EXT})
        string(REPLACE " " ";" CMAKE_CUDA_ARCHITECTURES_NO_EXT ${CMAKE_CUDA_ARCHITECTURES_NO_EXT})
    endmacro()
    if ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "native")
        cuda_arch_no_extensions()
    else ()
        set(CMAKE_CUDA_ARCHITECTURES_NO_EXT ${CMAKE_CUDA_ARCHITECTURES})
    endif ()

    if (("86" IN_LIST CMAKE_CUDA_ARCHITECTURES_NO_EXT) OR
        ("89" IN_LIST CMAKE_CUDA_ARCHITECTURES_NO_EXT))
        add_compile_definitions(CUDA_ARCH_86_89)
    endif()

    # Blackwell (sm_100, sm_120, sm_121) also needs reduced blocks per SM
    if (("100" IN_LIST CMAKE_CUDA_ARCHITECTURES_NO_EXT) OR
        ("120" IN_LIST CMAKE_CUDA_ARCHITECTURES_NO_EXT) OR
        ("121" IN_LIST CMAKE_CUDA_ARCHITECTURES_NO_EXT))
        add_compile_definitions(CUDA_ARCH_100_120_121)
    endif()
endblock()

# Get system ARCH
execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
string(STRIP ${ARCH} ARCH)

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if ("${ARCH}" STREQUAL "aarch64")
        set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../../cuPHY/cmake/toolchains/grace-cross)
    else ()
        set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../../cuPHY/cmake/toolchains/devkit)
    endif()
endif()
message(STATUS "cumac_ml is using toolchain ${CMAKE_TOOLCHAIN_FILE}")

# ----------------------------------------------------------------------
project(cumac_ml LANGUAGES C CXX CUDA)

message(STATUS "cumac_ml is using GPU architectures ${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

find_library(NVINFER libnvinfer.so)
find_library(NVONNXPARSER libnvonnxparser.so)

# -------------------------------------------------------------------------
# HDF5 related library
# Allow the user to specify their own HDF5 target location,
# for example, if building from source instead of using a
# development version installed in a system path.
if (NOT DEFINED HDF5_CXX_LIBRARIES)
  find_package(HDF5 REQUIRED COMPONENTS CXX)
else()
  message(STATUS "Using HDF5 CXX target ${HDF5_CXX_LIBRARIES}.")
endif()

# Add this check:
if (NOT HDF5_CXX_INCLUDE_DIRS)
  message(FATAL_ERROR "HDF5 CXX include directories not found. Please install HDF5 with C++ support.")
endif()

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    include_directories(../../../cuPHY/external)
endif()

add_subdirectory(trtEngine)
add_subdirectory(drlMcsSelection)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${cumac_SOURCE_DIR}/examples/tools)
include_directories(${cumac_SOURCE_DIR}/src)
include_directories(${HDF5_CXX_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# =============================================================================
# Build Configuration
# =============================================================================
# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA-specific options
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(trtEngine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --cudart static
        --expt-relaxed-constexpr
    >)
    target_compile_options(drlMcsSelection PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --cudart static
        --expt-relaxed-constexpr
    >)
endif()

# Print configuration summary
message(STATUS "==========================================")
message(STATUS "nonOpenSourceModules Build Configuration")
message(STATUS "==========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "HDF5 CXX libraries: ${HDF5_CXX_LIBRARIES}")
message(STATUS "TensorRT libraries: ${NVINFER}, ${NVONNXPARSER}")
message(STATUS "==========================================")
