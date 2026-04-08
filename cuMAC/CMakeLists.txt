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
        set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../cuPHY/cmake/toolchains/grace-cross)
    else ()
        set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../cuPHY/cmake/toolchains/devkit)
    endif()
endif()
message(STATUS "cuMAC is using toolchain ${CMAKE_TOOLCHAIN_FILE}")


# ----------------------------------------------------------------------
project(cumac LANGUAGES C CXX CUDA)

message(STATUS "Using GPU architectures ${CMAKE_CUDA_ARCHITECTURES}")

set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/eigen)
include_directories(${EIGEN3_INCLUDE_DIR}
                    ${PROJECT_SOURCE_DIR}/lib/cumac_msg
                    ${PROJECT_SOURCE_DIR}/src
                    ${PROJECT_SOURCE_DIR}/src/4T4R
                    ${PROJECT_SOURCE_DIR}/src/64T64R
                    ${PROJECT_SOURCE_DIR}/src/cpuMatAlg
                    ${PROJECT_SOURCE_DIR}/src/multiCellSrsScheduler
                    ${PROJECT_SOURCE_DIR}/src/pfmSort
                    ${PROJECT_SOURCE_DIR}/src/tools
                    ${PROJECT_SOURCE_DIR}/examples)

find_package(CUDAToolkit REQUIRED)

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
# set(HDF5DIR "/usr/lib/x86_64-linux-gnu/hdf5/serial") # current hdf5 is 1.10.4
# set(HDF5DIR "${HOME_DIR}/mnt/cuMAC/HDF5/1.13.3")

##########  Possible solution: HDF5 direct read to GPU memoery
# https://github.com/hpc-io/vfd-gds, requires HDF5 1.13.0

##########   using a different HDF5 version, build libs from cmake
#   wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-1.13.3/src/CMake-hdf5-1.13.3.tar.gz
#   tar -xf CMake-hdf5-1.13.3.tar.gz
#   cd CMake-hdf5-1.13.3
#   ./build-unix.sh
#   yes | ./HDF5-1.13.3-Linux.sh
#   cd ./HDF5-1.13.3-Linux
#   mv ./HDF5-1.13.3-Linux/HDF_Group/HDF5/ <some dir>

# set(HDF5DIR "<some dir>/1.13.3")

add_subdirectory(lib/cumac_msg)
add_subdirectory(examples)
add_subdirectory(src)

# ----------------------------------------------------------------------
# build chanModels library from testBenches/chanModels/src if it does not exist
if (NOT TARGET chanModels)
    add_subdirectory(../testBenches/chanModels/src examples/chanModels/src)
endif()
set(CHAN_MODELS_SRC_INCLUDE_DIR ../testBenches/chanModels/src)

# Add a target to build all cuMAC examples with a single command.
# This assumes that targets named 'cellAssociate' and 'channInput' are defined
# by the 'examples' subdirectory (e.g., from examples/cellAssociate/CMakeLists.txt
# and examples/channInput/CMakeLists.txt respectively).
# If other examples exist with different target names, they should be added to the DEPENDS list.
add_custom_target(cumac_examples ALL
    DEPENDS
        cellAssociateTest
        channInputTest
        cpuMatAlg
        cusolverSvd
        eigenMatOp
        drlMcsSelection
        multiCellMuMimoScheduler
        multiCellSchedulerUeSelection
        multiCellSrsScheduler
        scheduleTrafficTest
        singleCellSchedule
        pfmSortTest
        trafficModel
        trtEngine
        tvLoadingTest
    COMMENT "Builds all specified cuMAC examples"
)

# ----------------------------------------------------------------------
# Installation
install(DIRECTORY examples/cellAssociate DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/channInput DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
