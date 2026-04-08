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
# Build options
# ----------------------------------------------------------------------

option(BUILD_DOCS "Generate Doxygen documentation" OFF)
option(ENABLE_NVTX  "Enable use of NVTX" OFF)

# ----------------------------------------------------------------------
# Set the version number for this project
# ----------------------------------------------------------------------

set(cuPHYDriver_VERSION_MAJOR 1)
set(cuPHYDriver_VERSION_MINOR 0)

project(cuPHYDriver LANGUAGES C CXX CUDA)

# ----------------------------------------------------------------------
# Global C++ options
# ----------------------------------------------------------------------

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------
# Additional packages

find_package(CUDAToolkit REQUIRED)
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)

set(MISC_REQ_LIBS
    pthread
    dl
    rt
    m
    numa
    mnl
    data_lake
)

set(CUPHYDRIVER_HOME ${CMAKE_CURRENT_SOURCE_DIR})
if (DEFINED ENV{cuBB_SDK})
    set(CUBB_HOME $ENV{cuBB_SDK})
    message(STATUS "Setting CUBB_HOME to ${CUBB_HOME}")
else()
    message(FATAL_ERROR "You must define cuBB_SDK environment variable")
endif()

if (ENABLE_NVTX)
    add_definitions(-DUSE_NVTX=1)
endif (ENABLE_NVTX)

# ----------------------------------------------------------------------
# Generation of documentation
# ----------------------------------------------------------------------
# TODO FIXME: Disable buildling docs, this is currently unmaintained and generates many warnings.
if (0) #if (BUILD_DOCS)
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        set(DOXYGEN_IN  ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        message(STATUS "Configuring ${DOXYGEN_OUT}")
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        add_custom_target(cuphydriver_docs_doxygen ALL
          COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          COMMENT "Generating API documentation (doxygen)"
          VERBATIM)
    else (DOXYGEN_FOUND)
        message(FATAL_ERROR "Documentation generation requested, but Doxygen package not found")
    endif (DOXYGEN_FOUND)
endif (0) #endif (BUILD_DOCS)

if (BUILD_EXTERNALS)
    add_subdirectory(external)
endif ()

add_subdirectory(src)
add_subdirectory(benchmark)

include_directories(../data_lake/)

# ----------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------

install(TARGETS cuphydriver DESTINATION cuphydriver/lib)
install(FILES include/cuphydriver_api.hpp DESTINATION cuphydriver/include)
# install(TARGETS standalone RUNTIME DESTINATION cuphydriver/examples)
# install(DIRECTORY examples/standalone DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)

