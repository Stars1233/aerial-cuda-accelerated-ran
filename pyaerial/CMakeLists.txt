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

project(pyaerial LANGUAGES C CXX ASM)

# Though this is mentioned in the root CMakeLists.txt, in some projects, this directory
# is being added with add_subdirectory() hence the root CMakeLists.txt is not used and
# these variables are not set.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

# This is by default ON in cuPHY too.
option(ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR "Enable PUSCH per-UE pre-EQ noise-interference power reporting" ON)

if (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=1)
else (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=0)
endif (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

set(SOURCES_CPP pybind11/cuda_array_interface.cpp
                pybind11/pycuphy_srs_chest.cpp
                pybind11/pycuphy_srs_rx.cpp
                pybind11/pycuphy_srs_tx.cpp
                pybind11/pycuphy_srs_util.cpp
                pybind11/pycuphy_csirs_tx.cpp
                pybind11/pycuphy_csirs_rx.cpp
                pybind11/pycuphy_csirs_util.cpp
                pybind11/pycuphy_channel_est.cpp
                pybind11/pycuphy_channel_eq.cpp
                pybind11/pycuphy_noise_intf_est.cpp
                pybind11/pycuphy_cfo_ta_est.cpp
                pybind11/pycuphy_ldpc_decoder.cpp
                pybind11/pycuphy_ldpc_derate_match.cpp
                pybind11/pycuphy_rsrp.cpp
                pybind11/pycuphy_pusch_params.cpp
                pybind11/pycuphy_ldpc_util.cpp
                pybind11/pycuphy_crc_encode.cpp
                pybind11/pycuphy_crc_check.cpp
                pybind11/pycuphy_util.cpp
                pybind11/pycuphy_debug.cpp)
add_library(pycuphycpp ${SOURCES_CPP})
target_link_libraries(pycuphycpp PUBLIC cuphy_channels cuphy ${Python3_LIBRARIES})
target_include_directories(pycuphycpp PRIVATE ../cuPHY/examples/common)

set(SOURCES pybind11/cuda_array_interface.cpp
            pybind11/pycuphy_pybind.cpp
            pybind11/pycuphy_pdsch.cpp
            pybind11/pycuphy_dmrs.cpp
            pybind11/pycuphy_pusch.cpp
            pybind11/pycuphy_csirs_tx.cpp
            pybind11/pycuphy_csirs_rx.cpp
            pybind11/pycuphy_csirs_util.cpp
            pybind11/pycuphy_util.cpp
            pybind11/pycuphy_pdsch_params.cpp
            pybind11/pycuphy_pusch_params.cpp
            pybind11/pycuphy_channel_est.cpp
            pybind11/pycuphy_channel_eq.cpp
            pybind11/pycuphy_noise_intf_est.cpp
            pybind11/pycuphy_cfo_ta_est.cpp
            pybind11/pycuphy_srs_chest.cpp
            pybind11/pycuphy_srs_tx.cpp
            pybind11/pycuphy_srs_rx.cpp
            pybind11/pycuphy_srs_util.cpp
            pybind11/pycuphy_ldpc_decoder.cpp
            pybind11/pycuphy_ldpc_encoder.cpp
            pybind11/pycuphy_ldpc_rate_match.cpp
            pybind11/pycuphy_ldpc_derate_match.cpp
            pybind11/pycuphy_ldpc_util.cpp
            pybind11/pycuphy_crc_encode.cpp
            pybind11/pycuphy_crc_check.cpp
            pybind11/pycuphy_trt_engine.cpp
            pybind11/pycuphy_rsrp.cpp
            pybind11/pycuphy_debug.cpp
            ../cuPHY/src/cuphy/tensor_desc.cpp
)

if (ENABLE_TESTS)
    add_subdirectory(tests/cpp)
endif()

find_package(gsl-lite REQUIRED)
find_package(pybind11 REQUIRED)
pybind11_add_module(_pycuphy ${SOURCES})
target_link_libraries(_pycuphy PUBLIC cuphy_channels cuphy gsl-lite::gsl-lite CUDA::cudart)
target_include_directories(_pycuphy PRIVATE ../cuPHY/examples/common)

set(PYAERIAL_PYCUPHY_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/aerial/pycuphy")
set(PYAERIAL_EXTENSION_STAMP "${CMAKE_CURRENT_BINARY_DIR}/pyaerial_extension.stamp")
set(PYAERIAL_CLEAN_STAGED_LIBS_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CleanStagedLibs.cmake")

# Copy the compiled extension into the Python package tree before pip setup.
# Triggered automatically via pyaerial_extension.stamp when building
# pyaerial_setup (no separate CMake target). For editable installs the
# extension RPATH already points at the build tree, so the private shared
# libraries are resolved from there.
add_custom_command(
    OUTPUT "${PYAERIAL_EXTENSION_STAMP}"
    COMMAND "${CMAKE_COMMAND}"
        -DPYAERIAL_PYCUPHY_PKG_DIR="${PYAERIAL_PYCUPHY_PKG_DIR}"
        -P "${PYAERIAL_CLEAN_STAGED_LIBS_SCRIPT}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:_pycuphy>"
        "${PYAERIAL_PYCUPHY_PKG_DIR}/$<TARGET_FILE_NAME:_pycuphy>"
    COMMAND "${CMAKE_COMMAND}" -E touch "${PYAERIAL_EXTENSION_STAMP}"
    DEPENDS _pycuphy
    COMMENT "Staging _pycuphy extension into pyaerial/src/aerial/pycuphy/"
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PythonEnv.cmake)
if(NOT TARGET gpu3gppchan_wheel)
    if(NOT TARGET gpu3gppchan)
        message(FATAL_ERROR
            "pyAerial requires the existing gpu3gppchan native target"
        )
    endif()
    add_subdirectory(
        "${CMAKE_CURRENT_LIST_DIR}/../testBenches/gpu3GPPChan/python"
        "${CMAKE_CURRENT_BINARY_DIR}/gpu3gppchan-python"
    )
endif()
get_target_property(
    GPU3GPPCHAN_WHEEL_STAMP gpu3gppchan_wheel GPU3GPPCHAN_WHEEL_STAMP
)
get_target_property(
    GPU3GPPCHAN_WHEEL_DIR gpu3gppchan_wheel GPU3GPPCHAN_WHEEL_DIR
)
get_target_property(
    GPU3GPPCHAN_PYTHON_VENV gpu3gppchan_python_setup GPU3GPPCHAN_PYTHON_VENV
)
if(NOT GPU3GPPCHAN_WHEEL_STAMP OR NOT GPU3GPPCHAN_WHEEL_DIR OR
   NOT GPU3GPPCHAN_PYTHON_VENV)
    message(FATAL_ERROR "pyAerial requires the gpu3gppchan_wheel artifact target")
endif()
pyaerial_add_python_env_targets(
    "${PYAERIAL_EXTENSION_STAMP}"
    "${PYAERIAL_PYCUPHY_PKG_DIR}"
    "${PYAERIAL_CLEAN_STAGED_LIBS_SCRIPT}"
    "${GPU3GPPCHAN_WHEEL_STAMP}"
    "${GPU3GPPCHAN_WHEEL_DIR}"
    "${GPU3GPPCHAN_PYTHON_VENV}"
)
