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

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

if ("${ARCH}" STREQUAL "aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE \
    -Werror=return-type -DALLOW_EXPERIMENTAL_API")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE \
    -Werror=return-type -DALLOW_EXPERIMENTAL_API  -mcmodel=medium")
endif()

set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})

set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/ru_emulator.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/config_parser.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/fh.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/oran.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/tv_parser.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/cplane_rx_cores.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/uplane_rx_cores.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/setup.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/pdsch.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/pbch.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/pdcch.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/csirs.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/packet_timings.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/standalone.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/timing_utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/sectionid_validation.cpp
            )


function(link_target target)
    # ----------------------------------------------------------------------
    #  Include directories
    target_include_directories(${target} PRIVATE /usr/local/include)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator)
    target_include_directories(${target} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_include_directories(${target} PUBLIC ${HDF5_INCLUDE_DIRS})
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/cuPHY-CP/aerial-fh-driver/include)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/cuPHY/src/cuphy)

    target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cuda_driver)

    #  Libraries
    target_link_libraries(${target} PRIVATE slot_command nvlog yaml aerial-fh cuphyoamlib app_config aerial_sdk_version perf_metrics)
    target_link_libraries(${target} PRIVATE  ${HDF5_C_LIBRARIES})

#Link to DOCA shared libraries
if (DOCA_GPU_DPDK)
    target_link_libraries(${target} PRIVATE -L${DOCA_LIBRARY_DIRS} -ldoca_gpunetio -ldoca_common -ldoca_argp -ldoca_eth)
endif(DOCA_GPU_DPDK)

    target_link_libraries(${target} PRIVATE -Wl,--no-as-needed)

    if(${SUBMODULE_BUILD})
        target_compile_definitions(${target} PRIVATE SUBMODULE_BUILD=1)
    endif()
endfunction()
