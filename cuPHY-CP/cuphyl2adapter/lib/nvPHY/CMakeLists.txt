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

# Header-only library (using CMake INTERFACE) (for now)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

# Set GDB and optimizing option for debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

# Try modern config-based Protobuf first, fall back to legacy FindProtobuf.cmake
find_package(gsl-lite REQUIRED)
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
endif()

# Try modern config-based gRPC first, fall back to legacy FindgRPC.cmake
find_package(gRPC CONFIG QUIET)
if(NOT gRPC_FOUND)
    find_package(gRPC REQUIRED)
endif()

# Lib: nvphy_generic
set(SOURCES_NVPHY_GENERIC
            nv_phy_utils.hpp
            nv_phy_mac_transport.cpp
            nv_phy_epoll_context.cpp
            nv_mac.cpp
            nv_mac_factory.cpp
            nv_mac_factory.hpp)

add_library(nvphy_generic STATIC ${SOURCES_NVPHY_GENERIC})
set_property(TARGET nvphy_generic PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvphy_generic PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
					                            ../../../../cuPHY/src/cuphy)
target_include_directories(nvphy_generic PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(nvphy_generic PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(nvphy_generic PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})
target_link_libraries(nvphy_generic PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(nvphy_generic PUBLIC cuphyoamlib app_config nvlog nvipc)
target_link_libraries(nvphy_generic PUBLIC aerial_utils)

install(TARGETS nvphy_generic DESTINATION ${PROJECT_NAME}/lib)

# ---------------------------------------------------------------------------
# Static library: nvphy_fapi_tasks (compiled once, shared by nvphy and tests)
#
# Keeps nv_cplane_batch_tasks, nv_dl_aggr_tasks, nv_ul_aggr_tasks out of the
# heavy nvphy dependency chain so the test/benchmark builds can link these
# three files without pulling in cuphy, CUDA, or gRPC.
# ---------------------------------------------------------------------------
add_library(nvphy_fapi_tasks STATIC
    nv_cplane_batch_tasks.cpp
    nv_dl_aggr_tasks.cpp
    nv_ul_aggr_tasks.cpp)

set_property(TARGET nvphy_fapi_tasks PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(nvphy_fapi_tasks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

# nvipc and nvlog_static are not yet warning-clean; mark as SYSTEM to suppress
# their header warnings while sdk::warnings enforces strict checks on our code.
# scf::scf_5g_fapi_h is already declared SYSTEM in its own CMakeLists.txt.
target_link_system_libraries(nvphy_fapi_tasks PUBLIC nvipc nvlog_static)
target_link_libraries(nvphy_fapi_tasks PUBLIC scf::scf_5g_fapi_h task_instrumentation)
target_link_libraries(nvphy_fapi_tasks PRIVATE sdk::options sdk::warnings)
# NVLOGE_FMT and similar macros emit VectorCAST instrumentation pragmas that
# GCC does not recognise. Suppress until nvlog migrates away from VectorCAST.
target_compile_options(nvphy_fapi_tasks PRIVATE ${AERIAL_ARCH_TUNE_FLAGS} -Wno-unknown-pragmas)
# Example target opt-in. Re-enable after nvphy_fapi_tasks cppcheck findings are addressed.
# if(ENABLE_CPPCHECK)
#     set_property(TARGET nvphy_fapi_tasks PROPERTY CXX_CPPCHECK ${CUBB_CPPCHECK_COMMAND})
# endif()

# Guinea-pig target for staged clang-tidy rollout. Enable temporarily for local
# validation with -DENABLE_CLANG_TIDY=ON; keep opt-in commented in normal builds.
# enable_clang_tidy(nvphy_fapi_tasks)

# Guinea-pig target for staged clang-format rollout. FILES is optional; omit it
# to derive supported sources from the target's SOURCES property. Enable
# temporarily for local validation with -DENABLE_CLANG_FORMAT=ON; keep opt-in
# commented in normal builds.
# enable_clang_format(nvphy_fapi_tasks FILES
#     nv_cplane_batch_tasks.cpp
#     nv_dl_aggr_tasks.cpp
#     nv_ul_aggr_tasks.cpp)

# Guinea-pig CMake-format opt-in for staged rollout. Enable temporarily with
# -DENABLE_CMAKE_FORMAT=ON after validating the development container image.
# enable_cmake_format(FILES CMakeLists.txt)

# Lib: nvphy
set(SOURCES_NVPHY
            nv_phy_instance.cpp
            nv_phy_mac_transport.cpp
            nv_phy_epoll_context.cpp
            nv_mac.cpp
            nv_phy_module.hpp
            nv_phy_module.cpp
            nv_phy_factory.hpp
            nv_phy_factory.cpp
            nv_fapi_message_storage.hpp
            nv_ue_phy_module.hpp
            nv_ue_phy_module.cpp
            nv_ue_phy_module_factory.hpp
            nv_ue_phy_module_factory.cpp
            nv_phy_driver_proxy.hpp
            nv_phy_driver_proxy.cpp
            nv_simulate_phy_driver.cpp
            nv_tick_generator.hpp
            nv_tick_generator.cpp
            nv_phy_config_option.hpp
            nv_utils.hpp
            nv_phy_limit_errors.hpp
            nv_phy_limit_errors.cpp
            nv_cplane_batch_tasks.hpp
            nv_aggr_task_common.hpp
            nv_dl_aggr_tasks.hpp
            nv_ul_aggr_tasks.hpp
            nv_tx_data_h2d_helpers.hpp)

# Store-replay sources support both FAPI versions.
if(ENABLE_FAPI_STORE_REPLAY)
    list(APPEND SOURCES_NVPHY
        nv_fapi_message_storage.cpp
        nv_phy_non_slot_dispatch.cpp
        nv_phy_slot_dispatch.cpp
        nv_phy_dl_channels.cpp
        nv_phy_ul_channels.cpp
        nv_slot_telemetry.cpp)
endif()

add_library(nvphy STATIC ${SOURCES_NVPHY})
set_property(TARGET nvphy PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(nvphy PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(nvphy PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(nvphy PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(nvphy PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cuphydriver/include/>)
target_compile_options(nvphy PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})
target_link_libraries(nvphy PRIVATE gRPC::grpc++ gRPC::grpc++_reflection)
target_link_libraries(nvphy PUBLIC nvphy_generic cuphydriver scf::scf_5g_fapi_h cuda_driver_utils)
target_link_libraries(nvphy PUBLIC aerial_utils nvphy_fapi_tasks)
target_link_libraries(nvphy PRIVATE cuphy_hdf5 cuphy_cpp_header_only)
target_link_system_libraries(nvphy PRIVATE gsl-lite::gsl-lite)

# scf_5g_*_helpers are only used by the store-replay aggregation sources
# (gated above); link them only when those sources are compiled.
if(ENABLE_FAPI_STORE_REPLAY)
    target_link_libraries(nvphy PRIVATE scf_5g_csirs_helpers scf_5g_ssb_helpers)
endif()

if (TARGET aerial_metrics)
    target_link_libraries(nvphy PRIVATE aerial_metrics)
endif()

install(TARGETS nvphy DESTINATION ${PROJECT_NAME}/lib)

# Apply ENABLE_FAPI_STORE_REPLAY to the nvphy target as PUBLIC so that all
# translation units that include nv_phy_module.hpp (a public header) see the
# same class layout as the compiled library. The flag guards a data member in
# that header, making a uniform definition visible to all consumers mandatory
# to avoid ODR/ABI violations.
target_compile_definitions(nvphy PUBLIC
    $<$<BOOL:${ENABLE_FAPI_STORE_REPLAY}>:ENABLE_FAPI_STORE_REPLAY>
)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
