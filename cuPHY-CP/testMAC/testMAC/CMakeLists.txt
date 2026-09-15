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

set(TESTMAC_CORE_SOURCES
            launch_pattern.cpp
            test_mac_configs.cpp
            fapi_handler.cpp
            fapi_validate.cpp
            test_mac.cpp)

set(TESTMAC_APP_SOURCES
            main.cpp
            dummy.cu)

if (ENABLE_CUMAC AND TARGET cumac)
    list(APPEND TESTMAC_CORE_SOURCES cumac_pattern.cpp cumac_handler.cpp cumac_configs.cpp cumac_validate.cpp)
    # AERIAL_CUMAC_ENABLE is set as PUBLIC on testmac_core (after add_library)
    # to ensure ABI-consistent class layouts (e.g. test_mac_configs) across all consumers.
endif ()

if (TARGET scf_5g_fapi)
    list(APPEND TESTMAC_CORE_SOURCES scf_fapi_handler.cpp)
endif ()

message(VERBOSE "TESTMAC: CORE_SOURCES=${TESTMAC_CORE_SOURCES}")
message(VERBOSE "TESTMAC: APP_SOURCES=${TESTMAC_APP_SOURCES}")

add_library(testmac_core STATIC ${TESTMAC_CORE_SOURCES})

if (ENABLE_CUMAC AND TARGET cumac)
    target_compile_definitions(testmac_core PUBLIC AERIAL_CUMAC_ENABLE)
endif ()

if(ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
    target_compile_definitions(testmac_core PUBLIC ENABLE_CONFORMANCE_TM_PDSCH_PDCCH)
endif()

if (TARGET scf_5g_fapi)
    target_compile_definitions(testmac_core PUBLIC SCF_L2ADAPTER_EXISTS)
endif ()

set(TARGET test_mac)
add_executable(${TARGET} ${TESTMAC_APP_SOURCES})

target_include_directories(testmac_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Find system dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML REQUIRED yaml-0.1)
find_package(Threads REQUIRED)
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
endif()

find_package(gRPC CONFIG QUIET)
if(NOT gRPC_FOUND)
    find_package(gRPC REQUIRED)
endif()

set(TESTMAC_COMMON_LIBS
    Threads::Threads
    nvlog
    nvipc
    nvphy_generic
    app_config
    cuphyoamlib
    cuphy_h
    scf_5g_fapi_h
    oran_utils
    cuphy_examples_common
    cuda_driver_utils
    fmtlog-shared
    fmt
    yaml-cpp
    ${YAML_LIBRARIES}
)

target_link_libraries(testmac_core PRIVATE gRPC::grpc++ gRPC::grpc++_reflection hdf5_serial)
target_link_libraries(testmac_core PUBLIC cuphy_hdf5_h)
target_link_libraries(testmac_core PUBLIC ${TESTMAC_COMMON_LIBS})

if (ENABLE_CUMAC AND TARGET cumac)
    target_link_libraries(testmac_core PUBLIC cumac_msg)
endif()

target_link_libraries(${TARGET} PRIVATE
    testmac_core
)

# Add YAML include directories
target_include_directories(testmac_core PUBLIC ${YAML_INCLUDE_DIRS})
target_include_directories(${TARGET} PRIVATE ${YAML_INCLUDE_DIRS})

if (ENABLE_CUMAC AND TARGET cumac)
    target_link_libraries(${TARGET} PRIVATE nvphy)
endif ()

add_subdirectory(tests)
