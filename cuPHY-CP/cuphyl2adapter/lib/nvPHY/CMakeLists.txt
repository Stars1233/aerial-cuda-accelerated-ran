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

# Header-only library (using CMake INTERFACE) (for now)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

# Set GDB and optimizing option for debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

# Try modern config-based Protobuf first, fall back to legacy FindProtobuf.cmake
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
target_link_libraries(nvphy_generic PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(nvphy_generic PUBLIC cuphyoamlib app_config nvlog nvipc)
target_link_libraries(nvphy_generic PUBLIC aerial_utils)

install(TARGETS nvphy_generic DESTINATION ${PROJECT_NAME}/lib)

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
            nv_module_dispatch_factory.hpp
            nv_module_dispatch_factory.cpp
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
            nv_phy_limit_errors.cpp)

add_library(nvphy STATIC ${SOURCES_NVPHY})
set_property(TARGET nvphy PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(nvphy PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(nvphy PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(nvphy PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(nvphy PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cuphydriver/include/>)
target_link_libraries(nvphy PRIVATE gRPC::grpc++ gRPC::grpc++_reflection)
target_link_libraries(nvphy PUBLIC nvphy_generic cuphydriver)
target_link_libraries(nvphy PUBLIC aerial_utils)

install(TARGETS nvphy DESTINATION ${PROJECT_NAME}/lib)

