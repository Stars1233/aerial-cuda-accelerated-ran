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

set(FH_GENERATOR_SRCS
    src/fh_generator.cpp
    src/main.cpp
    src/oran_slot_iterator.cpp
    src/dl_tx_worker.cpp
    src/dl_rx_worker.cpp
    src/ul_tx_worker.cpp
    src/ul_rx_worker.cpp
    src/worker.cpp
    src/yaml_parser.cpp
    src/gpudevice.cpp
    src/grpc.cpp
    src/order_entity.cpp
    src/cell.cpp
    src/packet_counts.cpp
    src/cuda_kernels.cu)

#---- building FH traffic generator
set(FH_GENERATOR_TARGET fh_generator)
add_executable(${FH_GENERATOR_TARGET} ${FH_GENERATOR_SRCS})

set_target_properties(${FH_GENERATOR_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${FH_GENERATOR_TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

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

target_include_directories(${FH_GENERATOR_TARGET} PRIVATE ${CUDA_INCLUDE_DIRS})

if (DEFINED ENV{cuBB_SDK})
    set(CUBB_HOME $ENV{cuBB_SDK})
    message(STATUS "Setting CUBB_HOME to ${CUBB_HOME}")
else()
    message(FATAL_ERROR "You must define cuBB_SDK environment variable")
endif()
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo")

message(VERBOSE "FH Gen Cmake DOCA Lib Dirs: ${DOCA_LIBRARY_DIRS}")
target_include_directories(${FH_GENERATOR_TARGET} PRIVATE ${DOCA_INCLUDE_DIRS})

target_link_libraries(${FH_GENERATOR_TARGET} PRIVATE ${AERIAL_FH_TARGET} CUDA::cudart CUDA::cuda_driver yaml cuphy nvlog gdrapi cuphyoamlib slot_command gRPC::grpc++ protobuf gRPC::grpc++_reflection)
message(VERBOSE "CUDA_LIBRARIES is ${CUDA_LIBRARIES}")
message(VERBOSE "CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")

target_include_directories(${FH_GENERATOR_TARGET} PRIVATE ${DPDK_INCLUDE_DIRS})
target_include_directories(${FH_GENERATOR_TARGET} PRIVATE /usr/local/gdrcopy/include)


target_link_libraries(${FH_GENERATOR_TARGET} PRIVATE -L${DOCA_LIBRARY_DIRS} -ldoca_gpunetio -ldoca_gpunetio_device -ldoca_common -ldoca_argp -ldoca_eth -ldoca_dpdk_bridge)
target_link_directories(${FH_GENERATOR_TARGET} PRIVATE ${DPDK_LIBRARY_DIRS})
target_link_directories(${FH_GENERATOR_TARGET} PRIVATE ${DPDK_LIBRARY_DIRS}/dpdk/pmds-23.0)
