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

# Print the directory of the current CMakeLists.txt file
# message("ORDER_KERNEL_TEST_BENCH Current CMakeLists.txt Directory: ${CMAKE_CURRENT_LIST_DIR}")

# Print the root directory of the project
# message("ORDER_KERNEL_TEST_BENCH Root Project Directory: ${CMAKE_SOURCE_DIR}")

set(ORDER_KERNEL_TEST_BENCH
    src/cuphycp_ex_order_kernel.cpp
    src/order_kernel_test_bench.cpp
    src/yaml_parser.cpp
    ../../cuphydriver/src/uplink/order_cuda_kernels.cu
    )

#---- building Order Kernel Test Bench
set(ORDER_KERNEL_TEST_BENCH_TARGET order_kernel_test_bench)
add_executable(${ORDER_KERNEL_TEST_BENCH_TARGET} ${ORDER_KERNEL_TEST_BENCH})

set_target_properties(${ORDER_KERNEL_TEST_BENCH_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${ORDER_KERNEL_TEST_BENCH_TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Add compile options
if (ENABLE_LINE_INFO)
    target_compile_options(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE -lineinfo)
endif()

target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${CUDA_INCLUDE_DIRS})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo")
if (DEFINED ENV{cuBB_SDK})
    set(CUBB_HOME $ENV{cuBB_SDK})
    message(STATUS "Setting CUBB_HOME to ${CUBB_HOME}")
else()
    message(FATAL_ERROR "You must define cuBB_SDK environment variable")
endif()


# Additional packages
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)

message(VERBOSE "Order Kernel test bench Cmake DOCA Lib Dirs: ${DOCA_LIBRARY_DIRS}")
target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${DOCA_INCLUDE_DIRS})
target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PUBLIC ${HDF5_INCLUDE_DIRS})

target_link_libraries(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE CUDA::cudart CUDA::cuda_driver yaml nvlog gdrapi slot_command ${HDF5_C_LIBRARIES})

target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${DPDK_INCLUDE_DIRS})
target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE /usr/local/gdrcopy/include)
message(VERBOSE "Order Kernel test bench include: ${CUBB_HOME}/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi")
target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${CUBB_HOME}/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi)
message(VERBOSE "Order Kernel test bench include: ${CUBB_HOME}/cuPHY-CP/cuphydriver/include")
target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${CUBB_HOME}/cuPHY-CP/cuphydriver/include)
target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${CUBB_HOME}/cuPHY-CP/compression_decompression/comp_decomp_lib/include)
target_include_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${CUBB_HOME}/cuPHY-CP/ru-emulator/ru_emulator)
 
#Link to DOCA shared libraries
target_link_libraries(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE -L${DOCA_LIBRARY_DIRS} -ldoca_gpunetio -ldoca_gpunetio_device -ldoca_common -ldoca_argp -ldoca_eth)

target_link_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${DPDK_LIBRARY_DIRS})
target_link_directories(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE ${DPDK_LIBRARY_DIRS}/dpdk/pmds-22.1)
target_link_libraries(${ORDER_KERNEL_TEST_BENCH_TARGET} PRIVATE aerial_utils)