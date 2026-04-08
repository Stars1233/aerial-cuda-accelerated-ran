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
project(aerial-fh LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
ENABLE_TESTING()

if (DEFINED ENV{cuBB_SDK})
    set(CUBB_HOME $ENV{cuBB_SDK})
    message(STATUS "Setting CUBB_HOME to ${CUBB_HOME}")
else()
    message(FATAL_ERROR "You must define cuBB_SDK environment variable")
endif()

if (DEFINED DOCA_PATH)
	message(VERBOSE "DOCA_PATH cmake variable is seen from aeril-fh-drive cmake file doca include dirs Includes: ${DOCA_INCLUDE_DIRS}, Lib Dirs: ${DOCA_LIBRARY_DIRS} Libs:${DOCA_LIBRARIES} Cuda Driver Lib:${CUDA_DRIVER_LIBRARY} Cuda RT Lib:${CUDA_CUDART_LIBRARY}")
endif()

if (NOT DEFINED DOCA_PATH)
	message(FATAL_ERROR "DOCA_PATH cmake variable is not seen from aerial-fh-driver cmake file!")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_COMMON_FLAGS}") # -G -lineinfo")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true") # -G -lineinfo")
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_DOCS "Generate Doxygen documentation" OFF)
option(GTEST_POPULATE_CMAKE_TESTS "Populate CMake tests with GTest" ON)
option(ENABLE_DOCA_GPU_COMM "Enable DOCA GPU communication" ON)

add_compile_definitions("ALLOW_EXPERIMENTAL_API")
add_compile_definitions("DOCA_ALLOW_EXPERIMENTAL_API")

if (ENABLE_RIVERMAX)
	add_definitions(-DENABLE_RIVERMAX=1)
endif (ENABLE_RIVERMAX)

set(AERIAL_FH_SRCS
	lib/aerial_fh_driver.cpp
	lib/flow.cpp
    lib/fronthaul.cpp
    lib/fh_mutex.cpp
    lib/gpu_mempool.cpp
	lib/gpu.cpp
	lib/memreg.cpp
	lib/metrics.cpp
	lib/nic.cpp
	lib/pdump_client.cpp
	lib/peer.cpp
	lib/pcap_logger.cpp
	lib/queue_manager.cpp
	lib/queue.cpp
	lib/doca_obj.cpp
	lib/ring_buffer.cpp
	lib/time.cpp
	lib/fh_stats_dump.cpp
	lib/rivermax.cpp
	lib/gpu_comm.cpp
	lib/packet_stats.cpp
)

# Conditionally add DOCA GPU communication CUDA source
if (ENABLE_DOCA_GPU_COMM)
	list(APPEND AERIAL_FH_SRCS lib/gpu_comm_doca.cu)
	add_definitions(-DENABLE_DOCA_GPU_COMM)
	message(STATUS "DOCA GPU communication enabled")
else()
	message(STATUS "DOCA GPU communication disabled - using stubs in gpu_comm.cpp")
endif()

#---- building aerial-fh lib
set(AERIAL_FH_TARGET aerial-fh)
add_library(${AERIAL_FH_TARGET} SHARED ${AERIAL_FH_SRCS})
target_include_directories(${AERIAL_FH_TARGET} PUBLIC include/)
target_link_libraries(${AERIAL_FH_TARGET} PRIVATE CUDA::cudart CUDA::cuda_driver nvlog cuphyoamlib app_config slot_command cuphy_pti aerial_utils)#
target_include_directories(${AERIAL_FH_TARGET} PRIVATE $<BUILD_INTERFACE:${CUBB_HOME}/cuPHY/examples/common>)

#-- DOCA TX
set_target_properties(${AERIAL_FH_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${AERIAL_FH_TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

#-- RIVERMAX
if (ENABLE_RIVERMAX)
	target_include_directories(${AERIAL_FH_TARGET} PRIVATE /usr/local/include/mellanox/) 
	target_link_libraries(${AERIAL_FH_TARGET} PRIVATE rivermax)
endif (ENABLE_RIVERMAX)

# Link to DPDK shared libraries
target_compile_options(${AERIAL_FH_TARGET} PRIVATE ${DPDK_CFLAGS})
target_include_directories(${AERIAL_FH_TARGET} PRIVATE ${DPDK_INCLUDE_DIRS})
target_link_directories(${AERIAL_FH_TARGET} PRIVATE ${DPDK_LIBRARY_DIRS})
target_link_directories(${AERIAL_FH_TARGET} PRIVATE ${DPDK_LIBRARY_DIRS}/dpdk/pmds-23.0)
target_link_libraries(${AERIAL_FH_TARGET} PRIVATE -Wl,--no-as-needed)
target_link_libraries(${AERIAL_FH_TARGET} PRIVATE ${DPDK_LIBRARIES} -lrte_net_mlx5 -lrte_bus_pci -lrte_bus_vdev -lrte_common_mlx5 -lrte_bus_auxiliary -lrte_gpu_cuda)

target_compile_options(${AERIAL_FH_TARGET} PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

# silence warning in dpdk headers
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${AERIAL_FH_TARGET} PRIVATE -Wno-deprecated-volatile)
endif()

#Link to DOCA shared libraries
target_compile_options(${AERIAL_FH_TARGET} PRIVATE ${DOCA_CFLAGS})
target_include_directories(${AERIAL_FH_TARGET} PUBLIC ${DOCA_INCLUDE_DIRS})
target_link_libraries(${AERIAL_FH_TARGET} PRIVATE json-c -L${DOCA_LIBRARY_DIRS} -ldoca_gpunetio -ldoca_common -ldoca_argp -ldoca_eth -ldoca_dpdk_bridge)
# 	target_link_libraries(${AERIAL_FH_TARGET} PRIVATE -Wl,--whole-archive ${DOCA_LIBRARY_DIRS}/libdoca_gpu_device.a  -Wl,--no-whole-archive)
# 	target_include_directories(${AERIAL_FH_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../compression_decompression/comp_decomp_lib/include)

install(TARGETS ${AERIAL_FH_TARGET} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include/aerial-fh-driver)

#---- Header-only INTERFACE library for oran.hpp
add_library(oran_h INTERFACE)
add_library(aerial-fh::oran_h ALIAS oran_h)

target_include_directories(oran_h INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link to CUDA::cudart to get CUDA include directories (oran.hpp includes cuda.h)
target_link_libraries(oran_h INTERFACE CUDA::cudart)
target_compile_features(oran_h INTERFACE cxx_std_20)

#-- RIVERMAX
if (ENABLE_RIVERMAX)
	target_include_directories(${AERIAL_FH_MOCKED_TARGET} PRIVATE /usr/local/include/mellanox/) 
	target_link_libraries(${AERIAL_FH_MOCKED_TARGET} PRIVATE rivermax)
endif (ENABLE_RIVERMAX)

if(DEFINED DPDK_TARGET)
	add_dependencies(${AERIAL_FH_TARGET} ${DPDK_TARGET})
endif()

# if(DEFINED DOCA_TARGET)
#     add_dependencies(${AERIAL_FH_TARGET} ${DOCA_TARGET})
# 	add_dependencies(${AERIAL_FH_MOCKED_TARGET} ${DOCA_TARGET})
# endif()

#---- Submodules

if (ENABLE_RIVERMAX AND ENABLE_TESTS)
	add_subdirectory(test_rmax)
endif (ENABLE_RIVERMAX AND ENABLE_TESTS)

if (ENABLE_TESTS)
	add_subdirectory(test)
endif()

#---- generate documentation
# TODO FIXME: Disable buildling docs, this is currently unmaintained and generates many warnings.
if (0) #if (BUILD_DOCS)
	find_package(Doxygen)
	if (DOXYGEN_FOUND)
		set(DOXYGEN_IN  ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
		set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
		message(STATUS "Configuring ${DOXYGEN_OUT}")
		configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
		add_custom_target(aerial_fh_docs_doxygen ALL
							COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
							WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
							COMMENT "Generating API documentation (doxygen)"
							VERBATIM)
	else(DOXYGEN_FOUND)
		message(FATAL_ERROR "Documentation generation requested, but Doxygen package not found")
	endif(DOXYGEN_FOUND)
endif(0) #endif(BUILD_DOCS)

#---- Aerial metrics
if (AERIAL_METRICS)
	target_link_libraries(${AERIAL_FH_TARGET} PUBLIC aerial_metrics nvlog)
endif()

#---- FH traffic generator
add_subdirectory(app/fh_generator)
