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

include(GNUInstallDirs)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

# lib: nvipc
set(SOURCES configs/nv_ipc_config.c
            configs/nv_ipc_yaml_parser.cpp
            common/fd_share.c
            common/array_queue.c
            common/nv_ipc_debug.c
            common/nv_ipc_ring.c
            common/nv_ipc_mempool.c
            common/nv_ipc_shm.c
            common/nv_ipc_sem.c
            common/nv_ipc_efd.c
            common/nv_ipc_epoll.c
            common/shm_logger.c
            common/fh_shm_debug.c
            shm/nv_ipc_shm_if.c
            udp/nv_ipc_udp_if.c
            nv_ipc_common.c)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")

if(NVIPC_CUDA_ENABLE)
    set(SOURCES ${SOURCES} cuda/nv_ipc_cudapool.cu cuda/nv_ipc_cuda_utils.cu)
endif()

if(NVIPC_DPDK_ENABLE)
    set(SOURCES ${SOURCES} dpdk/nv_ipc_dpdk_if.c dpdk/nv_ipc_dpdk_utils.c)
endif()

if(NVIPC_GDRCPY_ENABLE)
    set(SOURCES ${SOURCES} gpudata/nv_ipc_gpudatapool.cu gpudata/nv_ipc_gpudata_utils.cu)
endif()

if(NVIPC_DOCA_ENABLE)
    set(SOURCES ${SOURCES} doca/nv_ipc_doca_if.c doca/nv_ipc_doca_utils.c)
endif()

add_library(nvipc SHARED ${SOURCES})
target_include_directories(nvipc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(nvipc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>)
target_include_directories(nvipc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda>)
target_include_directories(nvipc PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpudata>)

set_target_properties(nvipc PROPERTIES PUBLIC_HEADER
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_compile_options(nvipc PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

if(NVIPC_DPDK_ENABLE OR NVIPC_DOCA_GPUNETIO)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(DPDK REQUIRED libdpdk)
    # Link to DPDK shared libraries
    target_compile_options(nvipc PRIVATE ${DPDK_CFLAGS})
    target_include_directories(nvipc PRIVATE ${DPDK_INCLUDE_DIRS})
    target_link_directories(nvipc PRIVATE ${DPDK_LIBRARY_DIRS})
    target_link_libraries(nvipc PRIVATE -Wl,--no-as-needed)
    target_link_libraries(nvipc PRIVATE ${DPDK_LIBRARIES})
    if(DEFINED DPDK_TARGET)
        add_dependencies(nvipc ${DPDK_TARGET})
    endif()
endif()

if(NVIPC_DOCA_ENABLE)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(DOCA REQUIRED doca doca-libs)
    add_definitions(-DNVIPC_DOCA_ENABLE)
    message("DOCA_CFLAGS=${DOCA_CFLAGS}")
    message("DOCA_INCLUDE_DIRS=${DOCA_INCLUDE_DIRS}")
    message("DOCA_LIBRARY_DIRS=${DOCA_LIBRARY_DIRS}")
    message("DOCA_LIBRARY_DIRS=${DOCA_LIBRARIES}")

    target_compile_options(nvipc PUBLIC ${DOCA_CFLAGS})
    target_include_directories(nvipc PUBLIC ${DOCA_INCLUDE_DIRS})
    target_link_directories(nvipc PUBLIC ${DOCA_LIBRARY_DIRS})
    target_link_libraries(nvipc PUBLIC ${DOCA_LIBRARIES})

    add_definitions(-Wno-deprecated-declarations)
endif()

if(NVIPC_DOCA_GPUNETIO)
    add_definitions(-DNVIPC_DOCA_GPUNETIO)
    target_link_libraries(nvipc PUBLIC -ldoca_gpunetio CUDA::cuda_driver)
endif()

if(NVIPC_GDRCPY_ENABLE)
    target_include_directories(nvipc PRIVATE /usr/local/gdrcopy/include)
    target_link_libraries(nvipc PRIVATE CUDA::cuda_driver -lgdrapi)
endif()

if(BUILD_NVIPC_ONLY)
    # Build a independent libnvipc.so for release to partner
    target_link_libraries(nvipc PUBLIC rt -pthread nvlog_static)
    target_link_libraries(nvipc PUBLIC -static-libstdc++)
else()
    target_link_libraries(nvipc PUBLIC rt -pthread nvlog)
endif()

if(NVIPC_CUDA_ENABLE)
    target_link_libraries(nvipc PRIVATE CUDA::cuda_driver cuda_driver_utils)
endif()

install(TARGETS nvipc
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/..)

if (ENABLE_TESTS)
    add_subdirectory(tests)
endif()
