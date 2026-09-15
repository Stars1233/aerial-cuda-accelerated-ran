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

# Bring in backward-cpp
find_package(Backward REQUIRED)
find_package(gsl-lite REQUIRED)

set(TARGET cuphydriver)
message(VERBOSE "Cuphydriver Cmake DOCA Lib Dirs: ${DOCA_LIBRARY_DIRS}")
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#C / CXX flags
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE ${CMAKE_COMMON_FLAGS}")
set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#CUDA flags
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_COMMON_FLAGS} -lineinfo")

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#SOURCES
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --


set(SOURCES
	common/cuphydriver_api.cpp
	common/generic_cuda_kernels.cu
	common/time.cpp
	common/worker.cpp
	common/cell.cpp
	common/gpudevice.cpp
	common/task.cpp
	common/context.cpp
	common/fh.cpp
	common/mps.cpp
	common/phychannel.cpp
	common/memfoot.cpp
	common/metrics.cpp

	uplink/task_function_ul_aggr.cpp
	uplink/slot_map_ul.cpp
	uplink/order_cuda_kernels.cu
	uplink/phypusch_aggr.cpp
	uplink/phypucch_aggr.cpp
	uplink/phyprach_aggr.cpp
	uplink/phyulbfw_aggr.cpp
	uplink/physrs_aggr.cpp
	uplink/order_entity.cpp
	uplink/ulbuffer.cpp
	uplink/harq_pool.cpp
	uplink/wavgcfo_pool.cpp
	uplink/cv_memory_bank_srs_chest.cpp
	uplink/srs_ipc_manager.cpp
	uplink/ul_pcap_capture_thread.cpp

	downlink/dlbuffer.cpp
	downlink/pdsch_h2d_copy_manager.cpp
	downlink/phypdsch_aggr.cpp
	downlink/phypdcch_aggr.cpp
	downlink/phypbch_aggr.cpp
	downlink/phycsirs_aggr.cpp
	downlink/phydlbfw_aggr.cpp
	downlink/task_function_dl_aggr.cpp
	downlink/task_function_dl_val.cpp
	downlink/slot_map_dl.cpp
	downlink/dl_validation_params.cpp
)

# message ( STATUS "SOURCES ${SOURCES}" )

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#TARGET
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

add_library(${TARGET} SHARED ${SOURCES})
set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Offload (nonslot_lp) active-cell PRACH reconfiguration is part of the
# store-and-replay feature, so its sources compile only under
# ENABLE_FAPI_STORE_REPLAY. The PrachOffloadReconfig member in context.hpp is
# gated on the same macro; propagate it PUBLIC so out-of-tree includers of
# context.hpp that link cuphydriver (e.g. tests/dlc_test_bench) agree with
# cuphydriver's own TUs on PhyDriverCtx's layout (avoids an ODR mismatch).
target_compile_definitions(${TARGET} PUBLIC
    $<$<BOOL:${ENABLE_FAPI_STORE_REPLAY}>:ENABLE_FAPI_STORE_REPLAY>)
if(ENABLE_FAPI_STORE_REPLAY)
    target_sources(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/prach_offload_reconfig.cpp)
endif()

# FrameworkCPlaneService: real impl when aerial-framework is available, stub otherwise.
# Gated on AERIAL_FRAMEWORK_AVAILABLE (not `TARGET ran::oran`) because this subdirectory
# is processed before aerial-framework is added; target_link_libraries resolves ran::oran
# later at generate time.
if(ENABLE_FAPI_STORE_REPLAY AND AERIAL_FRAMEWORK_AVAILABLE)
    target_sources(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/direct_uplane_modcomp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/common/framework_cplane_service.cpp)
    target_link_libraries(${TARGET} PRIVATE ran::oran)
    # nv_phy_mac_transport.hpp lives under cuphyl2adapter (header-only dependency).
    target_include_directories(${TARGET} PRIVATE
        $<BUILD_INTERFACE:${CUBB_HOME}/cuPHY-CP/cuphyl2adapter/lib/nvPHY>)
    # ran::oran's static archive leaks -lrte_* names but not DPDK's -L path.
    target_link_directories(${TARGET} PRIVATE ${DPDK_LIBRARY_DIRS})
else()
    target_sources(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common/framework_cplane_service_stub.cpp)
endif()

# Always compile the stub to verify it satisfies FrameworkCPlaneServiceContract.
# This OBJECT library is never linked — it exists solely for compile-time checking.
add_library(fcs_interface_check OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/common/framework_cplane_service_stub.cpp)
set_property(TARGET fcs_interface_check PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(fcs_interface_check PRIVATE
    $<BUILD_INTERFACE:${CUPHYDRIVER_HOME}/include>)
target_link_libraries(fcs_interface_check PRIVATE scf::scf_5g_fapi_h)

if (TARGET aerial_metrics)
	target_link_libraries(${TARGET} PRIVATE aerial_metrics)
	target_compile_definitions(${TARGET} PRIVATE AERIAL_METRICS)
endif()

set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#Include directories
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

target_include_directories(${TARGET} PRIVATE /usr/local/include)
target_include_directories(${TARGET} PRIVATE /usr/local/gdrcopy/include)
target_include_directories(${TARGET} PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(${TARGET} PUBLIC  $<BUILD_INTERFACE:${CUPHYDRIVER_HOME}/include>)

target_compile_options(${TARGET} PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})
target_compile_options(${TARGET} PRIVATE
    $<$<AND:$<COMPILE_LANG_AND_ID:C,GNU,Clang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-g3>
    $<$<AND:$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-g3>
)

if (DEFINED CUBB_HOME)
	target_include_directories(${TARGET} PRIVATE $<BUILD_INTERFACE:${CUBB_HOME}/cuPHY/examples/common>)
	#Super ugly, can't link properly header-only compression decompression lib
	target_include_directories(${TARGET} PRIVATE $<BUILD_INTERFACE:${CUBB_HOME}/cuPHY-CP/compression_decompression/comp_decomp_lib/include>)
	target_include_directories(${TARGET} PRIVATE $<BUILD_INTERFACE:${CUBB_HOME}/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi>)
	
	# E3 Agent dependency
	find_package(nlohmann_json REQUIRED)
	target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)
else ()
	target_include_directories(${TARGET} PRIVATE $<BUILD_INTERFACE:${CUPHYDRIVER_HOME}/../compression_decompression/comp_decomp_lib/include>)
	target_include_directories(${TARGET} PRIVATE $<BUILD_INTERFACE:${CUPHYDRIVER_HOME}/scfl2adapter/lib/scf_5g_fapi>)
endif()

#Super ugly, can't link properly header-only compression decompression lib
target_include_directories(${TARGET} PRIVATE $<BUILD_INTERFACE:${CUPHYDRIVER_HOME}/../compression_decompression/comp_decomp_lib/include>)



#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#Library directories
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#target_link_directories(${TARGET} PRIVATE $ENV{CUPHY_PATH} / lib)

#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#Libraries
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

target_link_libraries(${TARGET} PRIVATE CUDA::cudart CUDA::cuda_driver ${MISC_REQ_LIBS})
target_link_libraries(${TARGET} PRIVATE aerial-fh cuphy_channels cuphy cuphy_hdf5 cuphy_cpp_header_only ${NVTX_LIB})
target_link_libraries(${TARGET} PRIVATE gsl-lite::gsl-lite)
target_link_libraries(${TARGET} PUBLIC slot_command nvlog oran_utils cuda_driver_utils app_config app_utils comp_decomp_lib Backward::Backward nvipc)
target_link_libraries(${TARGET} PRIVATE cuphyoamlib gdrapi perf_metrics)
target_link_libraries(${TARGET} PUBLIC task_instrumentation pmu_reader)
target_link_libraries(${TARGET} PRIVATE aerial_common)

target_include_directories(${TARGET} PUBLIC ${DOCA_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE json-c -L${DOCA_LIBRARY_DIRS} -ldoca_gpunetio -ldoca_common -ldoca_gpunetio_device -ldoca_argp -ldoca_eth)
# Temporary solution waiting for header file to expose CUDA device functions for Order Kernel
# target_link_libraries(${TARGET} PRIVATE -Wl,--whole-archive ${DOCA_LIBRARY_DIRS}/libdoca_gpu_device.a  -Wl,--no-whole-archive)

#target_link_libraries(${TARGET} PRIVATE comp_decomp_lib)
