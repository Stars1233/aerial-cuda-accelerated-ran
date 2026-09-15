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

# Source files
set(GPU3GPPCHAN_SRC
    gpu3gppchan.cpp
    gpu3gppchanCommon.cpp
    fastFadingCommon.cu
    fading_chan.cu
    ofdm_src/ofdmMod.cu
    ofdm_src/ofdmDemod.cu
    tdl_chan_src/tdl_chan.cu
    cdl_chan_src/cdl_chan.cu
    sls_chan_src/sls_chan.cu
    sls_chan_src/sls_chan_topology.cu
    sls_chan_src/sls_chan_large_scale.cu
    sls_chan_src/sls_chan_small_scale.cu
    sls_chan_src/sls_chan_large_scale_GPU.cu
    sls_chan_src/sls_chan_small_scale_GPU.cu
    sls_chan_src/sls_chan_isac.cpp
    sls_chan_src/sls_chan_ue_mobility.cpp
    gauNoiseAdder.cu
)

find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG PATHS "/usr/local/nvidia/mathdx/26.03/")
# Create library
add_library(gpu3gppchan ${GPU3GPPCHAN_SRC})

# Set library properties
set_property(TARGET gpu3gppchan PROPERTY POSITION_INDEPENDENT_CODE ON)
if (BUILD_SHARED_LIBS)
    set_property(TARGET gpu3gppchan PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET gpu3gppchan PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()


# Link libraries
target_link_libraries(gpu3gppchan
    PUBLIC
    ${HDF5_C_LIBRARIES}
    CUDA::cuda_driver
    CUDA::curand
    yaml-cpp
    mathdx::cufftdx
)

# Include directories
target_include_directories(gpu3gppchan
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../cuPHY/examples/common
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}
    ${HDF5_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/ofdm_src
    ${CMAKE_CURRENT_SOURCE_DIR}/tdl_chan_src
    ${CMAKE_CURRENT_SOURCE_DIR}/cdl_chan_src
    ${CMAKE_CURRENT_SOURCE_DIR}/sls_chan_src
)

# Add CUDA specific flags
if(CUDA_FOUND)
    set_target_properties(gpu3gppchan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Set CUDA architectures if not already set
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80-real 90-real)
    endif()
endif()

# Add compile definitions
target_compile_definitions(gpu3gppchan
    PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Parameter vintage: default is TR 38.901 V19. Turn this ON to use the
# Rel-18 values for the parameters the Rel-19 7-24 GHz study changed (UMa/UMi
# Table 7.5-6 DS/ASD/ASA/ZSA marginals, UMa c_ASD, Table 7.4.3-1 L_IRRglass) — needed when
# calibrating against pre-Rel-19 references. ISAC (7.9) exists only from Rel-19, so ISAC
# runs should keep this OFF.
option(TR38901_REL18_PARAMS "Use 3GPP TR 38.901 Rel-18 parameter values (pre-Rel-19 calibration)" OFF)
if(TR38901_REL18_PARAMS)
    target_compile_definitions(gpu3gppchan PRIVATE SLS_38901_REL18_PARAM)
endif()
