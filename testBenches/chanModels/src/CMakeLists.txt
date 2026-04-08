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
set(CHAN_MODELS_SRC
    chanModels.cpp
    chanModelsCommon.cpp
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

find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG PATHS "/usr/local/nvidia/mathdx/25.06/")
# Create library
add_library(chanModels ${CHAN_MODELS_SRC})

# Set library properties
set_property(TARGET chanModels PROPERTY POSITION_INDEPENDENT_CODE ON)
if (BUILD_SHARED_LIBS)
    set_property(TARGET chanModels PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET chanModels PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()


# Link libraries
target_link_libraries(chanModels
    PUBLIC
    ${HDF5_C_LIBRARIES}
    cuda
    curand
    yaml-cpp
    mathdx::cufftdx
)

# Include directories
target_include_directories(chanModels
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
    set_target_properties(chanModels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Set CUDA architectures if not already set
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80-real 90-real)
    endif()
endif()

# Add compile definitions
target_compile_definitions(chanModels
    PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)
