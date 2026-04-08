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

# Enforce C++20 for this example (std::bit_cast, etc.)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -fno-inline")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fno-inline")

add_executable(l2_muUeGrp_test)
target_sources(l2_muUeGrp_test PRIVATE l2_muUeGrp_test.cpp)
target_include_directories(l2_muUeGrp_test PUBLIC ${HDF5_CXX_INCLUDE_DIRS} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(l2_muUeGrp_test PRIVATE ${HDF5_CXX_LIBRARIES} cumac pthread nvlog nvipc yaml cumac_msg)

add_executable(cumac_muUeGrp_test)
target_sources(cumac_muUeGrp_test PRIVATE 
    cumac_muUeGrp_test.cu
    simple_srs_memory_bank.cpp
)

# Add dependency on cuphydriver for header files
add_dependencies(cumac_muUeGrp_test cuphydriver)

target_include_directories(cumac_muUeGrp_test PUBLIC 
    ${HDF5_CXX_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CMAKE_SOURCE_DIR}/cuPHY-CP/cuphydriver/include
    ${CMAKE_SOURCE_DIR}/cuPHY-CP/gt_common_libs/slot_command/include
    ${CMAKE_SOURCE_DIR}/cuPHY/util/aerial_utils
    ${CMAKE_SOURCE_DIR}/cuPHY/examples/common    # For cuphy.hpp
    ${CMAKE_SOURCE_DIR}/cuPHY/src/cuphy_hdf5     # For cuphy_hdf5.h
    ${CMAKE_SOURCE_DIR}/cuPHY/src                # For cuphy/tensor_desc.hpp
    ${CMAKE_SOURCE_DIR}/cuPHY/src/cuphy_channels # For pdsch_tx.hpp, pusch_rx.hpp
    ${CMAKE_SOURCE_DIR}/cuMAC/examples/tools     # For h5TvCreate.h, h5TvLoad.h
)

target_link_libraries(cumac_muUeGrp_test PRIVATE 
    ${HDF5_CXX_LIBRARIES} 
    cumac 
    pthread 
    nvlog 
    nvipc 
    yaml 
    cumac_msg
    cuphydriver          # For GpuDevice, dev_buf, CVSrsChestBuff
    cuphy                # For cuPHY tensor functions
    CUDA::cudart        # For CUDA runtime
    CUDA::cuda_driver   # For CUDA driver API
)

# Simplified SRS Memory Bank Test
# Tests GPU buffer allocation using original CVSrsChestBuff without full infrastructure
add_executable(l1_muUeGrp_test)
target_sources(l1_muUeGrp_test PRIVATE 
    l1_muUeGrp_test.cpp
    simple_srs_memory_bank.cpp
)

# Only need cuphydriver for GpuDevice and CVSrsChestBuff classes
add_dependencies(l1_muUeGrp_test cuphydriver)

target_include_directories(l1_muUeGrp_test PUBLIC 
    ${HDF5_CXX_INCLUDE_DIRS}                     # For hdf5.h
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    ${CMAKE_SOURCE_DIR}/cuPHY-CP/cuphydriver/include
    ${CMAKE_SOURCE_DIR}/cuPHY-CP/gt_common_libs/slot_command/include
    ${CMAKE_SOURCE_DIR}/cuPHY/util/aerial_utils
    ${CMAKE_SOURCE_DIR}/cuPHY/examples/common    # For cuphy.hpp
    ${CMAKE_SOURCE_DIR}/cuPHY/src/cuphy_hdf5     # For cuphy_hdf5.h
    ${CMAKE_SOURCE_DIR}/cuPHY/src                # For cuphy/tensor_desc.hpp
    ${CMAKE_SOURCE_DIR}/cuPHY/src/cuphy_channels # For pdsch_tx.hpp, pusch_rx.hpp
    ${CMAKE_SOURCE_DIR}/cuMAC/examples/tools     # For h5TvCreate.h, h5TvLoad.h
)

target_link_libraries(l1_muUeGrp_test PRIVATE 
    cuphydriver          # For GpuDevice, dev_buf, CVSrsChestBuff
    cuphy                # For cuPHY tensor functions (cuphyDestroyTensorDescriptor, etc.)
    CUDA::cudart        # For CUDA runtime (cudaMalloc/cudaFree)
    CUDA::cuda_driver   # For CUDA driver API
    nvlog               # For logging (used by GpuDevice)
    ${HDF5_CXX_LIBRARIES}  # For HDF5 library
)