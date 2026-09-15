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

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_COMMON_FLAGS}")

# cumac_cp app
set(SOURCES
    src/main.cpp
    src/msg_recv.cpp
    src/cumac_cp_handler.cpp
    src/cumac_cp_configs.cpp
    src/cumac_task.cpp
    src/cumac_cp_tv.cpp
    src/cumac_cuda.cu
    src/signal_handler.cpp
    )

message("cuMAC-CP: L2ADAPTER_LIBS=${L2ADAPTER_LIBS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

set(TARGET cumac_cp)
add_executable(${TARGET} ${SOURCES})
target_include_directories(${TARGET} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${HDF5_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../cuMAC/examples/tools)

target_link_libraries(${TARGET} PRIVATE pthread ${L2ADAPTER_LIBS} cumac cumac_msg cuphy nvlog nvipc cuphy_hdf5 cuphy_cpp_header_only nvphy yaml ${HDF5_C_LIBRARIES})
target_link_libraries(${TARGET} PRIVATE cuphydriver)

# Add L2 integration examples
add_subdirectory(examples/L2IntegrationExample)