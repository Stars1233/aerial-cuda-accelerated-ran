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

# lib: aerial_utils

project(aerial_utils LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message("CUDAToolkit_FOUND=${CUDAToolkit_FOUND} CUDAToolkit_VERSION=${CUDAToolkit_VERSION}: Disable CUDA")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")

set(SOURCES memfoot_global.cpp)
add_library(aerial_utils SHARED ${SOURCES})
set_property(TARGET aerial_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(aerial_utils PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(aerial_utils PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(aerial_utils PRIVATE ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} CUDA::cudart)
target_link_libraries(aerial_utils PRIVATE nvlog)