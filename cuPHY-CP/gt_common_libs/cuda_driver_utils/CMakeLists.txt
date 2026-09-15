# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Lib: cuda_driver_utils (header-only, thin CUDA driver API wrapper)
add_library(cuda_driver_utils INTERFACE)
target_include_directories(cuda_driver_utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(cuda_driver_utils INTERFACE
    nvlog
    CUDA::cuda_driver
)

# sdk::options provides ASan/sanitizer flags when building within the cuBB tree.
# Guard with TARGET check so standalone nvipc builds (nvipc_src container) are unaffected.
if(TARGET sdk::options)
    target_link_libraries(cuda_driver_utils INTERFACE sdk::options)
endif()

# Check if cuProfilerStart is declared in cuda.h (absent in stub headers e.g. sbsa-linux).
# Stub headers (targets/sbsa-linux/include) omit cuProfilerStart; skip check and use fallback.
string(FIND "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" "sbsa-linux" _sbsa_pos)
if(_sbsa_pos GREATER_EQUAL 0)
    set(HAVE_CU_PROFILER_START 0)
    message(STATUS "HAVE_CU_PROFILER_START=0 (sbsa-linux cuda.h - cuProfilerStart not declared, using Runtime API fallback)")
else()
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(CMAKE_REQUIRED_LIBRARIES CUDA::cuda_driver)
    check_cxx_source_compiles("
#include <cuda.h>
int main(void) {
    (void)cuProfilerStart();
    return 0;
}
" HAVE_CU_PROFILER_START)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    message(STATUS "HAVE_CU_PROFILER_START=${HAVE_CU_PROFILER_START} (CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES=${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})")
endif()

if(HAVE_CU_PROFILER_START)
    message(STATUS "cuProfilerStart is available (Driver API)")
    target_compile_definitions(cuda_driver_utils INTERFACE HAVE_CU_PROFILER_START)
else()
    message(STATUS "cuProfilerStart is NOT available - using Runtime API fallback (stub cuda.h?)")
endif()
