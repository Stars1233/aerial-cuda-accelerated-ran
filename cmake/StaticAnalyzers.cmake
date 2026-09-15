# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

include_guard(GLOBAL)

set(_SDK_CLANG_TIDY_CACHE_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/helpers/clang_tidy_cache.py")

# Apply clang-tidy to one explicitly opted-in C++ target. This deliberately uses
# a target property instead of CMAKE_CXX_CLANG_TIDY so ordinary SDK builds are
# unaffected and other targets are never analyzed accidentally.
function(enable_clang_tidy target)
    if(NOT ENABLE_CLANG_TIDY)
        return()
    endif()

    if(NOT TARGET ${target})
        message(FATAL_ERROR "Cannot enable clang-tidy: target '${target}' does not exist")
    endif()

    find_program(_SDK_CLANG_TIDY_PATH NAMES clang-tidy)
    if(NOT _SDK_CLANG_TIDY_PATH)
        message(FATAL_ERROR "clang-tidy requested for '${target}' but executable was not found")
    endif()

    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(_clang_tidy_command
        "${Python3_EXECUTABLE}"
        "${_SDK_CLANG_TIDY_CACHE_SCRIPT}"
        "${_SDK_CLANG_TIDY_PATH}"
        -extra-arg=-Wno-unknown-warning-option
        -extra-arg=-Wno-ignored-optimization-argument
        -extra-arg=-Wno-unused-command-line-argument
        -p
        "${CMAKE_BINARY_DIR}")

    if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "")
        list(APPEND _clang_tidy_command
            -extra-arg=-std=c++${CMAKE_CXX_STANDARD})
    endif()

    if(NOT ENABLE_CLANG_TIDY_CACHE)
        list(APPEND _clang_tidy_command --no-cache)
    endif()

    set_property(TARGET ${target} PROPERTY CXX_CLANG_TIDY "${_clang_tidy_command}")
    message(STATUS "Enabled target-scoped clang-tidy for ${target}")
endfunction()
