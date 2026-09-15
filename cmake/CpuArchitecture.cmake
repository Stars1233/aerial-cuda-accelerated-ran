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

include_guard(GLOBAL)

# Detect target architecture and return standardized architecture name
#
# This function detects the target architecture and returns a standardized architecture identifier
# that can be used for library paths, package selection, and conditional compilation.
#
# Arguments:
#   output_var - Variable name to store the detected architecture
#
# Supported architectures:
#   - x86_64: Intel/AMD 64-bit (includes amd64)
#   - aarch64: ARM 64-bit (includes arm64)
#   - arm: ARM 32-bit
#   - i386: Intel/AMD 32-bit (includes x86, i[3456]86)
#
function(detect_cpu_architecture output_var)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CMAKE_SYSTEM_PROCESSOR_LOWER)

    if(CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL x86_64
       OR CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL amd64
       OR CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL x64)
        set(detected_arch x86_64)
    elseif(CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL
                                                            arm64)
        set(detected_arch aarch64)
    elseif(CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL arm)
        set(detected_arch arm)
    elseif(CMAKE_SYSTEM_PROCESSOR_LOWER STREQUAL x86 OR CMAKE_SYSTEM_PROCESSOR_LOWER MATCHES
                                                        "^i[3456]86$")
        set(detected_arch i386)
    else()
        message(
            FATAL_ERROR
                "Unsupported CPU architecture: ${CMAKE_SYSTEM_PROCESSOR_LOWER}. "
                "Supported CPU architectures: x86_64, amd64, x64, aarch64, arm64, arm, x86, i[3456]86"
        )
    endif()

    message(STATUS "Target CPU architecture detected: ${detected_arch}")
    set(${output_var}
        ${detected_arch}
        PARENT_SCOPE)
endfunction()

# Get architecture-specific library directory suffix
#
# This function returns the standard library directory suffix used by most Linux distributions for
# multi-arch library installations.
#
# Arguments:
#   arch       - Architecture name (from detect_cpu_architecture)
#   output_var - Variable name to store the library directory suffix
#
# Examples:
#   x86_64  -> x86_64-linux-gnu
#   aarch64 -> aarch64-linux-gnu
#   arm     -> arm-linux-gnueabihf
#   i386    -> i386-linux-gnu
#
function(get_arch_lib_suffix arch output_var)
    if(arch STREQUAL x86_64)
        set(lib_suffix x86_64-linux-gnu)
    elseif(arch STREQUAL aarch64)
        set(lib_suffix aarch64-linux-gnu)
    elseif(arch STREQUAL arm)
        set(lib_suffix arm-linux-gnueabihf)
    elseif(arch STREQUAL i386)
        set(lib_suffix i386-linux-gnu)
    else()
        message(FATAL_ERROR "Unsupported CPU architecture for library suffix: ${arch}. "
                            "Supported CPU architectures: x86_64, aarch64, arm, i386")
    endif()

    set(${output_var}
        ${lib_suffix}
        PARENT_SCOPE)
endfunction()
