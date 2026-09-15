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

# By default, "${CMAKE_SOURCE_DIR}/aerial-sdk-version" includes "26-2-cubb"
# During packaging, this string will be replaced with a specific version.
# Below cmake code we will read it and populate AERIAL_SDK_VERSION
# The value of AERIAL_SDK_VERSION will be compared against distributed YAML files with
# a released version. This way we make sure that both the software built/running and the
# YAML files came out from the same distribution.
# If users mix old YAMLs (for example) cuphydriver and similar executables will bail out saying that
# the tags that the SW is compiled with, and the YAML files' version attribute - these are having a
# mismatch.
# AERIAL_SDK_VERSION is passed via compile time with -DAERIAL_SDK_VERSION="..." and is used during
# runtime to check the YAML version attribute.

include(${CMAKE_CURRENT_LIST_DIR}/CpuArchitecture.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Utilities.cmake)

# strip_debug_symbols(<target>)
#
# Post-build: extract debug info from <target> into <binary_dir>/debug/<target>.debug,
# strip the binary in place, and embed a GNU debuglink so debuggers can find the symbols.
#
# The debug file is placed in ${CMAKE_CURRENT_BINARY_DIR}/debug/ relative to the calling
# CMakeLists.txt. To load symbols in GDB, configure the search path:
#   (gdb) set debug-file-directory <build_dir>/debug
#
# Safe to call unconditionally: if no debug info exists the strip is a no-op.
# No-op if CMAKE_OBJCOPY or CMAKE_STRIP is not found (e.g. non-GNU toolchains).
function(strip_debug_symbols target)
    if(NOT CMAKE_OBJCOPY OR NOT CMAKE_STRIP)
        message(STATUS "strip_debug_symbols: skipping ${target} — CMAKE_OBJCOPY or CMAKE_STRIP not found")
        return()
    endif()

    set(debugdir "${CMAKE_CURRENT_BINARY_DIR}/debug")

    add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${debugdir}"
            COMMAND ${CMAKE_OBJCOPY} --only-keep-debug "$<TARGET_FILE:${target}>" "${debugdir}/$<TARGET_FILE_NAME:${target}>.debug"
            COMMAND ${CMAKE_STRIP} --strip-debug --strip-unneeded "$<TARGET_FILE:${target}>"
            COMMAND ${CMAKE_OBJCOPY} "--add-gnu-debuglink=${debugdir}/$<TARGET_FILE_NAME:${target}>.debug" "$<TARGET_FILE:${target}>"
            COMMENT "Stripping debug symbols from ${target} into ${debugdir}"
            VERBATIM
    )
endfunction()

function(target_link_dpdk_pmd_directories target)
    set(_dpdk_pmd_dirs)
    foreach(_dpdk_lib_dir ${DPDK_LIBRARY_DIRS})
        file(GLOB _dpdk_pmd_dir_candidates LIST_DIRECTORIES true "${_dpdk_lib_dir}/dpdk/pmds-*")
        list(APPEND _dpdk_pmd_dirs ${_dpdk_pmd_dir_candidates})
    endforeach()

    if(_dpdk_pmd_dirs)
        list(REMOVE_DUPLICATES _dpdk_pmd_dirs)
        list(SORT _dpdk_pmd_dirs COMPARE NATURAL)
        list(GET _dpdk_pmd_dirs -1 _dpdk_pmd_dir)
        target_link_directories(${target} PRIVATE ${_dpdk_pmd_dir})
        message(STATUS "DPDK PMD dir for ${target}: ${_dpdk_pmd_dir}")
    else()
        message(WARNING "No DPDK PMD dirs found under DPDK_LIBRARY_DIRS=${DPDK_LIBRARY_DIRS}")
    endif()
endfunction()

function(read_aerial_sdk_version_file aerial_sdk_version_dir_location)
    if (NOT AERIAL_SDK_VERSION)
        set(AERIAL_SDK_VERSION_FILE "${aerial_sdk_version_dir_location}/aerial-sdk-version")
        if(NOT EXISTS "${AERIAL_SDK_VERSION_FILE}")
            message(FATAL_ERROR "Error cannot find aerial-sdk-version file in directory ${aerial_sdk_version_dir_location}, halting")
        endif()
        file(READ ${AERIAL_SDK_VERSION_FILE} AERIAL_SDK_VERSION)
        string(STRIP "${AERIAL_SDK_VERSION}" AERIAL_SDK_VERSION) # strip \n EOL
        message(STATUS "The content of file: ${AERIAL_SDK_VERSION_FILE} is: ${AERIAL_SDK_VERSION}, assigned to AERIAL_SDK_VERSION")
        set(AERIAL_SDK_VERSION ${AERIAL_SDK_VERSION} PARENT_SCOPE)
    endif () # NOT AERIAL_SDK_VERSION
endfunction()
