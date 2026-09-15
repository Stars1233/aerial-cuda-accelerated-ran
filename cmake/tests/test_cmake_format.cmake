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

# Developer regression fixture for CMakeFormat.cmake. Run with:
# cmake -DMODULE_UNDER_TEST=$PWD/cmake/CMakeFormat.cmake
#       -DTEST_ROOT=/tmp/aerial-sdk-cmake-format-module-test
#       -P cmake/tests/test_cmake_format.cmake

if(NOT DEFINED MODULE_UNDER_TEST)
    message(FATAL_ERROR "MODULE_UNDER_TEST is required")
endif()

if(NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "TEST_ROOT is required")
endif()

if(NOT EXISTS "${MODULE_UNDER_TEST}")
    message(FATAL_ERROR "CMake-format module not found: ${MODULE_UNDER_TEST}")
endif()

set(test_root "${TEST_ROOT}")
set(source_dir "${test_root}/source")
set(default_build_dir "${test_root}/default-build")
set(enabled_build_dir "${test_root}/enabled-build")
set(tool_dir "${test_root}/bin")
set(tool_log "${test_root}/cmake-format.log")

file(REMOVE_RECURSE "${test_root}")
file(MAKE_DIRECTORY "${source_dir}" "${tool_dir}")
file(WRITE "${source_dir}/.cmake-format.yaml" "format:\n  line_width: 100\n")
file(WRITE "${source_dir}/format_me.cmake" "set(VALUE value)\n")
file(WRITE "${source_dir}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.25)
project(cmake_format_module_test NONE)

include("${CMAKE_FORMAT_MODULE}")
enable_cmake_format(FILES format_me.cmake)
]=])
if(CMAKE_HOST_WIN32)
    set(fake_tool "${tool_dir}/cmake-format.bat")
    file(WRITE "${fake_tool}" "@echo %*>>\"%CMAKE_FORMAT_LOG%\"\r\n")
else()
    set(fake_tool "${tool_dir}/cmake-format")
    file(WRITE "${fake_tool}" [=[#!/bin/sh
printf '%s\n' "$*" >> "$CMAKE_FORMAT_LOG"
]=])
    file(CHMOD "${fake_tool}"
         PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${source_dir}" -B "${default_build_dir}"
            "-DCMAKE_FORMAT_MODULE=${MODULE_UNDER_TEST}"
    RESULT_VARIABLE default_configure_result
    OUTPUT_VARIABLE default_configure_output
    ERROR_VARIABLE default_configure_error)
if(NOT default_configure_result EQUAL 0)
    message(FATAL_ERROR "Default configure failed:\n${default_configure_output}\n${default_configure_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${default_build_dir}" --target help
    RESULT_VARIABLE default_help_result
    OUTPUT_VARIABLE default_help_output
    ERROR_VARIABLE default_help_error)
if(NOT default_help_result EQUAL 0)
    message(FATAL_ERROR "Default target listing failed:\n${default_help_output}\n${default_help_error}")
endif()
if(default_help_output MATCHES "cmake-format-all")
    message(FATAL_ERROR "Default configuration created cmake-format-all")
endif()

cmake_path(CONVERT "$ENV{PATH}" TO_CMAKE_PATH_LIST path_list NORMALIZE)
list(PREPEND path_list "${tool_dir}")
cmake_path(CONVERT "${path_list}" TO_NATIVE_PATH_LIST native_path)
set(ENV{PATH} "${native_path}")
set(ENV{CMAKE_FORMAT_LOG} "${tool_log}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${source_dir}" -B "${enabled_build_dir}"
            "-DCMAKE_FORMAT_MODULE=${MODULE_UNDER_TEST}" "-DENABLE_CMAKE_FORMAT=ON"
    RESULT_VARIABLE enabled_configure_result
    OUTPUT_VARIABLE enabled_configure_output
    ERROR_VARIABLE enabled_configure_error)
if(NOT enabled_configure_result EQUAL 0)
    message(FATAL_ERROR "Enabled configure failed:\n${enabled_configure_output}\n${enabled_configure_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${enabled_build_dir}" --target help
    RESULT_VARIABLE enabled_help_result
    OUTPUT_VARIABLE enabled_help_output
    ERROR_VARIABLE enabled_help_error)
if(NOT enabled_help_result EQUAL 0)
    message(FATAL_ERROR "Enabled target listing failed:\n${enabled_help_output}\n${enabled_help_error}")
endif()
if(NOT enabled_help_output MATCHES "cmake-format-all")
    message(FATAL_ERROR "Enabled configuration did not create cmake-format-all")
endif()
if(NOT enabled_help_output MATCHES "cmake-format-fix-all")
    message(FATAL_ERROR "Enabled configuration did not create cmake-format-fix-all")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${enabled_build_dir}" --target cmake-format-all
    RESULT_VARIABLE check_result
    OUTPUT_VARIABLE check_output
    ERROR_VARIABLE check_error)
if(NOT check_result EQUAL 0)
    message(FATAL_ERROR "CMake-format check failed:\n${check_output}\n${check_error}")
endif()

file(READ "${tool_log}" tool_invocations)
if(NOT tool_invocations MATCHES "--check")
    message(FATAL_ERROR "Check target did not pass --check to cmake-format: ${tool_invocations}")
endif()
set(expected_arguments "${source_dir}/format_me.cmake -c ${source_dir}/.cmake-format.yaml")
string(FIND "${tool_invocations}" "${expected_arguments}" arguments_index)
if(arguments_index EQUAL -1)
    message(FATAL_ERROR "Check target did not pass the file before the root configuration: ${tool_invocations}")
endif()
if(tool_invocations MATCHES "(^| )-i( |$)")
    message(FATAL_ERROR "Check target invoked rewrite mode: ${tool_invocations}")
endif()
