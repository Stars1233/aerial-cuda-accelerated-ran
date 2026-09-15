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

# Find a substring from a string starting after a given prefix
function(find_substring_by_prefix output prefix input)
    string(FIND "${input}" "${prefix}" prefix_index)
    if("${prefix_index}" STREQUAL "-1")
        message(SEND_ERROR "Could not find ${prefix} in ${input}")
        return()
    endif()
    string(LENGTH "${prefix}" prefix_length)
    math(EXPR start_index "${prefix_index} + ${prefix_length}")
    string(SUBSTRING "${input}" "${start_index}" "-1" _output)
    set("${output}"
        "${_output}"
        PARENT_SCOPE)
endfunction()

# Parse a newline-separated KEY=VALUE string and set each entry as a CMake environment variable.
# Values containing '=' are handled correctly (only the first '=' is used as delimiter).
function(set_env_from_string env_string)
    # replace ; in paths with __sep__ so we can split on ;
    string(REGEX REPLACE ";" "__sep__" env_string_sep_added "${env_string}")

    # the variables are separated by newlines
    string(REGEX REPLACE "\r?\n" ";" env_list "${env_string_sep_added}")

    foreach(env_var ${env_list})
        # split on the first = only (values may contain = themselves)
        string(FIND "${env_var}" "=" eq_index)
        if(eq_index EQUAL -1)
            continue()
        endif()
        string(SUBSTRING "${env_var}" 0 ${eq_index} env_name)
        math(EXPR val_start "${eq_index} + 1")
        string(SUBSTRING "${env_var}" ${val_start} -1 env_value)

        # recover ; in paths
        string(REGEX REPLACE "__sep__" ";" env_value "${env_value}")

        # set env_name to env_value
        set(ENV{${env_name}} "${env_value}")

        # update cmake program path
        if("${env_name}" STREQUAL "PATH")
            list(APPEND CMAKE_PROGRAM_PATH ${env_value})
            set(CMAKE_PROGRAM_PATH ${CMAKE_PROGRAM_PATH} PARENT_SCOPE)
        endif()
    endforeach()
endfunction()

function(get_all_targets var)
    set(targets)
    get_all_targets_recursive(targets ${CMAKE_CURRENT_SOURCE_DIR})
    set(${var}
        ${targets}
        PARENT_SCOPE)
endfunction()

function(get_all_installable_targets var)
    set(targets)
    get_all_targets(targets)
    foreach(_target ${targets})
        get_target_property(_target_type ${_target} TYPE)
        if(NOT ${_target_type} MATCHES ".*LIBRARY|EXECUTABLE")
            list(REMOVE_ITEM targets ${_target})
        endif()
    endforeach()
    set(${var}
        ${targets}
        PARENT_SCOPE)
endfunction()

macro(get_all_targets_recursive targets dir)
    get_property(
        subdirectories
        DIRECTORY ${dir}
        PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(
        current_targets
        DIRECTORY ${dir}
        PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

function(is_verbose var)
    if("${CMAKE_MESSAGE_LOG_LEVEL}" STREQUAL "VERBOSE"
       OR "${CMAKE_MESSAGE_LOG_LEVEL}" STREQUAL "DEBUG"
       OR "${CMAKE_MESSAGE_LOG_LEVEL}" STREQUAL "TRACE")
        set(${var}
            ON
            PARENT_SCOPE)
    else()
        set(${var}
            OFF
            PARENT_SCOPE)
    endif()
endfunction()
