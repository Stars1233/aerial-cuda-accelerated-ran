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

# Creates opt-in cmake-format check and rewrite targets for explicitly listed CMake files.

include_guard(GLOBAL)

option(ENABLE_CMAKE_FORMAT "Enable cmake-format for explicitly opted-in files" OFF)

if(ENABLE_CMAKE_FORMAT)
    find_program(_SDK_CMAKE_FORMAT_EXECUTABLE NAMES cmake-format NO_CACHE REQUIRED)

    add_custom_target(cmake-format-all COMMENT "Checking all files opted into cmake-format")
    add_custom_target(cmake-format-fix-all COMMENT "Formatting all files opted into cmake-format")
endif()

function(enable_cmake_format)
    if(NOT ENABLE_CMAKE_FORMAT)
        return()
    endif()

    cmake_parse_arguments(CMAKE_FORMAT "" "" "FILES" ${ARGN})
    if(CMAKE_FORMAT_UNPARSED_ARGUMENTS OR CMAKE_FORMAT_KEYWORDS_MISSING_VALUES OR NOT CMAKE_FORMAT_FILES)
        message(FATAL_ERROR
            "Invalid arguments to enable_cmake_format(): use FILES <CMake-file> [...]; "
            "unexpected or incomplete arguments: ${CMAKE_FORMAT_UNPARSED_ARGUMENTS} "
            "${CMAKE_FORMAT_KEYWORDS_MISSING_VALUES}")
    endif()

    set(_cmake_format_files)
    foreach(_file IN LISTS CMAKE_FORMAT_FILES)
        if(IS_ABSOLUTE "${_file}")
            set(_absolute_file "${_file}")
        else()
            set(_absolute_file "${CMAKE_CURRENT_SOURCE_DIR}/${_file}")
        endif()

        if(NOT EXISTS "${_absolute_file}")
            message(FATAL_ERROR "Cannot enable cmake-format: file does not exist: ${_absolute_file}")
        endif()
        if(NOT _absolute_file MATCHES "(^|/)CMakeLists\\.txt$|\\.cmake(\\.in)?$")
            message(FATAL_ERROR "Cannot enable cmake-format: unsupported CMake file: ${_absolute_file}")
        endif()

        list(APPEND _cmake_format_files "${_absolute_file}")
    endforeach()

    get_property(_registration_count GLOBAL PROPERTY SDK_CMAKE_FORMAT_REGISTRATION_COUNT)
    if(NOT _registration_count)
        set(_registration_count 0)
    endif()
    math(EXPR _registration_count "${_registration_count} + 1")
    set_property(GLOBAL PROPERTY SDK_CMAKE_FORMAT_REGISTRATION_COUNT "${_registration_count}")

    set(_check_target "cmake-format-check-${_registration_count}")
    set(_fix_target "cmake-format-fix-${_registration_count}")
    set(_config_file "${CMAKE_SOURCE_DIR}/.cmake-format.yaml")

    add_custom_target("${_check_target}"
        COMMAND "${_SDK_CMAKE_FORMAT_EXECUTABLE}" --check ${_cmake_format_files} -c "${_config_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Checking explicitly opted-in CMake files"
        VERBATIM)
    add_custom_target("${_fix_target}"
        COMMAND "${_SDK_CMAKE_FORMAT_EXECUTABLE}" -i ${_cmake_format_files} -c "${_config_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Formatting explicitly opted-in CMake files"
        VERBATIM)

    add_dependencies(cmake-format-all "${_check_target}")
    add_dependencies(cmake-format-fix-all "${_fix_target}")
endfunction()
