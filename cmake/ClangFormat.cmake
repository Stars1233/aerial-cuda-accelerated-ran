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

option(ENABLE_CLANG_FORMAT "Enable clang-format for explicitly opted-in targets" OFF)

if(ENABLE_CLANG_FORMAT AND NOT TARGET clang-format-all)
    add_custom_target(clang-format-all
        COMMENT "Checking all targets opted into clang-format")
    add_custom_target(clang-format-fix-all
        COMMENT "Formatting all targets opted into clang-format")
endif()

function(enable_clang_format target)
    if(NOT ENABLE_CLANG_FORMAT)
        return()
    endif()

    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Cannot enable clang-format: target '${target}' does not exist")
    endif()

    find_program(_SDK_CLANG_FORMAT_EXECUTABLE NAMES clang-format NO_CACHE REQUIRED)

    cmake_parse_arguments(CLANG_FORMAT "" "" "FILES" ${ARGN})
    if(CLANG_FORMAT_UNPARSED_ARGUMENTS OR CLANG_FORMAT_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR
            "Invalid arguments to enable_clang_format('${target}'): "
            "${CLANG_FORMAT_UNPARSED_ARGUMENTS} ${CLANG_FORMAT_KEYWORDS_MISSING_VALUES}")
    endif()
    get_target_property(_clang_format_source_dir "${target}" SOURCE_DIR)

    if(CLANG_FORMAT_FILES)
        set(_clang_format_sources ${CLANG_FORMAT_FILES})
    else()
        get_target_property(_clang_format_sources "${target}" SOURCES)
    endif()

    set(_clang_format_files)
    foreach(_source IN LISTS _clang_format_sources)
        if(_source MATCHES "\\.(c|cc|cpp|cxx|h|hh|hpp|hxx|cu|cuh)$")
            if(IS_ABSOLUTE "${_source}")
                list(APPEND _clang_format_files "${_source}")
            else()
                list(APPEND _clang_format_files
                    "${_clang_format_source_dir}/${_source}")
            endif()
        endif()
    endforeach()

    if(NOT _clang_format_files)
        message(WARNING "clang-format enabled for '${target}', but no supported source files were found")
        return()
    endif()

    set(_check_target "${target}-clang-format-check")
    set(_fix_target "${target}-clang-format-fix")
    if(TARGET "${_check_target}" OR TARGET "${_fix_target}")
        message(FATAL_ERROR "clang-format targets already exist for '${target}'")
    endif()

    set(_check_commands)
    set(_fix_commands)
    foreach(_file IN LISTS _clang_format_files)
        list(APPEND _check_commands
            COMMAND "${_SDK_CLANG_FORMAT_EXECUTABLE}" --dry-run --Werror --style=file "${_file}")
        list(APPEND _fix_commands
            COMMAND "${_SDK_CLANG_FORMAT_EXECUTABLE}" --style=file -i "${_file}")
    endforeach()

    add_custom_target("${_check_target}"
        ${_check_commands}
        WORKING_DIRECTORY "${_clang_format_source_dir}"
        COMMENT "Checking ${target} sources with clang-format"
        VERBATIM)
    add_custom_target("${_fix_target}"
        ${_fix_commands}
        WORKING_DIRECTORY "${_clang_format_source_dir}"
        COMMENT "Formatting ${target} sources with clang-format"
        VERBATIM)

    add_dependencies(clang-format-all "${_check_target}")
    add_dependencies(clang-format-fix-all "${_fix_target}")

    foreach(_file IN LISTS _clang_format_files)
        # Keep the selected file list visible in configure output for CI logs.
        message(VERBOSE "clang-format file: ${_file}")
    endforeach()

    list(LENGTH _clang_format_files _clang_format_file_count)
    message(STATUS
        "Enabled clang-format for ${target}: ${_check_target}, ${_fix_target} (${_clang_format_file_count} file(s))")
endfunction()
