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

include(GNUInstallDirs)

# Include a system directory (which suppresses its warnings).
function(target_include_system_directories target)
    set(multiValueArgs INTERFACE PUBLIC PRIVATE)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

    foreach(scope IN ITEMS INTERFACE PUBLIC PRIVATE)
        foreach(lib_include_dirs IN LISTS ARG_${scope})
            set(_SYSTEM SYSTEM)
            if(${scope} STREQUAL "INTERFACE" OR ${scope} STREQUAL "PUBLIC")
                target_include_directories(
                    ${target} ${_SYSTEM} ${scope} "$<BUILD_INTERFACE:${lib_include_dirs}>"
                    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
            else()
                target_include_directories(${target} ${_SYSTEM} ${scope} ${lib_include_dirs})
            endif()
        endforeach()
    endforeach()

endfunction()

# Include the directories of a library target as system directories (which suppresses their
# warnings).
function(target_include_system_library target scope lib)
    # check if this is a target
    if(TARGET ${lib})
        get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
        if(lib_include_dirs)
            target_include_system_directories(${target} ${scope} ${lib_include_dirs})
        else()
            message(TRACE
                    "${lib} library does not have the INTERFACE_INCLUDE_DIRECTORIES property.")
        endif()
    endif()
endfunction()

# Link a library target as a system library (which suppresses its warnings).
function(target_link_system_library target scope lib)
    # Include the directories in the library
    target_include_system_library(${target} ${scope} ${lib})

    # Link the library
    target_link_libraries(${target} ${scope} ${lib})
endfunction()

# Link multiple library targets as system libraries (which suppresses their warnings).
function(target_link_system_libraries target)
    set(multiValueArgs INTERFACE PUBLIC PRIVATE)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

    foreach(scope IN ITEMS INTERFACE PUBLIC PRIVATE)
        foreach(lib IN LISTS ARG_${scope})
            target_link_system_library(${target} ${scope} ${lib})
        endforeach()
    endforeach()
endfunction()
