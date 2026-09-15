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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

if(NOT TARGET gpu3gppchan)
    message(FATAL_ERROR "The gpu3gppchan Python module requires the existing gpu3gppchan target")
endif()

set(
    ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE
    ""
    CACHE FILEPATH
    "Python interpreter containing nanobind and wheel (empty: auto-detect)"
)
if(ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE)
    set(GPU3GPPCHAN_PYTHON_CANDIDATES
        "${ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE}"
    )
else()
    # Prefer the unversioned Python command before version-specific candidates.
    find_program(GPU3GPPCHAN_SYSTEM_PYTHON NAMES python3 NO_CACHE)
    find_program(GPU3GPPCHAN_VERSIONED_PYTHON
        NAMES python3.12 python3.11 python3.10
        NO_CACHE
    )
    set(GPU3GPPCHAN_PYTHON_CANDIDATES
        "${GPU3GPPCHAN_SYSTEM_PYTHON}"
        "${GPU3GPPCHAN_VERSIONED_PYTHON}"
    )
endif()

set(GPU3GPPCHAN_PYTHON_IMPORT_ERRORS)
foreach(GPU3GPPCHAN_PYTHON_CANDIDATE IN LISTS GPU3GPPCHAN_PYTHON_CANDIDATES)
    if(NOT GPU3GPPCHAN_PYTHON_CANDIDATE OR
       NOT EXISTS "${GPU3GPPCHAN_PYTHON_CANDIDATE}")
        continue()
    endif()

    execute_process(
        COMMAND "${GPU3GPPCHAN_PYTHON_CANDIDATE}" -c
            "import nanobind"
        RESULT_VARIABLE GPU3GPPCHAN_PYTHON_IMPORT_RESULT
        ERROR_VARIABLE GPU3GPPCHAN_PYTHON_IMPORT_ERROR
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(GPU3GPPCHAN_PYTHON_IMPORT_RESULT EQUAL 0)
        set(ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE
            "${GPU3GPPCHAN_PYTHON_CANDIDATE}"
        )
        break()
    endif()
    list(APPEND GPU3GPPCHAN_PYTHON_IMPORT_ERRORS
        "${GPU3GPPCHAN_PYTHON_CANDIDATE}: ${GPU3GPPCHAN_PYTHON_IMPORT_ERROR}"
    )
endforeach()

if(NOT ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE)
    string(JOIN "\n  " GPU3GPPCHAN_PYTHON_IMPORT_ERRORS_TEXT
        ${GPU3GPPCHAN_PYTHON_IMPORT_ERRORS})
    message(FATAL_ERROR
        "Could not find a Python interpreter that imports nanobind. "
        "Set ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE to one that does. Tried:\n  "
        "${GPU3GPPCHAN_PYTHON_IMPORT_ERRORS_TEXT}"
    )
endif()

set(Python3_EXECUTABLE "${ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE}" CACHE FILEPATH
    "Python interpreter for gpu3gppchan" FORCE)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
set(Python_EXECUTABLE "${ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE}" CACHE FILEPATH
    "Python interpreter for gpu3gppchan" FORCE)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT nanobind_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        RESULT_VARIABLE GPU3GPPCHAN_NANOBIND_RESULT
        OUTPUT_VARIABLE GPU3GPPCHAN_NANOBIND_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(GPU3GPPCHAN_NANOBIND_RESULT EQUAL 0)
        set(nanobind_DIR "${GPU3GPPCHAN_NANOBIND_DIR}")
    endif()
endif()

# nanobind probes optional compiler features during discovery. Keep those probes
# active, but do not emit their STATUS output where it may be mistaken for a
# failed test.
if(DEFINED CMAKE_MESSAGE_LOG_LEVEL)
    set(GPU3GPPCHAN_PREVIOUS_MESSAGE_LOG_LEVEL "${CMAKE_MESSAGE_LOG_LEVEL}")
    set(GPU3GPPCHAN_HAD_MESSAGE_LOG_LEVEL TRUE)
endif()
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)
find_package(nanobind CONFIG REQUIRED)
if(GPU3GPPCHAN_HAD_MESSAGE_LOG_LEVEL)
    set(CMAKE_MESSAGE_LOG_LEVEL "${GPU3GPPCHAN_PREVIOUS_MESSAGE_LOG_LEVEL}")
else()
    unset(CMAKE_MESSAGE_LOG_LEVEL)
endif()

set(GPU3GPPCHAN_NANOBIND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/nanobind/gpu3gppchan_nanobind.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nanobind/gpu3gppchan_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nanobind/cuda_array_interface.cpp"
)

nanobind_add_module(
    _gpu3gppchan
    NB_STATIC
    NB_SUPPRESS_WARNINGS
    ${GPU3GPPCHAN_NANOBIND_SOURCES}
)
add_library(gpu3gppchan::python ALIAS _gpu3gppchan)
if(TARGET sdk::options)
    if(NOT TARGET sdk::warnings)
        message(FATAL_ERROR
            "Embedding gpu3gppchan requires both sdk::options and sdk::warnings targets"
        )
    endif()
elseif(TARGET sdk::warnings)
    message(FATAL_ERROR
        "Embedding gpu3gppchan requires both sdk::options and sdk::warnings targets"
    )
else()
    include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/CompileOptions.cmake")
endif()
target_link_libraries(_gpu3gppchan PRIVATE
    gpu3gppchan
    CUDA::cudart
    CUDA::curand
    sdk::options
    sdk::warnings
)
# These wrappers own objects through their exact concrete TDL/CDL types. The
# native classes are polymorphic but are never deleted through a base pointer.
target_compile_options(_gpu3gppchan PRIVATE -Wno-delete-non-virtual-dtor)
target_include_directories(_gpu3gppchan PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/nanobind"
)
target_include_directories(_gpu3gppchan SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)
set_target_properties(_gpu3gppchan PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

set(GPU3GPPCHAN_PYTHON_PACKAGE_INPUTS
    "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu3gppchan/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu3gppchan/channel_api_ref.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu3gppchan/channel_config.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu3gppchan/cuda_utils.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu3gppchan/fading_channel.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu3gppchan/py.typed"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu3gppchan/statistical_channel.py"
)

set(GPU3GPPCHAN_WHEEL_NATIVE_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheel-native")
set(GPU3GPPCHAN_WHEEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheelhouse")
set(GPU3GPPCHAN_WHEEL_STAMP "${CMAKE_CURRENT_BINARY_DIR}/gpu3gppchan_wheel.stamp")
set(GPU3GPPCHAN_PYTHON_VENV "${CMAKE_CURRENT_SOURCE_DIR}/.venv")

# Package the artifacts already produced by CMake. setup.py only copies these
# files into the wheel; it does not configure or compile the native library.
add_custom_command(
    OUTPUT "${GPU3GPPCHAN_WHEEL_STAMP}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf
        "${GPU3GPPCHAN_WHEEL_NATIVE_DIR}" "${GPU3GPPCHAN_WHEEL_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${GPU3GPPCHAN_WHEEL_NATIVE_DIR}" "${GPU3GPPCHAN_WHEEL_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:_gpu3gppchan>" "${GPU3GPPCHAN_WHEEL_NATIVE_DIR}/"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:gpu3gppchan>" "${GPU3GPPCHAN_WHEEL_NATIVE_DIR}/"
    COMMAND "${CMAKE_COMMAND}" -E env
        "GPU3GPPCHAN_NATIVE_DIR=${GPU3GPPCHAN_WHEEL_NATIVE_DIR}"
        "${Python_EXECUTABLE}" -m pip wheel
        --no-build-isolation
        --no-deps
        --wheel-dir "${GPU3GPPCHAN_WHEEL_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${GPU3GPPCHAN_WHEEL_STAMP}"
    DEPENDS
        _gpu3gppchan
        gpu3gppchan
        ${GPU3GPPCHAN_NANOBIND_SOURCES}
        ${GPU3GPPCHAN_PYTHON_PACKAGE_INPUTS}
    COMMENT "Building gpu3gppchan wheel from CMake artifacts"
    VERBATIM
)

add_custom_target(gpu3gppchan_wheel DEPENDS "${GPU3GPPCHAN_WHEEL_STAMP}")
set_target_properties(gpu3gppchan_wheel PROPERTIES
    GPU3GPPCHAN_WHEEL_DIR "${GPU3GPPCHAN_WHEEL_DIR}"
    GPU3GPPCHAN_WHEEL_STAMP "${GPU3GPPCHAN_WHEEL_STAMP}"
)

add_custom_target(gpu3gppchan_python_setup
    COMMAND "${CMAKE_COMMAND}" -E env
        "GPU3GPPCHAN_WHEEL_DIR=${GPU3GPPCHAN_WHEEL_DIR}"
        "GPU3GPPCHAN_PYTHON_EXECUTABLE=${Python_EXECUTABLE}"
        bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_venv.sh"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS
        gpu3gppchan_wheel
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_venv.sh"
    COMMENT "Setting up gpu3gppchan Python virtual environment"
)
set_target_properties(gpu3gppchan_python_setup PROPERTIES
    GPU3GPPCHAN_PYTHON_VENV "${GPU3GPPCHAN_PYTHON_VENV}"
)

# Export the artifact interface to the sibling pyAerial directory.
set(GPU3GPPCHAN_WHEEL_DIR "${GPU3GPPCHAN_WHEEL_DIR}" PARENT_SCOPE)
set(GPU3GPPCHAN_WHEEL_STAMP "${GPU3GPPCHAN_WHEEL_STAMP}" PARENT_SCOPE)
