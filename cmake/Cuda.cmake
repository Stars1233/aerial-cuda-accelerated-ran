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

# ! target_link_cuda A function that links Cuda to the given target
#
# # Example add_executable(main_cuda main.cu) target_compile_features(main_cuda PRIVATE cxx_std_17)
# target_link_libraries(main_cuda PRIVATE project_options project_warnings)
# target_link_cuda(main_cuda)
#
macro(target_link_cuda target)
    # add CUDA to cmake language
    enable_language(CUDA)

    # use the same C++ standard if not specified
    if("${CMAKE_CUDA_STANDARD}" STREQUAL "")
        set(CMAKE_CUDA_STANDARD "${CMAKE_CXX_STANDARD}")
    endif()

    # -fPIC
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # We need to explicitly state that we need all CUDA files in the ${target} library to be built
    # with -dc as the member functions could be called by other libraries and executables
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Resolve device symbols when linking against static libraries containing CUDA code
    set_target_properties(${target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # CUDA 13+ moved CCCL headers (cuda/std/*) to include/cccl/
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 _cccl_base)
        target_include_directories(${target} SYSTEM PUBLIC "${_cccl_base}/cccl")
    endif()

    # Add experimental relaxed constexpr flag for constexpr compatibility in device code Guard
    # relaxed constexpr flag for CUDA >=10.2
    target_compile_options(
        ${target}
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<VERSION_GREATER_EQUAL:${CMAKE_CUDA_COMPILER_VERSION},10.2>>:--expt-relaxed-constexpr>
    )

    # Add -lineinfo for debug builds to enable line-level profiling and debugging
    target_compile_options(
        ${target}
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:--generate-line-info>
    )

endmacro()

# check_cuda_architectures - Fatal error if CUDA architectures not set
function(check_cuda_architectures)
    if(CMAKE_CUDA_COMPILER AND PROJECT_IS_TOP_LEVEL)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            message(
                FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is empty or not set. "
                            "Please specify target GPU architectures (e.g., 'native', '75;80;86').")
        endif()
        message(STATUS "CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"")
    endif()
endfunction()

# cmake-format: off
# Detect CUDA GPU architecture and return standardized compute capability identifier
#
# This function detects the actual CUDA GPU architecture present on the system and returns
# a standardized compute capability identifier for library and binary selection.
#
# Detection method:
#   Uses nvidia-smi to query the GPU compute capability (no compilation required)
#
# Arguments:
#   gpu_arch - Variable name to store the detected GPU architecture
#
# Example:
#   detect_gpu_architecture(GPU_ARCH)
#   # If system has H100 GPU, GPU_ARCH will be set to "sm90"
#   # If system has A100 GPU, GPU_ARCH will be set to "sm80"
#
# cmake-format: on
function(detect_gpu_architecture gpu_arch)
    message(STATUS "Detecting GPU architecture using nvidia-smi...")

    # Query compute capability using nvidia-smi
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE compute_cap
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        RESULT_VARIABLE result)

    if(NOT result EQUAL 0 OR NOT compute_cap)
        message(
            FATAL_ERROR
                "Failed to detect GPU architecture. "
                "Please ensure NVIDIA GPU drivers are installed and nvidia-smi is available.")
    endif()

    # Use only the first GPU's compute capability (important for multi-GPU systems)
    string(REPLACE "\n" ";" compute_cap_list ${compute_cap})
    list(GET compute_cap_list 0 compute_cap_first)

    # Convert "X.Y" to "smXY" format (e.g., "8.0" -> "sm80")
    string(REPLACE "." "" compute_cap_no_dot ${compute_cap_first})
    set(detected_arch "sm${compute_cap_no_dot}")

    message(STATUS "GPU architecture detected: ${detected_arch}")
    set(${gpu_arch}
        ${detected_arch}
        PARENT_SCOPE)
endfunction()

# cmake-format: off
# Resolve GPU architecture from CMAKE_CUDA_ARCHITECTURES or auto-detect
#
# This function determines the GPU architecture to use for runtime file selection.
# If CMAKE_CUDA_ARCHITECTURES is explicitly set (and not "native"), it uses the
# lowest architecture from the list. Otherwise, it auto-detects using nvidia-smi.
#
# Arguments:
#   gpu_arch - Variable name to store the resolved GPU architecture
#
# Behavior:
#   - If CMAKE_CUDA_ARCHITECTURES is set and not "native", "all", or "all-major":
#     * Strips suffixes like "-real" or "-virtual"
#     * Finds the minimum (lowest common denominator) architecture
#     * Returns in smXX format
#   - Otherwise: calls detect_gpu_architecture() for auto-detection
#     ("native", "all", and "all-major" are not concrete architecture lists and
#      cannot resolve to a single smXX string for runtime file selection)
#
# Example:
#   resolve_gpu_architecture(GPU_ARCH)
#   # If CMAKE_CUDA_ARCHITECTURES="80;86;90", GPU_ARCH will be "sm80"
#   # If CMAKE_CUDA_ARCHITECTURES="90-real", GPU_ARCH will be "sm90"
#   # If CMAKE_CUDA_ARCHITECTURES="native" or not set, auto-detects
#
# cmake-format: on
function(resolve_gpu_architecture gpu_arch)
    if(DEFINED CMAKE_CUDA_ARCHITECTURES
       AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
       AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native"
       AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "all"
       AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "all-major")
        # CMAKE_CUDA_ARCHITECTURES is explicitly set (not native)
        message(STATUS "Resolving GPU architecture from CMAKE_CUDA_ARCHITECTURES...")

        set(min_arch "")
        foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            # Strip suffixes like "-real" or "-virtual"
            string(REGEX REPLACE "-real$" "" arch_clean "${arch}")
            string(REGEX REPLACE "-virtual$" "" arch_clean "${arch_clean}")

            # Find minimum (lowest common denominator)
            if(min_arch STREQUAL "" OR arch_clean LESS min_arch)
                set(min_arch "${arch_clean}")
            endif()
        endforeach()

        # Convert to smXX format (e.g., "80" -> "sm80")
        set(resolved_arch "sm${min_arch}")

        string(REPLACE ";" ", " cuda_archs_display "${CMAKE_CUDA_ARCHITECTURES}")
        message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${cuda_archs_display}")
        message(STATUS "Using lowest architecture for runtime: ${resolved_arch}")

        set(${gpu_arch}
            ${resolved_arch}
            PARENT_SCOPE)
    else()
        # Auto-detect GPU architecture
        detect_gpu_architecture(detected_arch)
        set(${gpu_arch}
            ${detected_arch}
            PARENT_SCOPE)
    endif()
endfunction()
