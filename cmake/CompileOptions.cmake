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

# Provides sdk::warnings and sdk::options INTERFACE targets for selective use in new code.
#
# Usage:
#   target_link_libraries(<target> PRIVATE sdk::options sdk::warnings)

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/CompilerWarnings.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/SystemLink.cmake)

# --- cppcheck ------------------------------------------------------------
option(ENABLE_CPPCHECK "Enable cppcheck for nvphy FAPI tasks" ON)

if(ENABLE_CPPCHECK)
    find_program(CUBB_CPPCHECK_EXECUTABLE NAMES cppcheck REQUIRED)

    set(CUBB_CPPCHECK_COMMAND
        ${CUBB_CPPCHECK_EXECUTABLE}
        --template=gcc
        --enable=style,performance,warning,portability
        --inline-suppr
        --inconclusive
        --suppress=cppcheckError
        --suppress=internalAstError
        --suppress=unmatchedSuppression
        --suppress=passedByValue
        --suppress=syntaxError
        --suppress=preprocessorErrorDirective
        --suppress=knownConditionTrueFalse
        --suppress=unknownMacro
        --suppress=*:${CMAKE_CURRENT_BINARY_DIR}/_deps/*.h
        --suppress=*:${CMAKE_CURRENT_BINARY_DIR}/_deps/*.hpp
        --suppress=*:/opt/mellanox/dpdk/include/dpdk/*.h
        --suppress=*:/opt/mellanox/dpdk/include/*/dpdk/*.h
        --suppress=*:/opt/mellanox/doca/include/*.h
        --suppress=*:/usr/include/x86_64-linux-gnu/NvInfer*.h
        --std=c++${CMAKE_CXX_STANDARD}
        --error-exitcode=2)
endif()

# --- sdk::warnings -------------------------------------------------------
# Full aerial-framework compiler warning set. Apply to new targets only.
add_library(sdk_warnings INTERFACE)
add_library(sdk::warnings ALIAS sdk_warnings)

option(ACAR_WARNINGS_AS_ERRORS "Treat compiler warnings as errors for sdk:: targets" ON)
set_project_warnings(sdk_warnings ${ACAR_WARNINGS_AS_ERRORS} "" "" "" "")

# --- sdk::options --------------------------------------------------------
# Enforces C++20 standard. Sanitizer flags are wired in via SanitizerOptions.cmake below.
add_library(sdk_options INTERFACE)
add_library(sdk::options ALIAS sdk_options)

target_compile_features(sdk_options INTERFACE cxx_std_20)

# gsl-lite contract violation policy: throw gsl::fail_fast on violated gsl_Expects/gsl_Ensures.
# Throwing allows callers to catch and recover — correct for a 24/7 telecom process.
# Matches aerial-framework's FRAMEWORK_GSL_CONTRACT_VIOLATION_THROWS=ON default.
target_compile_definitions(sdk_options INTERFACE gsl_CONFIG_CONTRACT_VIOLATION_THROWS)

include(${CMAKE_CURRENT_LIST_DIR}/SanitizerOptions.cmake)
