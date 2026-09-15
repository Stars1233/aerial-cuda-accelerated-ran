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

# Sanitizer support for sdk::options targets (new code only).
# Legacy targets that do not link sdk::options are unaffected.
#
# Usage: pass -DACAR_ENABLE_SANITIZER_ADDRESS=ON (and/or other flags) at configure time.
# Note: presets for sanitizer builds are a separate follow-on task.

include_guard(GLOBAL)

include(CheckCXXSourceCompiles)
include(CMakeDependentOption)           # needed for cmake_dependent_option() below
include(${CMAKE_CURRENT_LIST_DIR}/Sanitizers.cmake)

# --- Platform capability check -----------------------------------------------
macro(acar_supports_sanitizers)
    if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*")
       AND NOT WIN32)

        message(STATUS
            "Sanity checking UndefinedBehaviorSanitizer, it should be supported on this platform")
        set(TEST_PROGRAM "int main() { return 0; }")

        set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined")
        set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
        check_cxx_source_compiles("${TEST_PROGRAM}" HAS_UBSAN_LINK_SUPPORT)

        if(HAS_UBSAN_LINK_SUPPORT)
            message(STATUS "UndefinedBehaviorSanitizer is supported at both compile and link time.")
            set(SUPPORTS_UBSAN ON)
        else()
            message(WARNING "UndefinedBehaviorSanitizer is NOT supported at link time.")
            set(SUPPORTS_UBSAN OFF)
        endif()
        # Must unset after each probe — stale CMAKE_REQUIRED_* pollutes subsequent
        # find_package() calls (e.g. GTest, benchmark) that also invoke check_cxx_source_compiles.
        unset(CMAKE_REQUIRED_FLAGS)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)

        message(STATUS
            "Sanity checking AddressSanitizer, it should be supported on this platform")
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
        set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address")
        check_cxx_source_compiles("${TEST_PROGRAM}" HAS_ASAN_LINK_SUPPORT)

        if(HAS_ASAN_LINK_SUPPORT)
            message(STATUS "AddressSanitizer is supported at both compile and link time.")
            set(SUPPORTS_ASAN ON)
        else()
            message(WARNING "AddressSanitizer is NOT supported at link time.")
            set(SUPPORTS_ASAN OFF)
        endif()
        unset(CMAKE_REQUIRED_FLAGS)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
    else()
        set(SUPPORTS_UBSAN OFF)
        set(SUPPORTS_ASAN OFF)
    endif()

    # MSan is Clang-only — GCC silently ignores the option in enable_sanitizers().
    # Gate it on Clang detection so the option is never visible on GCC builds.
    # No compile/link probe here (unlike ASan/UBSan): a -fsanitize=memory probe
    # would fail on any standard system because it requires a fully MSan-instrumented
    # libc++, which is not the default. The compiler-ID check is the correct semantic
    # gate; full MSan setup is an explicit user responsibility.
    if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*")
        set(SUPPORTS_MSAN ON)
    else()
        set(SUPPORTS_MSAN OFF)
    endif()
endmacro()

# --- Run platform check (sets SUPPORTS_ASAN / SUPPORTS_UBSAN) ----------------
acar_supports_sanitizers()

# --- Sanitizer options --------------------------------------------------------
# cmake_dependent_option: if the capability probe above set SUPPORTS_* to OFF,
# the option is forced OFF regardless of the user's -D flag — prevents the
# configure-succeeds / build-fails scenario on unsupported platforms.
# When SUPPORTS_* is ON, these behave exactly like plain option() calls.
cmake_dependent_option(ACAR_ENABLE_SANITIZER_ADDRESS
    "Enable address sanitizer"            OFF "SUPPORTS_ASAN"  OFF)
cmake_dependent_option(ACAR_ENABLE_SANITIZER_LEAK
    "Enable leak sanitizer"               OFF "SUPPORTS_ASAN"  OFF)  # LSan shares ASan runtime
cmake_dependent_option(ACAR_ENABLE_SANITIZER_UNDEFINED
    "Enable undefined behavior sanitizer" OFF "SUPPORTS_UBSAN" OFF)
cmake_dependent_option(ACAR_ENABLE_SANITIZER_THREAD
    "Enable thread sanitizer"             OFF "SUPPORTS_ASAN"  OFF)  # same runtime probe
cmake_dependent_option(ACAR_ENABLE_SANITIZER_MEMORY
    "Enable memory sanitizer (Clang only)" OFF "SUPPORTS_MSAN" OFF)  # GCC does not support MSan
cmake_dependent_option(ACAR_ENABLE_SANITIZER_POINTER_COMPARE
    "Enable ASan pointer-compare/pointer-subtract (GCC only, requires ASan)" OFF "SUPPORTS_ASAN" OFF)

# --- Wire into sdk_options INTERFACE target -----------------------------------
# sdk_options is defined in CompileOptions.cmake (created before this include).
# All targets linking sdk::options automatically inherit these flags for C/CXX.
# CUDA code is explicitly excluded via generator expressions in Sanitizers.cmake.
enable_sanitizers(
    sdk_options
    ${ACAR_ENABLE_SANITIZER_ADDRESS}
    ${ACAR_ENABLE_SANITIZER_LEAK}
    ${ACAR_ENABLE_SANITIZER_UNDEFINED}
    ${ACAR_ENABLE_SANITIZER_THREAD}
    ${ACAR_ENABLE_SANITIZER_MEMORY}
    ${ACAR_ENABLE_SANITIZER_POINTER_COMPARE})
