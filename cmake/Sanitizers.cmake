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

function(
    enable_sanitizers
    project_name
    enable_address
    enable_leak
    enable_undefined_behavior
    enable_thread
    enable_memory
    enable_pointer_compare)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(SANITIZERS "")

        if(${enable_address})
            list(APPEND SANITIZERS "address")
        endif()

        if(${enable_leak})
            list(APPEND SANITIZERS "leak")
        endif()

        if(${enable_undefined_behavior})
            list(APPEND SANITIZERS "undefined")
        endif()

        if(${enable_thread})
            if("address" IN_LIST SANITIZERS OR "leak" IN_LIST SANITIZERS)
                message(
                    WARNING "Thread sanitizer does not work with Address and Leak sanitizer enabled"
                )
            else()
                list(APPEND SANITIZERS "thread")
                set(ENABLE_TSAN TRUE)
            endif()
        endif()

        if(${enable_memory} AND CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
            message(
                WARNING
                    "Memory sanitizer requires all the code (including libc++) to be MSan-instrumented otherwise it reports false positives"
            )
            if("address" IN_LIST SANITIZERS
               OR "thread" IN_LIST SANITIZERS
               OR "leak" IN_LIST SANITIZERS)
                message(
                    WARNING
                        "Memory sanitizer does not work with Address, Thread or Leak sanitizer enabled"
                )
            else()
                list(APPEND SANITIZERS "memory")
            endif()
        endif()

        # pointer-compare and pointer-subtract detect UB from comparing/subtracting pointers
        # into different objects. GCC-only; requires ASan to be active (shared runtime).
        # Enabled via ACAR_ENABLE_SANITIZER_POINTER_COMPARE + detect_invalid_pointer_pairs=2.
        if(${enable_pointer_compare})
            if(NOT "address" IN_LIST SANITIZERS)
                message(
                    WARNING
                        "pointer-compare/pointer-subtract require AddressSanitizer; ignoring"
                )
            elseif(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                message(
                    WARNING
                        "pointer-compare/pointer-subtract are GCC-only; ignoring on non-GCC compiler"
                )
            else()
                list(APPEND SANITIZERS "pointer-compare" "pointer-subtract")
            endif()
        endif()
    elseif(MSVC)
        set(SANITIZERS "")
        if(${enable_address})
            list(APPEND SANITIZERS "address")
        endif()
        if(${enable_leak}
           OR ${enable_undefined_behavior}
           OR ${enable_thread}
           OR ${enable_memory})
            message(WARNING "MSVC only supports address sanitizer")
        endif()
    endif()

    list(JOIN SANITIZERS "," LIST_OF_SANITIZERS)

    if(LIST_OF_SANITIZERS)
        if(NOT MSVC)
            # Apply sanitizer flags only to C and C++ languages, not CUDA. CUDA/nvcc has issues
            # with sanitizer flags during device compilation.
            target_compile_options(
                ${project_name}
                INTERFACE $<$<COMPILE_LANGUAGE:C>:-fsanitize=${LIST_OF_SANITIZERS}>
                          $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${LIST_OF_SANITIZERS}>)
            # GCC 12+ emits -Wtsan for std::atomic_thread_fence under TSan. This is a known
            # TSan limitation (incomplete fence modeling), not a real race condition.
            if(ENABLE_TSAN AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                target_compile_options(
                    ${project_name}
                    INTERFACE $<$<COMPILE_LANGUAGE:C>:-Wno-tsan>
                              $<$<COMPILE_LANGUAGE:CXX>:-Wno-tsan>)
            endif()
            # Link options: OR over C and CXX so pure-C executables also receive -fsanitize=
            # on the link command (prevents "DSO missing" for libasan). COMPILE_LANGUAGE is a
            # no-op at link time. CUDA device-link steps (LINK_LANGUAGE:CUDA) are excluded.
            target_link_options(
                ${project_name} INTERFACE
                $<$<OR:$<LINK_LANGUAGE:C>,$<LINK_LANGUAGE:CXX>>:-fsanitize=${LIST_OF_SANITIZERS}>)
        else()
            string(FIND "$ENV{PATH}" "$ENV{VSINSTALLDIR}" index_of_vs_install_dir)
            if("${index_of_vs_install_dir}" STREQUAL "-1")
                message(
                    SEND_ERROR
                        "Using MSVC sanitizers requires setting the MSVC environment before building the project. Please manually open the MSVC command prompt and rebuild the project."
                )
            endif()
            target_compile_options(${project_name} INTERFACE /fsanitize=${LIST_OF_SANITIZERS}
                                                             /Zi /INCREMENTAL:NO)
            target_compile_definitions(${project_name} INTERFACE _DISABLE_VECTOR_ANNOTATION
                                                                 _DISABLE_STRING_ANNOTATION)
            target_link_options(${project_name} INTERFACE /INCREMENTAL:NO)
        endif()
    endif()

endfunction()
