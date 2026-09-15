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

# from here:
#
# https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Available.md

function(
    set_project_warnings
    project_name
    warnings_as_errors
    msvc_warnings
    clang_warnings
    gcc_warnings
    cuda_warnings)
    if("${msvc_warnings}" STREQUAL "")
        set(msvc_warnings
            /W4 # Baseline reasonable warnings
            /w14242 # 'identifier': conversion from 'type1' to 'type2', possible loss of data
            /w14254 # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits', possible
                    # loss of data
            /w14263 # 'function': member function does not override any base class virtual member
                    # function
            /w14265 # 'classname': class has virtual functions, but destructor is not virtual
                    # instances of this class may not be destructed correctly
            /w14287 # 'operator': unsigned/negative constant mismatch
            /we4289 # nonstandard extension used: 'variable': loop control variable declared in the
                    # for-loop is used outside the for-loop scope
            /w14296 # 'operator': expression is always 'boolean_value'
            /w14311 # 'variable': pointer truncation from 'type1' to 'type2'
            /w14545 # expression before comma evaluates to a function which is missing an argument
                    # list
            /w14546 # function call before comma missing argument list
            /w14547 # 'operator': operator before comma has no effect; expected operator with
                    # side-effect
            /w14549 # 'operator': operator before comma has no effect; did you intend 'operator'?
            /w14555 # expression has no effect; expected expression with side- effect
            /w14619 # pragma warning: there is no warning number 'number'
            /w14640 # Enable warning on thread un-safe static member initialization
            /w14826 # Conversion from 'type1' to 'type2' is sign-extended. This may cause unexpected
                    # runtime behavior.
            /w14905 # wide string literal cast to 'LPSTR'
            /w14906 # string literal cast to 'LPWSTR'
            /w14928 # illegal copy-initialization; more than one user-defined conversion has been
                    # implicitly applied
            /permissive- # standards conformance mode for MSVC compiler.
        )
    endif()

    if("${clang_warnings}" STREQUAL "")
        set(clang_warnings
            -Wall
            -Wextra # reasonable and standard
            -Wshadow # warn the user if a variable declaration shadows one from a parent context
            -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual
                               # destructor. This helps
            # catch hard to track down memory errors
            -Wold-style-cast # warn for c-style casts
            -Wcast-align # warn for potential performance problem casts
            -Wunused # warn on anything being unused
            -Woverloaded-virtual # warn if you overload (not override) a virtual function
            -Wpedantic # warn if non-standard C++ is used
            -Wconversion # warn on type conversions that may lose data
            -Wsign-conversion # warn on sign conversions
            -Wnull-dereference # warn if a null dereference is detected
            -Wdouble-promotion # warn if float is implicit promoted to double
            -Wformat=2 # warn on security issues around functions that format output (ie printf)
            -Wimplicit-fallthrough # warn on statements that fallthrough without an explicit
                                   # annotation
        )
    endif()

    if("${gcc_warnings}" STREQUAL "")
        set(gcc_warnings
            ${clang_warnings} -Wmisleading-indentation # warn if indentation implies blocks where
                                                       # blocks do not exist
        )
        # GCC-only warnings (not supported by Clang)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            list(
                APPEND
                gcc_warnings
                -Wduplicated-cond # warn if if / else chain has duplicated conditions
                -Wduplicated-branches # warn if if / else branches have duplicated code
                -Wlogical-op # warn about logical operations being used where bitwise were probably
                             # wanted
                -Wuseless-cast # warn if you perform a cast to the same type
            )
        endif()
        list(APPEND gcc_warnings -Wsuggest-override # warn if an overridden member function is not
                                                    # marked 'override' or 'final'
        )
    endif()

    if("${cuda_warnings}" STREQUAL "")
        # These are host-compiler flags (GCC/Clang). nvcc forwards unrecognized -W flags
        # to the host compiler implicitly, consistent with how this project passes
        # -Wconversion/-Wsign-conversion to CUDA in the root CMakeLists.txt.
        set(cuda_warnings -Wall -Wextra -Wunused -Wconversion -Wshadow)
    endif()

    if(warnings_as_errors)
        message(TRACE "Warnings are treated as errors")
        list(APPEND clang_warnings -Werror)
        list(APPEND gcc_warnings -Werror)
        list(APPEND msvc_warnings /WX)
        # Treat CUDA warnings as errors for both device and host code Use explicit --Werror
        # all-warnings to avoid ambiguity with -Xcompiler
        list(APPEND cuda_warnings --Werror all-warnings "SHELL:-Xcompiler -Werror")
    endif()

    if(MSVC)
        set(PROJECT_WARNINGS_CXX ${msvc_warnings})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(PROJECT_WARNINGS_CXX ${clang_warnings})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(PROJECT_WARNINGS_CXX ${gcc_warnings})
    else()
        message(
            AUTHOR_WARNING "No compiler warnings set for CXX compiler: '${CMAKE_CXX_COMPILER_ID}'")
    endif()

    # C warnings: start from CXX set but remove C++-only flags that GCC/Clang reject
    # for C compilation (would become fatal errors under -Werror).
    set(PROJECT_WARNINGS_C "${PROJECT_WARNINGS_CXX}")
    list(REMOVE_ITEM PROJECT_WARNINGS_C
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Wold-style-cast
        -Wsuggest-override
        -Wuseless-cast)

    set(PROJECT_WARNINGS_CUDA "${cuda_warnings}")

    target_compile_options(
        ${project_name}
        INTERFACE # C++ warnings
                  $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS_CXX}>
                  # C warnings
                  $<$<COMPILE_LANGUAGE:C>:${PROJECT_WARNINGS_C}>
                  # Cuda warnings
                  $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_WARNINGS_CUDA}>)
endfunction()
