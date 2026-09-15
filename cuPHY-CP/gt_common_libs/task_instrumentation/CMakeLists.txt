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

set(SOURCES
    src/task_instrumentation_v3.cpp
)

set(HEADERS
    include/task_instrumentation/task_instrumentation_v3.hpp
    include/task_instrumentation/task_instrumentation_nested.hpp
)

add_library(task_instrumentation STATIC ${SOURCES} ${HEADERS})
add_library(task_instrumentation::task_instrumentation ALIAS task_instrumentation)
set_target_properties(task_instrumentation PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(task_instrumentation
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(task_instrumentation
    PUBLIC perf_metrics
    PRIVATE pmu_reader
)

# nvlog: link + INTERFACE includes as SYSTEM (see cmake/SystemLink.cmake — suppresses third-party warnings).
target_link_system_libraries(task_instrumentation PRIVATE nvlog)

# Propagate link to pmu_reader for static archive consumers without exposing pmu_reader headers
target_link_libraries(task_instrumentation INTERFACE $<LINK_ONLY:pmu_reader>)
target_link_libraries(task_instrumentation PRIVATE sdk::options sdk::warnings)

# NVLOGI_FMT expands VectorCAST _Pragma() at the call site in this .cpp, so -Wunknown-pragmas still
# fires even with -isystem on nvlog; disable until NVLOG removes those pragmas from the macro path.
target_compile_options(task_instrumentation PRIVATE ${AERIAL_ARCH_TUNE_FLAGS} -Wno-unknown-pragmas)
