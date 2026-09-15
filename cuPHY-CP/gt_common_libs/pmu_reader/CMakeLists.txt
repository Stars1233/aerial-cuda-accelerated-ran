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
    src/pmu_reader.cpp
)

set(HEADERS
    include/pmu_reader.hpp
)

add_library(pmu_reader STATIC ${SOURCES} ${HEADERS})
add_library(pmu_reader::pmu_reader ALIAS pmu_reader)
set_target_properties(pmu_reader PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(pmu_reader
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

target_link_system_libraries(pmu_reader PRIVATE nvlog)
target_link_libraries(pmu_reader PRIVATE sdk::options sdk::warnings)

# NVLOGI_FMT expands VectorCAST _Pragma() at the call site in this .cpp, so -Wunknown-pragmas still
# fires even with -isystem on nvlog; disable until NVLOG removes those pragmas from the macro path.
target_compile_options(pmu_reader PRIVATE -Wno-unknown-pragmas)
