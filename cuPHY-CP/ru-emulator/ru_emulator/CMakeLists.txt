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

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/helper.cmake)

add_library(ru_emulator_common STATIC ${RU_EMULATOR_COMMON_SOURCES})
link_target(ru_emulator_common)

# Mode-specific variants keep pre-refactor semantics where compile definitions
# apply to all RU emulator translation units for each executable.
add_library(ru_emulator_standalone_common STATIC ${RU_EMULATOR_COMMON_SOURCES})
target_compile_definitions(ru_emulator_standalone_common PUBLIC STANDALONE=1)
link_target(ru_emulator_standalone_common)

add_library(ru_emulator_debug_common STATIC ${RU_EMULATOR_COMMON_SOURCES})
target_compile_definitions(ru_emulator_debug_common PUBLIC RU_EM_UL_TIMERS_ENABLE=1)
link_target(ru_emulator_debug_common)

set(TARGET ru_emulator)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE ru_emulator_common)
link_target(${TARGET})

# ----------------------------------------------------------------------
# Functional test
# ----------------------------------------------------------------------
set(STANDALONE ru_standalone)
add_executable(${STANDALONE} main.cpp)
target_link_libraries(${STANDALONE} PRIVATE ru_emulator_standalone_common)
link_target(${STANDALONE})

set(DEBUG ru_debug)
add_executable(${DEBUG} main.cpp)
target_link_libraries(${DEBUG} PRIVATE ru_emulator_debug_common)
link_target(${DEBUG})
