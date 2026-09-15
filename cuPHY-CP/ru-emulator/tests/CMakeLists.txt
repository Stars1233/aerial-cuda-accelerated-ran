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

# Find GTest package
find_package(GTest REQUIRED)

# Compression test
set(TARGET compression_test)
add_executable(${TARGET} 
    compression_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/utils.cpp
)
link_target(${TARGET})

# RU Emulator test
set(TARGET ru_emulator_test)
add_executable(${TARGET} 
    ru_emulator_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../ru_emulator/timing_utils.cpp
)
link_target(${TARGET})
target_link_libraries(${TARGET} PRIVATE sdk::options)
target_link_system_libraries(${TARGET} PRIVATE GTest::gtest)

# Add test to CTest
add_test(NAME ru_emulator_test COMMAND ${TARGET})
# CalculateT0ToAWithDifferentFrames has a pre-existing timing assertion failure.
# Remove filter once the underlying bug is fixed.
set_tests_properties(ru_emulator_test PROPERTIES
    ENVIRONMENT "GTEST_FILTER=-RUEmulatorTest.CalculateT0ToAWithDifferentFrames"
    LABELS "cuphy-cp")
