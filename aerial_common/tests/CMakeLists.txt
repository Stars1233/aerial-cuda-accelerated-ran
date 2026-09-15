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

add_executable(aerial_common_tests
    casts_test.cpp
    static_vector_test.cpp)
target_link_libraries(aerial_common_tests PRIVATE
    aerial_common
    sdk::options
    sdk::warnings)
target_link_system_libraries(aerial_common_tests PRIVATE GTest::gtest_main)

add_test(
    NAME aerial_common.unit_tests
    COMMAND aerial_common_tests --gtest_output=xml:aerial_common_tests.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(aerial_common.unit_tests PROPERTIES LABELS "parallel;cuphy-cp")
