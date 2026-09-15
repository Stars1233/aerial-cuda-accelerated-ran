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

add_executable(testDescrambling testDescrambling.cpp dummy.cu)

add_dependencies(testDescrambling CRCLUTGEN)
target_link_libraries(testDescrambling cuphy)
target_link_libraries(testDescrambling gtest gtest_main)
add_test(NAME testDescrambling COMMAND testDescrambling)

add_custom_target(testDescramblingFull
    COMMAND testDescrambling --gtest_also_run_disabled_tests
    DEPENDS testDescrambling)

option(CUPHY_ENABLE_FULL_TESTS "Enable exhaustive descrambling ctest tier" OFF)
if(CUPHY_ENABLE_FULL_TESTS)
    add_test(NAME testDescramblingFull COMMAND testDescrambling --gtest_also_run_disabled_tests)
    set_tests_properties(testDescramblingFull PROPERTIES LABELS "full")
endif()
