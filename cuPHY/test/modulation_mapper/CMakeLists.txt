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

########################################################################

add_executable(testModulationMapper "${PROJECT_SOURCE_DIR}/src/cuphy/modulation_mapper/testModulationMapper.cpp" dummy.cu)
target_link_libraries(testModulationMapper cuphy_hdf5 cuphy)
target_link_libraries(testModulationMapper gtest gtest_main)
add_test(NAME testModulationMapper COMMAND testModulationMapper WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/modulation_mapper)

add_executable(test_modulation_mapper test_modulation_mapper.cpp dummy.cu)
target_include_directories(test_modulation_mapper PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_modulation_mapper cuphy gtest)
target_compile_options(test_modulation_mapper PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_modulation_mapper COMMAND test_modulation_mapper)

#                        --keep
