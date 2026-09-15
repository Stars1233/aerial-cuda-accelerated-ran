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

########################################################################
# test_multiCellUeSelection
add_executable(test_multiCellUeSelection
    test_multiCellUeSelection.cpp
    dummy.cu)

target_include_directories(test_multiCellUeSelection PRIVATE
    ${cumac_SOURCE_DIR}/src
    ${cumac_SOURCE_DIR}/src/4T4R
    ${cumac_SOURCE_DIR}/lib
    ${HDF5_CXX_INCLUDE_DIRS})

target_link_libraries(test_multiCellUeSelection PRIVATE
    cumac
    GTest::gtest
    CUDA::cudart
    CUDA::cuda_driver)

target_compile_options(test_multiCellUeSelection PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_test(NAME test_multiCellUeSelection COMMAND test_multiCellUeSelection)
