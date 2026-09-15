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
# test_roundRobinScheduler
#
# Drives the GPU multiCellRRScheduler (round-robin PRG allocation) kernels,
# which live in the instrumented `cumac` library. The test only links cumac;
# it does not compile the scheduler sources directly.
add_executable(test_roundRobinScheduler
    test_roundRobinScheduler.cpp
    dummy.cu
)
target_include_directories(test_roundRobinScheduler PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/4T4R
    ${HDF5_CXX_INCLUDE_DIRS}
)
target_link_libraries(test_roundRobinScheduler PRIVATE
    cumac
    ${HDF5_CXX_LIBRARIES}
    GTest::gtest
    CUDA::cudart
)
target_compile_options(test_roundRobinScheduler PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
add_test(NAME test_roundRobinScheduler COMMAND test_roundRobinScheduler)
