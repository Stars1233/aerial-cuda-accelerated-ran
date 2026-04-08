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
# test_ss
add_executable(test_ss test_ss.cpp dummy.cu)
target_include_directories(test_ss PRIVATE 
    ${PROJECT_SOURCE_DIR}/examples/common
    ${PROJECT_SOURCE_DIR}/src/cuphy/ss)
target_link_libraries(test_ss cuphy_channels cuphy gtest)
target_compile_options(test_ss PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ss COMMAND test_ss)


