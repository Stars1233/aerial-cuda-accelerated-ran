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

find_package(gsl-lite REQUIRED)

add_library(aerial_common INTERFACE)
add_library(aerial::common ALIAS aerial_common)
target_include_directories(aerial_common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_system_libraries(aerial_common INTERFACE gsl-lite::gsl-lite)
target_compile_definitions(aerial_common INTERFACE gsl_CONFIG_CONTRACT_VIOLATION_THROWS)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
