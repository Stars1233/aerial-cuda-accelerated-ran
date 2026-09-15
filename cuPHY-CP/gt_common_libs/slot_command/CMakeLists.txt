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

# Lib: slot_command

find_package(gsl-lite REQUIRED)

add_library(slot_command INTERFACE)
message(VERBOSE "CMAKE_CURRENT_SOURCE_DIR is ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(slot_command INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
message(VERBOSE "PROJECT_SOURCE_DIR is ${PROJECT_SOURCE_DIR}/nvIPC/include/")
target_include_directories(slot_command INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/nvIPC/include/>")

# Link only header providers needed by slot_command.
target_link_libraries(slot_command INTERFACE cuphy_h oran_h gsl-lite::gsl-lite)

if (ENABLE_CELL_GROUP)
    set_target_properties(slot_command PROPERTIES INTERFACE_COMPILE_DEFINITIONS ENABLE_CELL_GROUP)
endif (ENABLE_CELL_GROUP)
