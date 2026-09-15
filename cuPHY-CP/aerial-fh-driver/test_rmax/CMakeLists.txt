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

if(ENABLE_RIVERMAX)
    add_executable(ut_rmax aerial_fh_rmax_tests.cpp)
    set_property(TARGET ut_rmax PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_include_directories(ut_rmax PRIVATE ${CUDA_INCLUDE_DIRS} /usr/local/include/mellanox/)
    target_link_libraries(ut_rmax gtest_main ${AERIAL_FH_MOCKED_TARGET} nvlog rivermax)

    install(TARGETS ut_rmax RUNTIME DESTINATION bin)
    include(GoogleTest)

    if(GTEST_POPULATE_CMAKE_TESTS)
        gtest_discover_tests(ut_rmax PROPERTIES LABELS "cuphy-cp")
    endif(GTEST_POPULATE_CMAKE_TESTS)

    add_custom_target(check_rmax COMMAND CTEST_OUTPUT_ON_FAILURE=1 ${CMAKE_CTEST_COMMAND})
    add_dependencies(check_rmax ut_rmax)

else(ENABLE_RIVERMAX)
    message("Rivermax not enable, ut_rmax not built")
endif(ENABLE_RIVERMAX)
