# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Lib: perf_metrics

# Source files
set(SOURCES
    src/perf_metrics_accumulator.cpp
    src/perf_metrics_utils.cpp
    src/percentile_tracker.cpp
)

# Header files
set(HEADERS
    include/perf_metrics/perf_metrics_accumulator.hpp
    include/perf_metrics/perf_metrics_logging.hpp
    include/perf_metrics/perf_metrics_utils.hpp
    include/perf_metrics/percentile_tracker.hpp
)

# Create the library (now with implementation)
add_library(perf_metrics STATIC ${SOURCES} ${HEADERS})

# Enable Position Independent Code for linking with shared libraries
set_target_properties(perf_metrics PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set include directories
target_include_directories(perf_metrics 
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

# Link with nvlog for logging functionality and event codes
if(TARGET nvlog)
    target_link_libraries(perf_metrics PRIVATE nvlog)
    message(STATUS "Linking perf_metrics with nvlog target")
else()
    message(WARNING "nvlog target not available")
endif()

# Add test subdirectory
if (ENABLE_TESTS)
    add_subdirectory(test)
endif()

message(VERBOSE "CMAKE_CURRENT_SOURCE_DIR is ${CMAKE_CURRENT_SOURCE_DIR}")