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

# Performance metrics unit test

add_executable(perf_metrics_accumulator_unit_test 
    perf_metrics_accumulator_unit_test.cpp
)

# Set C++ standard
set_target_properties(perf_metrics_accumulator_unit_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Link with the perf_metrics library
target_link_libraries(perf_metrics_accumulator_unit_test PRIVATE perf_metrics)

# Ensure nvlog headers are available for the test
if(TARGET nvlog)
    target_link_libraries(perf_metrics_accumulator_unit_test PRIVATE nvlog)
endif()

# No cuphydriver dependency needed - using self-contained timing

# Performance metrics benchmark

add_executable(perf_metrics_accumulator_benchmark 
    perf_metrics_accumulator_benchmark.cpp
)

# Set C++ standard
set_target_properties(perf_metrics_accumulator_benchmark PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Link with the perf_metrics library
target_link_libraries(perf_metrics_accumulator_benchmark PRIVATE perf_metrics)

# Ensure nvlog headers are available for the benchmark
if(TARGET nvlog)
    target_link_libraries(perf_metrics_accumulator_benchmark PRIVATE nvlog)
endif()

# Percentile tracker unit test

add_executable(percentile_tracker_unit_test 
    percentile_tracker_unit_test.cpp
)

# Set C++ standard
set_target_properties(percentile_tracker_unit_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Link with the perf_metrics library
target_link_libraries(percentile_tracker_unit_test PRIVATE perf_metrics)

# Ensure nvlog headers are available for the test
if(TARGET nvlog)
    target_link_libraries(percentile_tracker_unit_test PRIVATE nvlog)
endif()
