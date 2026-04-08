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

# v3 Performance test executable (dependency-free: no Worker/CUDA/DPDK)
set(PERF_TEST_V3_TARGET test_instrumentation_v3_perf)
add_executable(${PERF_TEST_V3_TARGET} test_instrumentation_v3_perf.cpp)

# Lightweight includes only (PMU via pmu_reader.hpp, no heavy deps)
target_include_directories(${PERF_TEST_V3_TARGET} PRIVATE ${CUPHYDRIVER_HOME}/include)
target_include_directories(${PERF_TEST_V3_TARGET} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(${PERF_TEST_V3_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/cuPHY/nvlog/include)

# Link to cuphydriver (brings v3 implementation) and perf_metrics (for PercentileTracker)
target_link_libraries(${PERF_TEST_V3_TARGET} PRIVATE cuphydriver perf_metrics nvlog)

# Compile options
target_compile_options(${PERF_TEST_V3_TARGET} PRIVATE ${AERIAL_ARCH_TUNE_FLAGS} -O2)

# uint64_t to string conversion benchmark (fast_u64toa_safe vs fmt)
set(U64TOA_PERF_TARGET test_u64toa_perf)
add_executable(${U64TOA_PERF_TARGET} test_u64toa_perf.cpp)

# Include paths
target_include_directories(${U64TOA_PERF_TARGET} PRIVATE ${CUPHYDRIVER_HOME}/include)
target_include_directories(${U64TOA_PERF_TARGET} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(${U64TOA_PERF_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/cuPHY/nvlog/include)

# Link to perf_metrics (for fast_u64toa_safe) and fmt (for comparison)
target_link_libraries(${U64TOA_PERF_TARGET} PRIVATE perf_metrics fmt)

# Compile options - use O2 for fair comparison
target_compile_options(${U64TOA_PERF_TARGET} PRIVATE ${AERIAL_ARCH_TUNE_FLAGS} -O2)
