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

# v3 performance test (dependency-free: no Worker/CUDA/DPDK). Lives with task_instrumentation.
set(PERF_TEST_V3_TARGET test_instrumentation_v3_perf)
add_executable(${PERF_TEST_V3_TARGET} test_instrumentation_v3_perf.cpp)

# uint64_t → string microbenchmark (perf_metrics::fmt_u64toa_safe vs fmt / sprintf).
# nvlog: perf_metrics public headers (e.g. percentile_tracker.hpp) include nvlog.hpp; perf_metrics
# links nvlog PRIVATE so include paths are not propagated — link nvlog here too.
set(U64TOA_PERF_TARGET test_u64toa_perf)
add_executable(${U64TOA_PERF_TARGET} test_u64toa_perf.cpp)
target_link_libraries(${U64TOA_PERF_TARGET} PRIVATE perf_metrics fmt sdk::options sdk::warnings)
target_link_system_libraries(${U64TOA_PERF_TARGET} PRIVATE nvlog)
target_compile_options(${U64TOA_PERF_TARGET} PRIVATE ${AERIAL_ARCH_TUNE_FLAGS} -O2 -Wno-unknown-pragmas)

# pmu_reader: PMUDeltaSummarizer in benchmark (headers not re-exported by task_instrumentation LINK_ONLY pmu).
# nvlog: nvlog_c_init / fmtlog. perf_metrics comes in via task_instrumentation (PUBLIC).
target_link_libraries(${PERF_TEST_V3_TARGET} PRIVATE
    task_instrumentation
    pmu_reader
)

target_link_system_libraries(${PERF_TEST_V3_TARGET} PRIVATE nvlog)

target_link_libraries(${PERF_TEST_V3_TARGET} PRIVATE sdk::options sdk::warnings)

# NVLOGI_FMT expands VectorCAST _Pragma() at the call site in benchmark TUs — same suppression as library targets.
target_compile_options(${PERF_TEST_V3_TARGET} PRIVATE ${AERIAL_ARCH_TUNE_FLAGS} -O2 -Wno-unknown-pragmas)
