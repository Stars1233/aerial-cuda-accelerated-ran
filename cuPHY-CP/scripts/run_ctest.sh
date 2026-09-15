#!/usr/bin/env bash

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

# run_ctest.sh — configure, build, and run CTest for a given CMake preset.
#
# Used in two contexts:
#   1. Local development: run directly to exercise cuPHY-CP tests with a
#      single command.  Example:
#        PRESET=cicd-test-arm bash cuPHY-CP/scripts/run_ctest.sh
#
#   2. Jenkins CI: executed inside the build container by the Jenkins job
#      triggered by cicd/job_scripts/ctest_testing.py.  Environment variables
#      PRESET, TEST_PARALLEL_JOBS, and TEST_TIMEOUT are injected by Jenkins.
#
# Prerequisites: cmake, ninja, and the project dependencies must be available
# in PATH (they are in the standard cuBB dev/CI container).

set -euo pipefail

PRESET="${PRESET:-cicd-test-arm}"
# nproc is Linux; getconf fallback covers macOS / cross-compile hosts
TEST_PARALLEL_JOBS="${TEST_PARALLEL_JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
TEST_TIMEOUT="${TEST_TIMEOUT:-300}"

echo "=== Configure: preset=${PRESET} ==="
cmake --preset "${PRESET}"

echo "=== Build ==="
cmake --build --preset "${PRESET}"

# testPreset filter.include.label="cuphy-cp" scopes execution to cuPHY-CP
# tests only.  Add LABELS "cuphy-cp" to any new test targets in cuPHY-CP.
echo "=== CTest: preset=${PRESET} label=cuphy-cp parallel=${TEST_PARALLEL_JOBS} timeout=${TEST_TIMEOUT}s ==="
CTEST_COMMAND=(ctest)
if [[ "${PRESET}" == tsan-arm-* ]]; then
    echo "Disabling ASLR for ThreadSanitizer tests"
    CTEST_COMMAND=(setarch "$(uname -m)" -R ctest)
fi

"${CTEST_COMMAND[@]}" --preset "${PRESET}" \
    --parallel "${TEST_PARALLEL_JOBS}" \
    --timeout "${TEST_TIMEOUT}"
