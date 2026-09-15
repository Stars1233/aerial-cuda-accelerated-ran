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

# run_pyaerial_ci.sh — configure, build, and run pyAerial CI targets.
#
# Used in two contexts:
#   1. Local development inside the Aerial devel container.
#   2. Jenkins CI via Jenkinsfile_cuphy_container_unit_test_pipeline (RUN_PYAERIAL=true).
#
# Environment variables:
#   PRESET              CMake configure preset (default: pyaerial-x86)
#   BUILD_ID            pyAerial package build id (default: 1)
#   TEST_VECTOR_DIR     cuPHY test vector root for unit tests

set -euo pipefail

# aerial_build_devel installs the GNU toolchain to /usr/local/gnu/bin.
# Prepend it so the cross-triplet compiler names used by CMake toolchain files
# (x86_64-linux-gnu-gcc, aarch64-linux-gnu-gcc) are found regardless of how
# the container was launched (interactive vs docker.image().inside() in Jenkins).
[ -d /usr/local/gnu/bin ] && export PATH="/usr/local/gnu/bin:${PATH}"

# Auto-select preset from host architecture; callers can still override via PRESET.
case "$(uname -m)" in
    aarch64) PRESET="${PRESET:-pyaerial-arm}" ;;
    *)       PRESET="${PRESET:-pyaerial-x86}" ;;
esac
BUILD_ID="${BUILD_ID:-${BUILD_NUMBER:-1}}"
export BUILD_ID

SCRIPT=$(readlink -f "$0")
PYAERIAL_ROOT=$(dirname "$(dirname "$SCRIPT")")
CUBB_SDK=$(realpath "${PYAERIAL_ROOT}/..")

cd "${CUBB_SDK}"

echo "=== Configure: preset=${PRESET} ==="
cmake --preset "${PRESET}"

echo "=== Build: _pycuphy (CTest fixture builds pyaerial_setup) ==="
cmake --build --preset "${PRESET}" --target _pycuphy

# No per-test timeout: the develop pyAerial pipeline runs uncapped, and the
# suite (full TV set) legitimately exceeds any short cap. The Jenkins job
# timeout is the backstop against a hung run.
echo "=== CTest: preset=${PRESET} ==="
ctest --preset "${PRESET}"

echo "=== pyAerial CI finished successfully ==="
