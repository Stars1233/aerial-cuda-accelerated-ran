#!/bin/bash

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

# Run the package-owned tests with an environment where the CMake-built wheel
# is already installed. This script deliberately performs no build or install.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMPONENT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
SDK_ROOT=$(cd "${COMPONENT_ROOT}/../.." && pwd)
TEST_ROOT="${COMPONENT_ROOT}/python/tests"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    case "$(uname -m)" in
        x86_64|amd64) DEFAULT_PRESET="gpu3gppchan-x86" ;;
        aarch64|arm64) DEFAULT_PRESET="gpu3gppchan-arm" ;;
        *)
            echo "ERROR: unsupported architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
    GPU3GPPCHAN_CMAKE_PRESET="${GPU3GPPCHAN_CMAKE_PRESET:-${DEFAULT_PRESET}}"
    (
        cd "${SDK_ROOT}"
        cmake --preset "${GPU3GPPCHAN_CMAKE_PRESET}"
        cmake --build --preset "${GPU3GPPCHAN_CMAKE_PRESET}" \
            --target gpu3gppchan_python_setup
    )
    PYTHON_BIN="${COMPONENT_ROOT}/python/.venv/bin/python"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: gpu3gppchan Python environment not found at ${PYTHON_BIN}" >&2
    exit 1
fi

"${PYTHON_BIN}" -m pytest -sv \
    "${TEST_ROOT}/test_chmod_fading_chan.py" \
    "${TEST_ROOT}/test_chmod_ofdm.py" \
    "${TEST_ROOT}/test_chmod_sls.py" \
    "${TEST_ROOT}/test_chmod_isac.py" \
    "${TEST_ROOT}/test_optional_dependency_errors.py" \
    "${TEST_ROOT}/test_analysis_channel_stats.py" \
    "${TEST_ROOT}/test_wheel_smoke.py"
