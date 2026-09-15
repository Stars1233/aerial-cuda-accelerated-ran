#!/bin/bash

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

# Usage:
#   run_unit_tests.sh            # TRT engine prep + pytest (direct invocation)
#   run_unit_tests.sh trt        # TRT engine prep only (cmake step 1)
#   run_unit_tests.sh pytest     # pytest only (cmake step 2)

set -euo pipefail

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

cd "${PROJECT_ROOT}"

# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"

STEP="${1:-all}"

run_trt_engines() {
    export CUDA_MODULE_LOADING=LAZY

    mkdir -p "${HOME}/models"

    trtexec --onnx="${PROJECT_ROOT}/models/llrnet.onnx" \
        --saveEngine="${HOME}/models/llrnet.trt" \
        --skipInference \
        --minShapes=input:1x2 \
        --optShapes=input:12345x2 \
        --maxShapes=input:42588x2 \
        --inputIOFormats=fp32:chw \
        --outputIOFormats=fp32:chw

    trtexec --onnx="${PROJECT_ROOT}/models/neural_rx.onnx" \
        --saveEngine="${HOME}/models/neural_rx.trt" \
        --skipInference \
        --shapes=rx_slot_real:1x3276x12x4,rx_slot_imag:1x3276x12x4,h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4
}

run_pytest() {
    if [[ -z "${TEST_VECTOR_DIR:-}" ]]; then
        echo "Test vector directory is not set - defaulting to /mnt/cicd_tvs/develop/GPU_test_input/."
        echo "Unit tests will be skipped if test vectors are not found."
        echo "Set test vector directory as follows:"
        echo "export TEST_VECTOR_DIR=<test vector directory>"
        echo ""
    fi

    export CUDA_MODULE_LOADING=LAZY

    echo "pyAerial: Run Python unit tests..."
    pushd tests > /dev/null
    "${PYTHON}" -m pytest --color=yes -o cache_dir="${HOME}/.pytest_cache" -sv
    popd > /dev/null

    rm -f "${HOME}/models/llrnet.trt" "${HOME}/models/neural_rx.trt"
    rmdir "${HOME}/models" 2>/dev/null || true
}

case "${STEP}" in
    trt)
        run_trt_engines
        ;;
    pytest)
        run_pytest
        ;;
    all)
        run_trt_engines
        run_pytest
        ;;
    *)
        echo "Unknown step: ${STEP} (expected trt, pytest, or all)" >&2
        exit 1
        ;;
esac

echo "${SCRIPT} finished with success."
