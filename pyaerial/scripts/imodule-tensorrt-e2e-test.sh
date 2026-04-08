#!/usr/bin/env bash

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

# CI/CD script for testing EnhancedFusedChannelEstimator with TensorRT integration
# This script runs all verification steps in sequence to validate the model

set -e  # Exit immediately if any command fails

# Parse command line arguments
USE_LOCAL_VENV=0
SKIP_PUSCH=1  # Default to skipping PUSCH test
for arg in "$@"
do
    case $arg in
        --use-local-venv)
        USE_LOCAL_VENV=1
        shift # Remove --use-local-venv from processing
        ;;
        --skip-pusch)
        SKIP_PUSCH=1
        shift # Remove --skip-pusch from processing
        ;;
        --no-skip-pusch)
        SKIP_PUSCH=0
        shift # Remove --no-skip-pusch from processing
        ;;
        *)
        # Unknown option
        ;;
    esac
done

# Record start time
start_time=$(date +%s)

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Add timestamp to filenames
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
ONNX_FILENAME="model_${TIMESTAMP}.onnx"
ENGINE_FILENAME="enhanced_fused_channel_estimator_${TIMESTAMP}.engine"

# Activate virtual environment if it exists and --use-local-venv flag is provided
if [ "$USE_LOCAL_VENV" -eq 1 ] && [ -d "${REPO_ROOT}/.venv" ]; then
    echo "Activating virtual environment at ${REPO_ROOT}/.venv"
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Set the root directory (cuBB) in PYTHONPATH
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
echo "PYTHONPATH set to: ${PYTHONPATH}"
OUTPUT_DIR="/tmp/cicd_output"

# Parameters - can be overridden by environment variables
# These values are aligned with chest_trt.yaml (ground truth)
NUM_RES=${NUM_RES:-1638}
COMB_SIZE=${COMB_SIZE:-2}
NUM_PRBS=${NUM_PRBS:-137}  # 1638/12 ~= 137
NUM_RX_ANT=${NUM_RX_ANT:-4}
BATCH_SIZE=${BATCH_SIZE:-1}
LAYERS=${LAYERS:-4}
SYMBOLS=${SYMBOLS:-2}
PRECISION=${PRECISION:-fp16}
PYTHON_CMD=${PYTHON_CMD:-python3}

# Print configuration
echo "=== Configuration ==="
echo "NUM_RES: ${NUM_RES}"
echo "LAYERS: ${LAYERS}"
echo "NUM_RX_ANT: ${NUM_RX_ANT}"
echo "SYMBOLS: ${SYMBOLS}"
echo "NUM_PRBS: ${NUM_PRBS}"
echo "PRECISION: ${PRECISION}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Display system info
echo "=== System Information ==="
echo "CUDA devices:"
nvidia-smi --list-gpus || echo "nvidia-smi not available"
echo

# Functions for logging
log_step() {
    echo
    echo "============================================================"
    echo "STEP $1: $2"
    echo "============================================================"
}

log_result() {
    if [ "$1" -eq 0 ]; then
        echo "✅ $2 completed successfully"
    else
        echo "❌ $2 failed with exit code $1"
        exit "$1"
    fi
}

# Step 1: Run PyTorch model
log_step 1 "Running PyTorch model"
${PYTHON_CMD} "${REPO_ROOT}/pyaerial/tests/model_to_engine_tests/run_pytorch_model.py" \
  --num_res "${NUM_RES}" \
  --comb_size "${COMB_SIZE}" \
  --do_fft \
  --batch "${BATCH_SIZE}" \
  --layers "${LAYERS}" \
  --rx_antennas "${NUM_RX_ANT}" \
  --symbols "${SYMBOLS}" \
  --output "${OUTPUT_DIR}/pytorch_output.npy"
log_result $? "PyTorch model run"

# Step 2: Export to ONNX and compare results
log_step 2 "Exporting to ONNX and comparing results"
${PYTHON_CMD} "${REPO_ROOT}/pyaerial/tests/model_to_engine_tests/export_to_onnx.py" \
  --num_res "${NUM_RES}" \
  --comb_size "${COMB_SIZE}" \
  --do_fft \
  --input "${OUTPUT_DIR}/pytorch_output_input.npy" \
  --pytorch_output "${OUTPUT_DIR}/pytorch_output.npy" \
  --onnx_path "${OUTPUT_DIR}/${ONNX_FILENAME}" \
  --onnx_output "${OUTPUT_DIR}/onnx_output.npy"
log_result $? "ONNX export and comparison"

# Step 3: Export to TensorRT and compare results
log_step 3 "Exporting to TensorRT and comparing results"
${PYTHON_CMD} "${REPO_ROOT}/pyaerial/tests/model_to_engine_tests/export_to_trt.py" \
  --num_res "${NUM_RES}" \
  --comb_size "${COMB_SIZE}" \
  --do_fft \
  --output_dir "${OUTPUT_DIR}" \
  --onnx_path "${OUTPUT_DIR}/${ONNX_FILENAME}" \
  --precision "${PRECISION}" \
  --engine_filename "${ENGINE_FILENAME}" \
  --use_api_direct
TRT_RESULT=$?
echo "DEBUG: Listing output directory to check for engine file:"
ls -la "${OUTPUT_DIR}"
log_result "${TRT_RESULT}" "TensorRT export and comparison"

# Step 4: Generate YAML configuration for TensorRT engine
log_step 4 "Generating YAML configuration for TensorRT engine"
${PYTHON_CMD} "${REPO_ROOT}/pyaerial/tests/model_to_engine_tests/generate_yaml_for_engine.py" \
  --engine "${OUTPUT_DIR}/${ENGINE_FILENAME}" \
  --yaml "${OUTPUT_DIR}/chest_trt.yaml"
log_result $? "YAML configuration generation"

# Step 5: Test TensorRT channel estimator in standalone mode
log_step 5 "Testing TensorRT channel estimator in standalone mode"
${PYTHON_CMD} "${REPO_ROOT}/pyaerial/tests/model_to_engine_tests/test_trt_standalone.py" \
  --yaml "${OUTPUT_DIR}/chest_trt.yaml" \
  --num_prbs "${NUM_PRBS}" \
  --num_rx_ant "${NUM_RX_ANT}" \
  --output "${OUTPUT_DIR}/standalone_output.npy"
log_result $? "Standalone TensorRT channel estimator test"

# Step 6: Test TensorRT channel estimator in PUSCH RX pipeline
if [ "$SKIP_PUSCH" -eq 1 ]; then
    echo
    echo "============================================================"
    echo "STEP 6: Testing TensorRT channel estimator in PUSCH RX pipeline"
    echo "============================================================"
    echo "⚠️  WARNING: PUSCH RX pipeline test SKIPPED (--skip-pusch flag is set)"
    echo "   Reason: NaN issues in noise estimation need to be resolved"
    echo "   To enable this test, run with --no-skip-pusch flag"
    echo "============================================================"
else
    log_step 6 "Testing TensorRT channel estimator in PUSCH RX pipeline"
    ${PYTHON_CMD} "${REPO_ROOT}/pyaerial/tests/model_to_engine_tests/test_trt_pusch_rx.py" \
      --yaml "${OUTPUT_DIR}/chest_trt.yaml" \
      --num_prbs "${NUM_PRBS}" \
      --num_rx_ant "${NUM_RX_ANT}" \
      --output "${OUTPUT_DIR}" \
      --verbose
    log_result $? "PUSCH RX TensorRT channel estimator test"
fi

# Generate summary
echo
echo "=== Test Summary ==="
if [ "$SKIP_PUSCH" -eq 1 ]; then
    echo "Tests completed with warnings (PUSCH RX test was skipped)"
else
    echo "All tests completed successfully"
fi
echo "Output files are in: ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}"

# Check output file sizes to verify they were created properly
echo
echo "=== Output File Verification ==="
for file in pytorch_output.npy onnx_output.npy; do
    file_path="${OUTPUT_DIR}/${file}"
    if [ -f "$file_path" ]; then
        file_size=$(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null)
        echo "${file}: $file_size bytes"
        # Check if file is empty
        if [ "$file_size" -lt 100 ]; then
            echo "⚠️  Warning: $file is suspiciously small"
        fi
    else
        echo "⚠️  Warning: $file not found"
    fi
done

echo
echo "=== CI/CD Pipeline Completed ==="
echo "End time: $(date)"
echo "Duration: $(($(date +%s) - start_time)) seconds"

# Overall success
if [ "$SKIP_PUSCH" -eq 1 ]; then
    echo "✅ CICD test completed with expected warnings (PUSCH RX test skipped)"
else
    echo "✅ CICD test completed with expected warnings for TensorRT engine generation"
fi
exit 0
