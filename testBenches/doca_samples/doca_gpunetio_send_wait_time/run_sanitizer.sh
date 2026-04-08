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

# Run doca_gpunetio_send_wait_time under compute-sanitizer (memcheck, racecheck, synccheck, initcheck).
#
# Usage: ./run_sanitizer.sh <tool> -n <NIC_PCIE> -g <GPU_PCIE> [-t <TIME_NS>]
#   tool: memcheck | racecheck | synccheck | initcheck
#
# Example (same as run.sh):
#   ./run_sanitizer.sh memcheck   -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000
#   ./run_sanitizer.sh racecheck  -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000
#   ./run_sanitizer.sh synccheck  -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000
#   ./run_sanitizer.sh initcheck  -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000

set -e

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

if [ -f "/tmp/doca_builds/gpunetio_send_wait_time/build/doca_gpunetio_send_wait_time" ]; then
    BUILD_DIR="/tmp/doca_builds/gpunetio_send_wait_time/build"
else
    BUILD_DIR="${SCRIPT_DIR}/build"
fi
EXECUTABLE="${BUILD_DIR}/doca_gpunetio_send_wait_time"

ARCH=$(uname -m)
case "${ARCH}" in
    x86_64)  LIB_ARCH="x86_64-linux-gnu" ;;
    aarch64) LIB_ARCH="aarch64-linux-gnu" ;;
    *)       echo "ERROR: Unsupported architecture: ${ARCH}"; exit 1 ;;
esac

DOCA_LIB_PATH="/opt/mellanox/doca/lib/${LIB_ARCH}"
DPDK_LIB_PATH="/opt/mellanox/dpdk/lib/${LIB_ARCH}"

if [ $# -lt 5 ]; then
    echo "Usage: $0 <tool> -n <NIC_PCIE> -g <GPU_PCIE> [-t <TIME_NS>]"
    echo "  tool: memcheck | racecheck | synccheck | initcheck"
    echo "Example: $0 memcheck -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000"
    exit 1
fi

SANITIZER_TOOL="$1"
shift

case "${SANITIZER_TOOL}" in
    memcheck|racecheck|synccheck|initcheck) ;;
    *)
        echo "ERROR: Unknown tool '${SANITIZER_TOOL}'. Use memcheck, racecheck, synccheck, or initcheck."
        exit 1
        ;;
esac

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: Executable not found: ${EXECUTABLE}. Run build.sh first."
    exit 1
fi

# Use full path to compute-sanitizer so sudo finds it (sudo often resets PATH)
COMPUTE_SANITIZER=$(command -v compute-sanitizer 2>/dev/null || true)
if [ -z "${COMPUTE_SANITIZER}" ] && [ -x "/usr/local/cuda/bin/compute-sanitizer" ]; then
    COMPUTE_SANITIZER="/usr/local/cuda/bin/compute-sanitizer"
fi
if [ -z "${COMPUTE_SANITIZER}" ] || [ ! -x "${COMPUTE_SANITIZER}" ]; then
    echo "ERROR: compute-sanitizer not found. Add CUDA bin to PATH or install Compute Sanitizer."
    exit 1
fi

export LD_LIBRARY_PATH="${DOCA_LIB_PATH}:${DPDK_LIB_PATH}:${LD_LIBRARY_PATH}"

echo "========================================"
echo "compute-sanitizer --tool ${SANITIZER_TOOL}"
echo "========================================"
echo "Running: sudo -E ${COMPUTE_SANITIZER} --tool ${SANITIZER_TOOL} -- ${EXECUTABLE} $*"
echo ""

cd "${SCRIPT_DIR}"
sudo -E "${COMPUTE_SANITIZER}" --tool "${SANITIZER_TOOL}" -- "${EXECUTABLE}" "$@"
