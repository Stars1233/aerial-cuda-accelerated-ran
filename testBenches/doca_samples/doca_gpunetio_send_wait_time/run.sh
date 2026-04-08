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

# Run script for DOCA gpunetio_send_wait_time sample application
#
# Usage: ./run.sh -n <NIC_PCIE_ADDR> -g <GPU_PCIE_ADDR> [options]
#
# Required arguments:
#   -n <NIC_PCIE_ADDR>   PCIe address of the NIC port (e.g., 0002:03:00.0)
#   -g <GPU_PCIE_ADDR>   PCIe address of the GPU (e.g., 0002:09:00.0)
#
# Optional arguments:
#   -t <TIME_NS>         Wait time in nanoseconds (default: 5000000)
#   -h                   Show this help message
#
# All other arguments are passed directly to the application.
#
# Example:
#   ./run.sh -n 0002:03:00.0 -g 0002:09:00.0 -t 5000000

set -e

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# Executable - check /tmp first (to avoid NFS), then script directory
if [ -f "/tmp/doca_builds/gpunetio_send_wait_time/build/doca_gpunetio_send_wait_time" ]; then
    BUILD_DIR="/tmp/doca_builds/gpunetio_send_wait_time/build"
else
    BUILD_DIR="${SCRIPT_DIR}/build"
fi
EXECUTABLE="${BUILD_DIR}/doca_gpunetio_send_wait_time"

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: ${ARCH}"

# Set architecture-specific library paths
case "${ARCH}" in
    x86_64)
        LIB_ARCH="x86_64-linux-gnu"
        ;;
    aarch64)
        LIB_ARCH="aarch64-linux-gnu"
        ;;
    *)
        echo "ERROR: Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

# Define library paths
DOCA_LIB_PATH="/opt/mellanox/doca/lib/${LIB_ARCH}"
DPDK_LIB_PATH="/opt/mellanox/dpdk/lib/${LIB_ARCH}"

# Function to display usage
usage() {
    echo "Usage: $0 -n <NIC_PCIE_ADDR> -g <GPU_PCIE_ADDR> [options]"
    echo ""
    echo "Required arguments:"
    echo "  -n <NIC_PCIE_ADDR>   PCIe address of the NIC port (e.g., 0002:03:00.0)"
    echo "  -g <GPU_PCIE_ADDR>   PCIe address of the GPU (e.g., 0002:09:00.0)"
    echo ""
    echo "Optional arguments:"
    echo "  -t <TIME_NS>         Wait time in nanoseconds (default: 5000000)"
    echo "  -h                   Show this help message"
    echo ""
    echo "All other arguments are passed directly to the application."
    echo ""
    echo "Example:"
    echo "  $0 -n 0002:03:00.0 -g 0002:09:00.0 -t 5000000"
    echo ""
    echo "To discover PCIe addresses:"
    echo "  NIC: lspci | grep -i mellanox"
    echo "  GPU: lspci | grep -i nvidia"
    exit 1
}

# Function to discover and display available devices
show_devices() {
    echo ""
    echo "Available Mellanox NICs:"
    lspci | grep -i mellanox || echo "  No Mellanox NICs found"
    echo ""
    echo "Available NVIDIA GPUs:"
    lspci | grep -i nvidia || echo "  No NVIDIA GPUs found"
    echo ""
}

# Check if executable exists
if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: Executable not found: ${EXECUTABLE}"
    echo "Please run build.sh first."
    exit 1
fi

# Parse arguments
NIC_ADDR=""
GPU_ADDR=""
WAIT_TIME=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            NIC_ADDR="$2"
            shift 2
            ;;
        -g)
            GPU_ADDR="$2"
            shift 2
            ;;
        -t)
            WAIT_TIME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "${NIC_ADDR}" ] || [ -z "${GPU_ADDR}" ]; then
    echo "ERROR: Both NIC (-n) and GPU (-g) PCIe addresses are required."
    show_devices
    usage
fi

# Build command line arguments
CMD_ARGS="-n ${NIC_ADDR} -g ${GPU_ADDR}"

if [ -n "${WAIT_TIME}" ]; then
    CMD_ARGS="${CMD_ARGS} -t ${WAIT_TIME}"
fi

if [ -n "${EXTRA_ARGS}" ]; then
    CMD_ARGS="${CMD_ARGS} ${EXTRA_ARGS}"
fi

echo "========================================"
echo "DOCA gpunetio_send_wait_time"
echo "========================================"
echo "Architecture: ${ARCH}"
echo "NIC PCIe Address: ${NIC_ADDR}"
echo "GPU PCIe Address: ${GPU_ADDR}"
if [ -n "${WAIT_TIME}" ]; then
    echo "Wait Time: ${WAIT_TIME} ns"
fi
echo "========================================"

# Set library path and run
export LD_LIBRARY_PATH="${DOCA_LIB_PATH}:${DPDK_LIB_PATH}:${LD_LIBRARY_PATH}"

echo "Running: ${EXECUTABLE} ${CMD_ARGS}"
echo ""

cd "${SCRIPT_DIR}"
sudo -E "${EXECUTABLE}" ${CMD_ARGS}

