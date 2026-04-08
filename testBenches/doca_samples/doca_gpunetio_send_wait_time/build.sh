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

# Build script for DOCA gpunetio_send_wait_time sample application
# Usage: ./build.sh [--clean]
#
# Uses out-of-tree build to avoid need for sudo during build.

set -e

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# DOCA sample source directory (read-only)
DOCA_SAMPLE_SRC="/opt/mellanox/doca/samples/doca_gpunetio/gpunetio_send_wait_time"

# Build directory - use local filesystem to avoid NFS clock skew issues
# Falls back to script directory if /tmp is not available
if [ -d "/tmp" ] && [ -w "/tmp" ]; then
    BUILD_DIR="/tmp/doca_builds/gpunetio_send_wait_time/build"
else
    BUILD_DIR="${SCRIPT_DIR}/build"
fi

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

echo "Using library architecture: ${LIB_ARCH}"

# Define pkg-config paths
PKG_CONFIG_DPDK="/opt/mellanox/dpdk/lib/${LIB_ARCH}/pkgconfig"
PKG_CONFIG_DOCA="/opt/mellanox/doca/lib/${LIB_ARCH}/pkgconfig"

# Stub for doca-rdma when not installed (gpunetio_send_wait_time does not use doca_rdma;
# upstream meson.build declares it as a dependency). Generate stub at build time so we do
# not depend on any committed pkgconfig file.
if [ ! -f "${PKG_CONFIG_DOCA}/doca-rdma.pc" ]; then
    PKG_CONFIG_STUB_DIR="$(dirname "${BUILD_DIR}")/pkgconfig_stub"
    mkdir -p "${PKG_CONFIG_STUB_DIR}"
    cat > "${PKG_CONFIG_STUB_DIR}/doca-rdma.pc" << STUBPC
prefix=/opt/mellanox/doca
libdir=\${prefix}/lib/${LIB_ARCH}
includedir=\${prefix}/include

Name: doca_rdma
Description: DOCA RDMA stub (satisfies dependency when doca-rdma not installed)
Version: 3.2.1025
Requires: doca-common
Cflags: -I\${includedir}
Libs:
STUBPC
    echo "Using generated stub doca-rdma.pc (doca-rdma not installed in DOCA)"
    export PKG_CONFIG_PATH="${PKG_CONFIG_STUB_DIR}:${PKG_CONFIG_DPDK}:${PKG_CONFIG_DOCA}:${PKG_CONFIG_PATH}"
else
    export PKG_CONFIG_PATH="${PKG_CONFIG_DPDK}:${PKG_CONFIG_DOCA}:${PKG_CONFIG_PATH}"
fi

# Check if DOCA sample directory exists
if [ ! -d "${DOCA_SAMPLE_SRC}" ]; then
    echo "ERROR: DOCA sample directory not found: ${DOCA_SAMPLE_SRC}"
    echo "Please ensure DOCA SDK is installed."
    exit 1
fi

# Check for required dependencies
echo "Checking dependencies..."

# Check for doca-common (required by doca-rdma stub and meson)
if [ ! -f "${PKG_CONFIG_DOCA}/doca-common.pc" ]; then
    echo "WARNING: doca-common.pc not found. Attempting to install libdoca-sdk-common-dev..."
    sudo apt-get install -y libdoca-sdk-common-dev || {
        echo "ERROR: Failed to install libdoca-sdk-common-dev"
        exit 1
    }
fi

# Check for libdoca-sdk-flow-dev package; if not installed, update and install
if ! dpkg -s libdoca-sdk-flow-dev &>/dev/null; then
    echo "libdoca-sdk-flow-dev not installed. Running apt-get update and install..."
    sudo apt-get update || { echo "ERROR: apt-get update failed"; exit 1; }
    sudo apt-get install -y libdoca-sdk-flow-dev || {
        echo "ERROR: Failed to install libdoca-sdk-flow-dev"
        exit 1
    }
fi

# Check for libcuda.so symlink (needed for meson CUDA module)
CUDA_LIB_DIR="/usr/local/cuda/lib64"
if [ ! -f "${CUDA_LIB_DIR}/libcuda.so" ]; then
    echo "Creating libcuda.so symlink from stubs directory..."
    if [ -f "${CUDA_LIB_DIR}/stubs/libcuda.so" ]; then
        sudo ln -sf "${CUDA_LIB_DIR}/stubs/libcuda.so" "${CUDA_LIB_DIR}/libcuda.so"
    else
        echo "ERROR: libcuda.so stub not found in ${CUDA_LIB_DIR}/stubs/"
        exit 1
    fi
fi

# Handle --clean option
if [ "$1" == "--clean" ]; then
    echo "Cleaning build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
    rm -rf "$(dirname "${BUILD_DIR}")/pkgconfig_stub"
    # Also clean script directory build if it exists
    rm -rf "${SCRIPT_DIR}/build"
    echo "Clean complete."
    if [ "$2" != "--rebuild" ]; then
        exit 0
    fi
fi

# Create parent directory if using /tmp
mkdir -p "$(dirname "${BUILD_DIR}")"

# Remove existing build directory if present
if [ -d "${BUILD_DIR}" ]; then
    echo "Removing existing build directory..."
    rm -rf "${BUILD_DIR}"
fi

echo "========================================"
echo "Running Meson setup (out-of-tree build)..."
echo "========================================"
echo "Source directory: ${DOCA_SAMPLE_SRC}"
echo "Build directory:  ${BUILD_DIR}"

CUDA_PATH="/usr/local/cuda" \
meson setup "${BUILD_DIR}" "${DOCA_SAMPLE_SRC}"

echo "========================================"
echo "Building with Ninja..."
echo "========================================"

ninja -C "${BUILD_DIR}"

echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo "Executable: ${BUILD_DIR}/doca_gpunetio_send_wait_time"
