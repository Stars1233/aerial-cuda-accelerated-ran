#!/bin/bash -e

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
# AERIAL_BASE_IMAGE=<base image> $cuBB_SDK/pyaerial/container/build.sh [--clean]
#
# Options:
#   --clean    Remove build directory and run cmake configure step before building

# Switch to SCRIPT_DIR directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
echo $SCRIPT starting...
cd $SCRIPT_DIR

# Parse arguments
CLEAN_BUILD_DIR=false
for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN_BUILD_DIR=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done


cuBB_SDK=$(realpath $SCRIPT_DIR/../..)
source $cuBB_SDK/cuPHY-CP/container/setup.sh

AERIAL_PLATFORM=${AERIAL_PLATFORM:-amd64}
TARGETARCH=$(basename $AERIAL_PLATFORM)
case "$TARGETARCH" in
    "amd64")
        CPU_TARGET=x86_64
        ;;
    "arm64")
        CPU_TARGET=aarch64
        ;;
    *)
        echo "Unsupported target architecture"
        exit 1
        ;;
esac

# Base image repository and version.
AERIAL_BASE_IMAGE=${AERIAL_REPO}${AERIAL_IMAGE_NAME}:${AERIAL_VERSION_TAG}

# Target image name.
PYAERIAL_IMAGE=${PYAERIAL_IMAGE:-pyaerial:$USER-${AERIAL_VERSION_TAG}}

cd $cuBB_SDK

# Determine if we need to run cmake configure step
if [[ "$CLEAN_BUILD_DIR" == "true" ]]; then
    rm -rf build
fi

# Build cmake command
CMAKE_BUILD_CMD="cmake --build build -t _pycuphy pycuphycpp"
if [[ ! -d build ]]; then
    if [[ "$TARGETARCH" == "arm64" ]]; then
        TOOLCHAIN="\$cuBB_SDK/cuPHY/cmake/toolchains/grace-cross"
    else
        TOOLCHAIN="\$cuBB_SDK/cuPHY/cmake/toolchains/x86-64"
    fi
    CMAKE_CONFIGURE_CMD="cmake -Bbuild -GNinja --log-level=warning -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN"
    CMAKE_FULL_CMD="$CMAKE_CONFIGURE_CMD && $CMAKE_BUILD_CMD"
else
    CMAKE_FULL_CMD="$CMAKE_BUILD_CMD"
fi

./cuPHY-CP/container/run_aerial.sh bash -x -c "\"$CMAKE_FULL_CMD\""

hpccm --recipe pyaerial/container/pyaerial_recipe.py --cpu-target $CPU_TARGET --format docker --userarg AERIAL_BASE_IMAGE=$AERIAL_BASE_IMAGE > Dockerfile_tmp
if [[ -n "$AERIAL_BUILDER" ]]
then
    docker buildx build --builder $AERIAL_BUILDER --pull --push --progress plain --platform $AERIAL_PLATFORM -t ${PYAERIAL_IMAGE}-${TARGETARCH} -f Dockerfile_tmp .
else
    DOCKER_BUILDKIT=1 docker build --network host --platform $AERIAL_PLATFORM -t ${PYAERIAL_IMAGE}-${TARGETARCH} -f Dockerfile_tmp .
fi
rm Dockerfile_tmp
