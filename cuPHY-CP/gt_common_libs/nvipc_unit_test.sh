#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LOCAL_DIR=$(dirname $(readlink -f "$0"))
cd $LOCAL_DIR

# Check for required external dependencies
REQUIRED_DIRS=(
    "external/fmtlog"
    "external/libyaml"
    "external/yaml-cpp"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Required external library '$dir' does not exist"
        echo "Please run ./pack_nvipc.sh to get the source code and run this script in the <nvipc_src> directory"
        exit 1
    fi
done

echo "All required external dependencies found"

mkdir -p build && cd build

function build_and_test {
    options="$1 -DENABLE_TESTS=ON"
    echo "Build and test start ... cmake options: $options"

    rm -rf *

    # Configure
    cmake .. $options
    if [ $? -ne 0 ]; then
        echo "Configure failed with cmake options: $options"
        exit 1
    fi

    # Build
    make -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Build failed with cmake options: $options"
        exit 1
    fi

    # Run SHM IPC unit test
    if [ `whoami` = "root" ];then
        ./nvIPC/tests/cunit/nvipc_cunit 3 2
    else
        sudo ./nvIPC/tests/cunit/nvipc_cunit 3 2
    fi

    if [ $? -ne 0 ]; then
        echo "Test failed with cmake options: $options"
        exit 1
    fi

    echo "Build and test passed: cmake options: $options"
    return 0
}

# Test with default options
build_and_test ""

# Test without fmtlog support for nvIPC
build_and_test "-DNVIPC_FMTLOG_ENABLE=OFF"

# Test with CUDA disabled
build_and_test "-DNVIPC_CUDA_ENABLE=OFF"

echo "libnvipc build and test all passed"
exit 0
