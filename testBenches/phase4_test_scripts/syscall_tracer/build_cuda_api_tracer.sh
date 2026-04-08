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

# Build libcuda_api_tracer.so for LD_PRELOAD CUDA API interception.
# No CUDA SDK required at build time.
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building libcuda_api_tracer.so ..."
gcc -shared -fPIC -O2 -Wall -o libcuda_api_tracer.so cuda_api_tracer.c -ldl

if [ -f libcuda_api_tracer.so ]; then
    echo "Built successfully: $SCRIPT_DIR/libcuda_api_tracer.so"
    file libcuda_api_tracer.so
else
    echo "Build failed" >&2
    exit 1
fi
