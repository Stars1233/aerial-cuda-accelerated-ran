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

# Exit on first error
set -e

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
CUPHY_ROOT=$(realpath $PROJECT_ROOT/..)
echo $SCRIPT starting...
cd $PROJECT_ROOT

# Define BUILD_ID if it does not exist yet.
if [ -z "${BUILD_ID}" ]; then
    export BUILD_ID=1
fi

# Build the package
rm -rf dist
python3 -m build

# Install the package in developer mode
sudo chmod o+w -R src/pyaerial.egg-info || chmod o+w -R src/pyaerial.egg-info
pip install -e .

# Finished
echo $SCRIPT finished.
