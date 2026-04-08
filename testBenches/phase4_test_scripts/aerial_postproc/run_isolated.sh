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

#
# Runs a Python script using the isolated aerial_postproc virtual environment
# without needing to activate it.
#
# Usage: ./run_isolated.sh <script.py> [args...]
#
# Example:
#   ./run_isolated.sh scripts/cicd/cicd_parse.py --input file.log
#

VENV_DIR="${AERIAL_POSTPROC_VENV:-$HOME/.aerial_postproc_venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at: $VENV_DIR"
    echo "Run venv_create.sh first to create the virtual environment."
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

exec "$VENV_DIR/bin/python" "$@"
