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
# Creates an isolated virtual environment for aerial_postproc in the user's home directory
# (or a custom location via AERIAL_POSTPROC_VENV environment variable)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${AERIAL_POSTPROC_VENV:-$HOME/.aerial_postproc_venv}"

echo "Creating aerial_postproc virtual environment at: $VENV_DIR"

# Create the virtual environment
python3 -m venv "$VENV_DIR"

# Upgrade pip
"$VENV_DIR/bin/pip" install --upgrade pip

# Install requirements
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing requirements from $SCRIPT_DIR/requirements.txt"
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

# Install the aerial_postproc package in editable mode
echo "Installing aerial_postproc package"
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR"

echo ""
echo "Virtual environment created successfully!"
echo "Location: $VENV_DIR"
echo ""
echo "To activate (optional): source $SCRIPT_DIR/venv_activate.sh"
echo "To run scripts directly: $SCRIPT_DIR/run_isolated.sh <script.py> [args...]"
