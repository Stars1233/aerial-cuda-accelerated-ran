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
# Activates the aerial_postproc virtual environment
# Usage: source venv_activate.sh
#

VENV_DIR="${AERIAL_POSTPROC_VENV:-$HOME/.aerial_postproc_venv}"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated aerial_postproc venv at: $VENV_DIR"
else
    echo "Virtual environment not found at: $VENV_DIR"
    echo "Run venv_create.sh first to create the virtual environment."
    return 1 2>/dev/null || exit 1
fi
