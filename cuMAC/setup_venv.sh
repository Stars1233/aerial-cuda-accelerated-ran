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

# Create the isolated environment used only to generate cuMAC ONNX models.
# Keep PyTorch and its CUDA dependencies out of the main Aerial container image.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"

BASE_PYTHON_REQUESTED="${BASE_PYTHON:-python3}"
BASE_PYTHON="$(command -v "${BASE_PYTHON_REQUESTED}" || true)"

if [[ -z "${BASE_PYTHON}" || ! -x "${BASE_PYTHON}" ]]; then
    echo "ERROR: Python interpreter not found: ${BASE_PYTHON_REQUESTED}" >&2
    echo "Install python3, or set BASE_PYTHON explicitly." >&2
    exit 1
fi

if [[ "$("${BASE_PYTHON}" -c 'import sys, venv; print("cuMAC-python-ok")' 2>/dev/null)" \
      != "cuMAC-python-ok" ]]; then
    echo "ERROR: Python interpreter does not support venv: ${BASE_PYTHON}" >&2
    echo "Install python3-venv, or set BASE_PYTHON to an interpreter with venv support." >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${BASE_PYTHON}" -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install \
    'torch==2.13.0' \
    'onnx==1.22.0' \
    'pyyaml==6.0.3'

"${PYTHON}" -c 'import onnx, torch, yaml; print(f"cuMAC model-generation venv ready: torch={torch.__version__}, onnx={onnx.__version__}")'
