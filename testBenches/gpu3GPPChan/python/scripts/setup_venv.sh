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

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
VENV="${PYTHON_ROOT}/.venv"

: "${GPU3GPPCHAN_WHEEL_DIR:?GPU3GPPCHAN_WHEEL_DIR must point to the CMake-built wheel directory}"
: "${GPU3GPPCHAN_PYTHON_EXECUTABLE:=python3}"

if [[ ! -x "${VENV}/bin/python" || ! -x "${VENV}/bin/pip" ]]; then
    echo "Creating gpu3gppchan Python venv at ${VENV}"
    "${GPU3GPPCHAN_PYTHON_EXECUTABLE}" -m venv --clear --system-site-packages "${VENV}"
else
    echo "Reusing gpu3gppchan Python venv at ${VENV}"
fi

mapfile -t GPU3GPPCHAN_WHEELS < <(
    find "${GPU3GPPCHAN_WHEEL_DIR}" -maxdepth 1 -type f -name 'gpu3gppchan-*.whl' -print
)
if [[ ${#GPU3GPPCHAN_WHEELS[@]} -ne 1 ]]; then
    echo "Expected exactly one gpu3gppchan wheel in ${GPU3GPPCHAN_WHEEL_DIR}; found ${#GPU3GPPCHAN_WHEELS[@]}" >&2
    exit 1
fi

"${VENV}/bin/pip" install --force-reinstall --no-deps "${GPU3GPPCHAN_WHEELS[0]}"

echo "gpu3gppchan Python venv setup complete"
