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

SCRIPT=$(readlink -f "$0")
PYAERIAL_ROOT=$(dirname "$(dirname "$SCRIPT")")
ARCH="${PYAERIAL_ARCH:-amd64}"
: "${GPU3GPPCHAN_WHEEL_DIR:?GPU3GPPCHAN_WHEEL_DIR must point to the CMake-built wheel directory}"

# shellcheck source=_env.sh
source "${SCRIPT%/*}/_env.sh"

export BUILD_ID="${BUILD_ID:-1}"

case "${ARCH}" in
    amd64) ML_EXTRAS="ml,ml-amd64" ;;
    arm64) ML_EXTRAS="ml,ml-arm64" ;;
    *) echo "Unsupported PYAERIAL_ARCH: ${ARCH} (expected amd64 or arm64)" >&2; exit 1 ;;
esac

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "Creating pyAerial venv at ${VENV} (arch=${ARCH})"
    python3 -m venv --prompt "${PYAERIAL_VENV_PROMPT}" "${VENV}"
else
    echo "Reusing pyAerial venv at ${VENV} (arch=${ARCH})"
fi

"${VENV}/bin/pip" install --upgrade pip

mapfile -t GPU3GPPCHAN_WHEELS < <(
    find "${GPU3GPPCHAN_WHEEL_DIR}" -maxdepth 1 -type f \
        -name 'gpu3gppchan-*.whl' -print
)
if [[ ${#GPU3GPPCHAN_WHEELS[@]} -ne 1 ]]; then
    echo "Expected exactly one gpu3gppchan wheel in ${GPU3GPPCHAN_WHEEL_DIR}; found ${#GPU3GPPCHAN_WHEELS[@]}" >&2
    exit 1
fi
"${VENV}/bin/pip" install --force-reinstall --no-deps \
    "${GPU3GPPCHAN_WHEELS[0]}"

"${VENV}/bin/pip" install \
    --extra-index-url "${TORCH_EXTRA_INDEX_URL}" \
    -e "${PYAERIAL_ROOT}[dev,${ML_EXTRAS},notebooks]"

echo "pyAerial venv setup complete"
