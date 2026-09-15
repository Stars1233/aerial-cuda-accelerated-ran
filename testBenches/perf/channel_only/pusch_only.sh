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

# Isolated PUSCH via Phase-3 YAML (--no_pdsch).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.env"

YAML="${1:-${SCRIPT_DIR}/yaml/pusch_only_F14_64TR_run1.yaml}"
if [[ "${YAML}" != /* ]]; then
  YAML="${SCRIPT_DIR}/${YAML}"
fi
OUT_TAG="${2:-pusch_only_F14_64TR_6cell}"
EXTRA_ARGS=("${@:3}")

cd "${PERF_DIR}"
python3 measure.py \
  --yaml "${YAML}" \
  --cuphy "${CUPHY}" \
  --vectors "${VECTORS}" \
  --freq "${FREQ}" \
  --power "${POWER}" \
  --target "${TARGET_SM}" \
  "${EXTRA_ARGS[@]}"

cp "${SWEEP_OUT}" "${OUT_TAG}.json"
echo "Saved: ${PERF_DIR}/${OUT_TAG}.json"
