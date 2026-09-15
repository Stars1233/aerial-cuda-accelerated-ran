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

# F14 64TR TVnr Stage-1 matrix via channel_only/ wrappers.

set -euo pipefail
DIR="$(dirname "$0")"
source "${DIR}/common.env"

echo "=== F14 64TR Stage 1 (channel_only) ==="
echo "FREQ=${FREQ}  TARGET_SM=${TARGET_SM}  VECTORS=${VECTORS}"

"${DIR}/pusch_only.sh" "${DIR}/yaml/pusch_only_F14_64TR_run1.yaml" "1612_pusch_only_F14_64TR_6cell"
"${DIR}/pdsch_only.sh" "${DIR}/yaml/pdsch_only_F14_64TR_run1.yaml" "1612_pdsch_run1_F14_64TR_6cell"
"${DIR}/dl_bf_only.sh" "${DIR}/yaml/dl_bf_only_F14_64TR_run1.yaml" "1612_dlbfw_16L_F14_64TR_6cell"
"${DIR}/srs_only.sh"   "${DIR}/yaml/srs_only_F14_64TR_run1.yaml" "1612_srs_mmse_F14_64TR_6cell"

"${DIR}/pdsch_only.sh" "${DIR}/yaml/pdsch_only_F14_64TR_run2.yaml" "1612_pdsch_run2_F14_64TR_6cell"
"${DIR}/dl_bf_only.sh" "${DIR}/yaml/dl_bf_only_F14_64TR_run2.yaml" "1612_dlbfw_32L_F14_64TR_6cell"
"${DIR}/srs_only.sh"   "${DIR}/yaml/srs_only_F14_64TR_srs_rkhs.yaml" "1612_srs_rkhs_F14_64TR_6cell"

echo ""
echo "Done. Results in ${PERF_DIR}/1612_*_F14_64TR_6cell.json"
ls -la "${PERF_DIR}"/1612_*_F14_64TR_6cell.json 2>/dev/null || true
