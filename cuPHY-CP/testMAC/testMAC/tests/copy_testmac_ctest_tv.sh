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

# copy_testmac_ctest_tv.sh — copy launch-pattern YAMLs and test vectors required
# by test_testmac_core.  Invoked as a CTest FIXTURES_SETUP step before
# testmac_core.unit_tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cuBB_SDK="${cuBB_SDK:-$(realpath "${SCRIPT_DIR}/../../../..")}"
COPY_SCRIPT="${cuBB_SDK}/testBenches/phase4_test_scripts/copy_test_files.sh"
DST="${cuBB_SDK}/testVectors"

# Launch-pattern suffix and max cell count pairs from
# test_testmac_core_config.yaml (launch_pattern_F08_*C_<suffix>.yaml).
readonly PATTERN_SPECS=(
    "60:20"   # F08_1C_60, F08_20C_60
    "66c:3"   # F08_3C_66c
)

mkdir -p "${DST}/multi-cell"

for spec in "${PATTERN_SPECS[@]}"; do
    pattern="${spec%%:*}"
    max_cells="${spec#*:}"
    copy_args=(--dst "${DST}" --max_cells "${max_cells}")
    if [[ -n "${TESTVECTOR_DIR:-}" ]]; then
        copy_args=(--src "${TESTVECTOR_DIR}" "${copy_args[@]}")
    fi

    echo "=== copy_testmac_ctest_tv: F08 pattern ${pattern} (max_cells=${max_cells}) ==="
    "${COPY_SCRIPT}" "${copy_args[@]}" "${pattern}"
done
