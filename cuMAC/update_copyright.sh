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

# Update copyright in each file to match code compliance (SPDX format YYYY-YYYY, no spaces).
# Uses earliest commit year per file from git with --follow (same as compliance checker),
# clamped to OPEN_SOURCE_RELEASE_YEAR. If no history (e.g. file not in base branch in CI),
# uses CURRENT_YEAR so compliance "first publicly available in 2026" is satisfied.
# Paths in CURRENT_YEAR_ONLY are not in target branch at merge base in CI → always use current year.
set -e
OPEN_SOURCE_RELEASE_YEAR=2025
CURRENT_YEAR=2026
# Files that CI treats as first publicly available in current year (not in base branch)
CURRENT_YEAR_ONLY=(
  cuMAC/examples/muMimoUeGrpL2Integration/l1_muUeGrp_test.cpp
  cuMAC/examples/muMimoUeGrpL2Integration/l1_muUeGrp_test.h
  cuMAC/examples/muMimoUeGrpL2Integration/run_tests.sh
  cuMAC/examples/muMimoUeGrpL2Integration/simple_srs_memory_bank.cpp
  cuMAC/examples/muMimoUeGrpL2Integration/simple_srs_memory_bank.hpp
  cuMAC/lib/cumac_msg/cumac_muUeGrp.h
  cuMAC/src/muMimoUserPairing/muMimoUserPairing.cu
  cuMAC/src/muMimoUserPairing/muMimoUserPairing.cuh
)
cd "$(dirname "$0")/.."
for f in $(find cuMAC -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" -o -name "*.py" -o -name "*.yaml" -o -name "*.sh" -o -name "CMakeLists.txt" \) ! -path "cuMAC/eigen/*" 2>/dev/null); do
  [ -f "$f" ] || continue
  grep -q "SPDX-FileCopyrightText: Copyright" "$f" 2>/dev/null || continue
  # Override: these paths must use current year only (CI: first publicly available in current year)
  use_current_only=0
  for override in "${CURRENT_YEAR_ONLY[@]}"; do
    [ "$f" = "$override" ] && use_current_only=1 && break
  done
  if [ "$use_current_only" = 1 ]; then
    year=$CURRENT_YEAR
  else
    year=$(git log --all --reverse --follow --format='%ad' --date=format:'%Y' -- "$f" 2>/dev/null | head -1)
    [ -z "$year" ] && year=$CURRENT_YEAR
    [ "$year" -lt "$OPEN_SOURCE_RELEASE_YEAR" ] 2>/dev/null && year=$OPEN_SOURCE_RELEASE_YEAR
  fi
  if [ "$year" = "$CURRENT_YEAR" ]; then
    new="$CURRENT_YEAR"
  else
    new="${year}-${CURRENT_YEAR}"
  fi
  # Match "Copyright (c) YYYY" or "Copyright (c) YYYY-CURRENT_YEAR" (with or without spaces)
  sed -i "s/Copyright (c) [0-9]\{4\}\( *- *${CURRENT_YEAR}\)\?/Copyright (c) $new/g" "$f"
  echo "$f -> $new"
done
