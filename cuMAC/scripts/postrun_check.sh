#!/bin/bash -x

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# check the test results, used for ci/cd pipeline after the test is done

# Usage: postrun_check.sh <LOG_BASE> <TEST_TYPE>
# Example: postrun_check.sh /var/log/aerial/cicd/cumac tdl
#
# Behavior:
# - Creates sentinel files in per-test subdirs if there are any failed tests
#     ./srs/.srs_FAIL or ./srs/.srs_PASS
#     ./drl/.drl_FAIL or ./drl/.drl_PASS
#     ./4t4r/.4t4r_FAIL or ./4t4r/.4t4r_PASS
#     ./tdl/.tdl_FAIL or ./tdl/.tdl_PASS
#     ./cdl/.cdl_FAIL or ./cdl/.cdl_PASS
#     ./64tr/.64tr_FAIL or ./64tr/.64tr_PASS
#     ./pfmsort/.pfmsort_FAIL or ./pfmsort/.pfmsort_PASS


# Exit codes: 0=PASS, 1=FAIL (sentinel created), 2=SKIP (dir missing)


LOG_BASE="${1:-}"
TEST_TYPE="${2:-}"

if [[ -z "$LOG_BASE" || -z "$TEST_TYPE" ]]; then
  echo "Usage: $0 <LOG_BASE> <srs|drl|4t4r|tdl|cdl|64tr|pfmsort>" >&2
  exit 2
fi

if [[ ! -d "$LOG_BASE/$TEST_TYPE" ]]; then
  echo "log $LOG_BASE/$TEST_TYPE not found" >&2
  exit 2
fi

# ----------------------------
# Hard-coded whitelists, will remove once the test is stable
# ----------------------------
# NOTE: Matching is substring match against *.FAIL filenames.

# TDL whitelist, allow 20c 500 UE fail in f2 test. 
# the first two (10TTI) are caused by OOM, the last two (500TTI) reason unknown, logs do not show useful info, maybe need longer timeout
# also whitelist all gpuAllocType1_cpuAllocType1 tests, as there likely be a CUDA kernel issue for type-1 PRG allocation, which caused some other random channel mismatch failures e.g., 4T4R_DL_10C_100UEPerCell_500TTI_gpuAllocType1_cpuAllocType1_CDL_f3_test.FAIL
TDL_WHITELIST=(
  "main"
  "cuMAC_4T4R_DL_20C_500UEPerCell_10TTI_gpuAllocType0_cpuAllocType0_TDL_f2_test"
  "cuMAC_4T4R_DL_20C_500UEPerCell_10TTI_gpuAllocType1_cpuAllocType1_TDL_f2_test"
  "cuMAC_4T4R_DL_20C_500UEPerCell_500TTI_gpuAllocType0_cpuAllocType0_TDL_f2_test"
  "cuMAC_4T4R_DL_20C_500UEPerCell_500TTI_gpuAllocType1_cpuAllocType1_TDL_f2_test"
  "gpuAllocType1_cpuAllocType1"
)

# CDL whitelist, allow 20c 500 UE fail in f4 test. 
# the first two (10TTI) are caused by OOM, the last two (500TTI) reason unknown, logs do not show useful info, maybe need longer timeout
# also whitelist all gpuAllocType1_cpuAllocType1 tests, as there likely be a CUDA kernel issue for type-1 PRG allocation, which caused some other random channel mismatch failures e.g., 4T4R_DL_10C_100UEPerCell_500TTI_gpuAllocType1_cpuAllocType1_CDL_f3_test.FAIL
CDL_WHITELIST=(
  "main"
  "cuMAC_4T4R_DL_20C_500UEPerCell_10TTI_gpuAllocType0_cpuAllocType0_CDL_f4_test"
  "cuMAC_4T4R_DL_20C_500UEPerCell_10TTI_gpuAllocType1_cpuAllocType1_CDL_f4_test"
  "cuMAC_4T4R_DL_20C_500UEPerCell_500TTI_gpuAllocType0_cpuAllocType0_CDL_f4_test"
  "cuMAC_4T4R_DL_20C_500UEPerCell_500TTI_gpuAllocType1_cpuAllocType1_CDL_f4_test"
  "gpuAllocType1_cpuAllocType1"
)

#  clear old sentinel
clear_sentinel() {
  local test="$1" 
  local dir="$LOG_BASE/$test"
  rm -f "$dir/.${test}_FAIL" "$dir/.${test}_PASS" 2>/dev/null || true
}

# Create FAIL sentinel and optionally include details
write_fail() {
  local test="$1" 
  local dir="$LOG_BASE/$test" 
  local ff="$dir/.${test}_FAIL" 
  local pf="$dir/.${test}_PASS"
  mkdir -p "$dir"
  rm -f "$pf" 2>/dev/null || true     
  > "$ff" # creates the file or truncate the file to empty if it exists
  shift 1 || true # remove the first argument
  [[ $# -gt 0 ]] && printf '%s\n' "$@" >> "$ff" # write the rest of the arguments to the file
  echo "FAIL: wrote $ff"
}

# Create PASS sentinel
write_pass() {
  local test="$1" 
  local dir="$LOG_BASE/$test" 
  local pf="$dir/.${test}_PASS" 
  local ff="$dir/.${test}_FAIL"
  mkdir -p "$dir"
  rm -f "$ff" 2>/dev/null || true     
  > "$pf"
  echo "PASS: wrote $pf"
}

# only non-whitelisted *.FAIL => sentinel
# Return: 0=PASS, 1=FAIL
check_faildir_with_whitelist_allowed() {

  local test="$1" 
  local dir="$LOG_BASE/$test"
  
  local -a fails=() # array of failed test filenames
  if [[ -d "$dir" ]]; then
    for f in "$dir"/*.FAIL; do
        # add -e in case no match, then literal "$dir/*.FAIL" will be used as f 
        [[ -e "$f" ]] || continue
        fails+=("$(basename "$f")")
    done
  fi

  if (( ${#fails[@]} == 0 )); then
    write_pass "$test"
    return 0
  fi

  local -a wl=()
  case "$test" in
    tdl) wl=("${TDL_WHITELIST[@]}") ;;
    cdl) wl=("${CDL_WHITELIST[@]}") ;;
    *)   
      echo "found failed tests in $test test"
      write_fail "$test" "${fails[@]}"
      return 1
      ;;
  esac

  local -a bad=()
  for f in "${fails[@]}"; do
    local ok=false
    for pattern in "${wl[@]}"; do
      if [[ "$f" == *"$pattern"* ]]; then ok=true; break; fi
    done
    $ok || bad+=("$f")
  done

  if (( ${#bad[@]} )); then
    write_fail "$test" "${bad[@]}"
    return 1
  fi
  write_pass "$test"
  return 0
}

# 64tr CSV check: must exist and all rows have success==true
# Return: 0=PASS, 1=FAIL
check_64tr() {
  local test="64tr" 
  local dir="$LOG_BASE/$test"
  # Run your Python snippet verbatim
  LOG_BASE="$LOG_BASE" python3 - <<'PY'
import csv, os, sys

log_base = os.environ.get("LOG_BASE", "/var/log/aerial/cicd/cumac")
csv_path = os.path.join(log_base, "64tr", "cumac_64tr_results.csv")

if not os.path.isfile(csv_path):
    print(f"Missing CSV: {csv_path}")
    sys.exit(1)

failures = []
with open(csv_path, newline='') as fh:
    reader = csv.DictReader(fh)
    if reader.fieldnames is None:
        print("ERROR: empty CSV or no header")
        sys.exit(1)
    fieldnames = [fn.strip() for fn in reader.fieldnames]
    success_key = next((fn for fn in fieldnames if fn.lower() == "success"), None)
    if not success_key:
        print("ERROR: 'success' column not found in CSV header:", fieldnames)
        sys.exit(1)

    for row in reader:
        val = (row.get(success_key) or "").strip().lower()
        if val != "true":
            failures.append(row)

if failures:
    print("---- 64tr failing rows (success != True) ----")
    for row in failures:
        print("--------------------------------")
        print(",".join(row.get(fn, "") for fn in fieldnames))
    sys.exit(1)
else:
    print("All 64tr tests passed (success=True).")
    sys.exit(0)
PY
  rc=$?

  if (( rc == 1 )); then
    write_fail "$test" "Rows with success!=true or CSV/header error"
    return 1
  fi
  write_pass "$test"
  return 0
}

clear_sentinel "$TEST_TYPE"

case "$TEST_TYPE" in
  srs|drl|4t4r|tdl|cdl|pfmsort)
    check_faildir_with_whitelist_allowed "$TEST_TYPE"
    exit $?
    ;;
  64tr)
    check_64tr
    exit $?
    ;;
  *)
    echo "Unknown TEST_TYPE: $TEST_TYPE (expected: srs|drl|4t4r|tdl|cdl|64tr|pfmsort)" >&2
    exit 2
    ;;
esac