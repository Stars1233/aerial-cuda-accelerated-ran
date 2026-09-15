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

# Run channel_only wrappers and validate JSON isolation + latencies.
#
# Usage:
#   ./verify_channel_only.sh              # full Stage-1 matrix (7 runs)
#   ./verify_channel_only.sh --quick      # smoke: 4 runs (run1 configs)
#   ./verify_channel_only.sh --check-only # validate existing JSON only
#   VERIFY_TARGET_SM=1 ./verify_channel_only.sh --quick  # also test TARGET_SM wiring
#
# Environment: same as other channel_only/*.sh (see common.env).

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "${DIR}/common.env"

MODE="full"
CHECK_ONLY=0
VERIFY_TARGET_SM="${VERIFY_TARGET_SM:-0}"

usage() {
  echo "Usage: $0 [--quick | --full | --check-only]"
  echo "  --quick       Run 4 smoke tests (PUSCH/PDSCH/DLBFW/SRS run1)"
  echo "  --full        Run full F14 Stage-1 matrix (7 tests, default)"
  echo "  --check-only  Skip measure.py runs; validate JSON files in PERF_DIR"
  echo ""
  echo "Optional: VERIFY_TARGET_SM=1 runs SRS twice (82 vs 16 SM) to confirm --target wiring."
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE="quick" ;;
    --full) MODE="full" ;;
    --check-only) CHECK_ONLY=1 ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
  esac
  shift
done

CELL_KEY="$(printf '%02d+00' "${CELLS_START}")"
BENCH="${CUPHY}/cubb_gpu_test_bench/cubb_gpu_test_bench"
FAILURES=0

log() { echo "[verify] $*"; }
fail() { echo "[verify] FAIL: $*" >&2; FAILURES=$((FAILURES + 1)); }

preflight() {
  log "Preflight (AUTO_DETECT_GPU=${AUTO_DETECT_GPU}, FREQ=${FREQ}, POWER=${POWER}, TARGET_SM=${TARGET_SM}, CELL_KEY=${CELL_KEY})"
  if [[ ! -x "${BENCH}" ]]; then
    fail "Missing cubb_gpu_test_bench: ${BENCH} (rebuild: cmake --preset perf.aarch64 && cmake --build build.perf.aarch64 -j)"
  fi
  if [[ ! -f "${PERF_DIR}/uc_avg_F14_TDD.json" ]] && [[ ! -f "${UC}" ]]; then
    fail "Missing use case file: ${UC}"
  fi
  local tvs=(
    TVnr_ULMIX_21564_PUSCH_gNB_CUPHY_s4p7.h5
    TVnr_DLMIX_20566_PDSCH_gNB_CUPHY_s6p15.h5
    TVnr_DLMIX_20956_PDSCH_gNB_CUPHY_s6p31.h5
    TVnr_9366_BFW_gNB_CUPHY_s0.h5
    TVnr_9464_BFW_gNB_CUPHY_s0.h5
    TVnr_ULMIX_21673_SRS_gNB_CUPHY_s3p47.h5
  )
  for tv in "${tvs[@]}"; do
    if [[ ! -f "${VECTORS}/${tv}" ]]; then
      fail "Missing test vector: ${VECTORS}/${tv}"
    fi
  done
  if [[ "${MODE}" == "full" ]] && [[ ! -f "${VECTORS}/TVnr_ULMIX_21673_SRS_gNB_CUPHY_s3p47_rkhs.h5" ]]; then
    fail "Missing RKHS test vector (full matrix): ${VECTORS}/TVnr_ULMIX_21673_SRS_gNB_CUPHY_s3p47_rkhs.h5"
  fi
}

run_matrix() {
  if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    log "Skipping runs (--check-only)"
    return 0
  fi

  log "Running channel_only matrix (mode=${MODE})"
  if [[ "${MODE}" == "quick" ]]; then
    "${DIR}/pusch_only.sh" "${DIR}/yaml/pusch_only_F14_64TR_run1.yaml" "verify_pusch_only"
    "${DIR}/pdsch_only.sh" "${DIR}/yaml/pdsch_only_F14_64TR_run1.yaml" "verify_pdsch_only"
    "${DIR}/dl_bf_only.sh" "${DIR}/yaml/dl_bf_only_F14_64TR_run1.yaml" "verify_dlbfw_only"
    "${DIR}/srs_only.sh"   "${DIR}/yaml/srs_only_F14_64TR_run1.yaml" "verify_srs_only"
  else
    "${DIR}/run_f14_stage1.sh"
  fi
}

run_target_sm_check() {
  if [[ "${CHECK_ONLY}" -eq 1 ]] || [[ "${VERIFY_TARGET_SM}" != "1" ]]; then
    return 0
  fi

  log "TARGET_SM check: SRS @ 82 SM vs 16 SM"
  TARGET_SM=82 "${DIR}/srs_only.sh" "${DIR}/yaml/srs_only_F14_64TR_run1.yaml" "verify_srs_target_82"
  TARGET_SM=16 "${DIR}/srs_only.sh" "${DIR}/yaml/srs_only_F14_64TR_run1.yaml" "verify_srs_target_16"

  python3 - "${PERF_DIR}/verify_srs_target_82.json" "${PERF_DIR}/verify_srs_target_16.json" << 'PY'
import json, statistics, sys

def mean_srs(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    key = next(k for k in data if k.endswith("+00"))
    srs = data[key].get("SRS1") or []
    if not srs:
        raise SystemExit(f"no SRS1 data in {path}")
    return statistics.mean(srs)

m82 = mean_srs(sys.argv[1])
m16 = mean_srs(sys.argv[2])
print(f"[verify] SRS1 mean @ 82 SM: {m82:.1f} us")
print(f"[verify] SRS1 mean @ 16 SM: {m16:.1f} us")
if abs(m82 - m16) < 1.0:
    raise SystemExit("[verify] FAIL: TARGET_SM had no effect (latencies identical)")
print("[verify] TARGET_SM check passed (latencies differ)")
PY
}

validate_json() {
  log "Validating JSON outputs in ${PERF_DIR}"
  python3 - "${MODE}" "${PERF_DIR}" "${CELL_KEY}" << 'PY'
import json
import statistics
import sys
from pathlib import Path

mode = sys.argv[1]
perf_dir = Path(sys.argv[2])
cell_key = sys.argv[3]

if mode == "quick":
    cases = [
        ("verify_pusch_only.json", ["PUSCH1", "PUSCH2"], ["PDSCH", "DLBFW", "SRS1", "ULBFW1"]),
        ("verify_pdsch_only.json", ["PDSCH"], ["PUSCH1", "PUSCH2", "DLBFW", "SRS1", "ULBFW1"]),
        ("verify_dlbfw_only.json", ["DLBFW"], ["PUSCH1", "PUSCH2", "SRS1", "ULBFW1"]),
        ("verify_srs_only.json", ["SRS1"], ["PUSCH1", "PUSCH2", "PDSCH", "DLBFW", "ULBFW1"]),
    ]
else:
    cases = [
        ("1612_pusch_only_F14_64TR_6cell.json", ["PUSCH1", "PUSCH2"], ["PDSCH", "DLBFW", "SRS1", "ULBFW1"]),
        ("1612_pdsch_run1_F14_64TR_6cell.json", ["PDSCH"], ["PUSCH1", "PUSCH2", "DLBFW", "SRS1", "ULBFW1"]),
        ("1612_dlbfw_16L_F14_64TR_6cell.json", ["DLBFW"], ["PUSCH1", "PUSCH2", "SRS1", "ULBFW1"]),
        ("1612_srs_mmse_F14_64TR_6cell.json", ["SRS1"], ["PUSCH1", "PUSCH2", "PDSCH", "DLBFW", "ULBFW1"]),
        ("1612_pdsch_run2_F14_64TR_6cell.json", ["PDSCH"], ["PUSCH1", "PUSCH2", "DLBFW", "SRS1", "ULBFW1"]),
        ("1612_dlbfw_32L_F14_64TR_6cell.json", ["DLBFW"], ["PUSCH1", "PUSCH2", "SRS1", "ULBFW1"]),
        ("1612_srs_rkhs_F14_64TR_6cell.json", ["SRS1"], ["PUSCH1", "PUSCH2", "PDSCH", "DLBFW", "ULBFW1"]),
    ]

optional = [
    ("verify_srs_target_82.json", ["SRS1"], []),
    ("verify_srs_target_16.json", ["SRS1"], []),
]

failures = 0

def check_file(name, expect, forbid):
    global failures
    path = perf_dir / name
    if not path.is_file():
        print(f"[verify] FAIL: missing {path}")
        failures += 1
        return
    with path.open(encoding="utf-8") as f:
        root = json.load(f)
    if cell_key not in root:
        print(f"[verify] FAIL: {name} missing cell key {cell_key!r}")
        failures += 1
        return
    d = root[cell_key]
    row = [name]
    for ch in expect:
        vals = d.get(ch) or []
        if not vals:
            print(f"[verify] FAIL: {name} expected {ch} with data, got empty")
            failures += 1
            return
        row.append(f"{ch}={statistics.mean(vals):.0f}us")
    for ch in forbid:
        vals = d.get(ch) or []
        if vals:
            print(f"[verify] FAIL: {name} leaked channel {ch} (mean={statistics.mean(vals):.0f}us)")
            failures += 1
            return
    print("[verify] OK:", " | ".join(row))

for case in cases:
    check_file(*case)

if (perf_dir / "verify_srs_target_82.json").is_file():
    for case in optional:
        check_file(*case)

sys.exit(1 if failures else 0)
PY
}

main() {
  preflight
  if [[ "${FAILURES}" -gt 0 ]]; then
    echo "[verify] aborting (${FAILURES} preflight failure(s))"
    exit 1
  fi

  run_matrix
  if ! run_target_sm_check; then
    fail "TARGET_SM verification failed"
  fi
  if ! validate_json; then
    fail "JSON validation failed"
  fi

  if [[ "${FAILURES}" -gt 0 ]]; then
    echo ""
    echo "[verify] ${FAILURES} check(s) failed"
    exit 1
  fi

  echo ""
  echo "[verify] All channel_only checks passed (mode=${MODE})"
}

main
