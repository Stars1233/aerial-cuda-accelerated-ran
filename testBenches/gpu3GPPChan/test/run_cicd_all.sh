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

# Entry point for the gpu3gppchan regression tests.
#
# Runs all legs and aggregates the result into a single exit code:
#   1. Calibration leg: run_cicd_calibration.sh (UMa + ISAC, KS gate)
#   2. Sweep leg:       run_param_sweep.sh (compute-sanitizer smoke test)
#   3. Python leg:      run_python_tests.sh
#
# This is the script a future CICD job should invoke.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="$PWD/gpu3gppchan_cicd_results"
EXTRA_CAL_ARGS=()
EXTRA_SWEEP_ARGS=()
LIST_CASES=false
LIST_LEGS=false
CAL_SELECTED=false     # true once --leg is passed
SWEEP_SELECTED=false   # true once --case is passed

usage() {
    cat <<EOF
gpu3gppchan test: calibration (KS gate) + parameter sweep + Python tests

Usage: $0 [options]
  --output-dir DIR   Results root (default: ./gpu3gppchan_cicd_results)
  --skip-build       Passed through to the calibration leg
  --case NAME        Run only this sweep variant (repeatable); runs the sweep leg only
  --leg NAME         Run only this calibration leg (repeatable); runs the calibration leg only
  --list-cases       List sweep variant names and exit
  --list-legs        List calibration leg names and exit
  --help             This message

With neither --case nor --leg, all three legs run in full.
Selecting only --case runs just the sweep leg; only --leg runs just the
calibration leg; passing both runs both, each filtered to its selection.
EOF
}

require_option_value() {
    if [[ $# -lt 2 ]] || [[ $2 == -* ]]; then
        echo "ERROR: $1 requires a value" >&2
        usage >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            if ! mkdir -p "$2"; then
                echo "ERROR: unable to create output directory: $2" >&2
                exit 1
            fi
            if ! RESULTS_ROOT="$(cd "$2" && pwd)"; then
                echo "ERROR: unable to resolve output directory: $2" >&2
                exit 1
            fi
            shift 2
            ;;
        --skip-build) EXTRA_CAL_ARGS+=(--skip-build); shift ;;
        --case)       require_option_value "$@"; EXTRA_SWEEP_ARGS+=(--case "$2"); SWEEP_SELECTED=true; shift 2 ;;
        --leg)        require_option_value "$@"; EXTRA_CAL_ARGS+=(--leg "$2"); CAL_SELECTED=true; shift 2 ;;
        --list-cases) LIST_CASES=true; shift ;;
        --list-legs)  LIST_LEGS=true; shift ;;
        --help|-h)    usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup env: use the container system Python by default and verify that its
# required analysis dependencies are available. Use PYTHON_BIN from the
# environment when provided.
# ---------------------------------------------------------------------------
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if ! PYTHON_BIN="$(command -v python3)"; then
        echo "ERROR: python3 not found; set PYTHON_BIN to an executable Python interpreter." >&2
        exit 1
    fi
fi

check_python_dependencies() {
    if [ -x "$PYTHON_BIN" ] && \
       "$PYTHON_BIN" -c 'import matplotlib,scipy,h5py,numpy,yaml' 2>/dev/null; then
        return 0
    fi
    echo "ERROR: Python interpreter is missing required modules: $PYTHON_BIN" >&2
    echo "Use an Aerial container with the system Python requirements installed," >&2
    echo "or set PYTHON_BIN to an equivalent interpreter." >&2
    return 1
}

if ! check_python_dependencies; then
    echo "ERROR: Python dependency check failed; aborting" >&2
    exit 1
fi
export PYTHON_BIN

# Listing is a query, not a run: forward to the relevant leg(s) and exit.
if $LIST_LEGS || $LIST_CASES; then
    $LIST_LEGS  && "$SCRIPT_DIR/run_cicd_calibration.sh" --list-legs
    $LIST_CASES && "$SCRIPT_DIR/run_param_sweep.sh" --list-cases
    exit 0
fi

# Selecting cases in one leg implies you only want that leg; with no selection
# both legs run (unchanged default).
RUN_CAL=true
RUN_SWEEP=true
RUN_PYTHON=true
if $SWEEP_SELECTED && ! $CAL_SELECTED; then RUN_CAL=false; fi
if $CAL_SELECTED && ! $SWEEP_SELECTED; then RUN_SWEEP=false; fi
if $SWEEP_SELECTED || $CAL_SELECTED; then RUN_PYTHON=false; fi

CAL_RC=0
if $RUN_CAL; then
    "$SCRIPT_DIR/run_cicd_calibration.sh" \
        --output-dir "$RESULTS_ROOT/calibration" \
        ${EXTRA_CAL_ARGS[@]+"${EXTRA_CAL_ARGS[@]}"} || CAL_RC=$?
fi

SWEEP_RC=0
if $RUN_SWEEP; then
    "$SCRIPT_DIR/run_param_sweep.sh" \
        --output-dir "$RESULTS_ROOT/sweep" \
        ${EXTRA_SWEEP_ARGS[@]+"${EXTRA_SWEEP_ARGS[@]}"} || SWEEP_RC=$?
fi

PYTHON_RC=0
if $RUN_PYTHON; then
    # The package test owns its local venv. Clear the system-Python selection
    # used by the analysis legs so it can configure, build, and install the
    # wheel with the appropriate GPU3GPP preset.
    env -u PYTHON_BIN "$SCRIPT_DIR/run_python_tests.sh" || PYTHON_RC=$?
fi

echo ""
echo "============================================================"
echo "gpu3gppchan CICD SUMMARY"
echo "============================================================"
if $RUN_CAL; then
    [ $CAL_RC -eq 0 ] && echo "  calibration : PASS" || echo "  calibration : FAIL (exit $CAL_RC)"
else
    echo "  calibration : SKIP (per-case selection)"
fi
if $RUN_SWEEP; then
    [ $SWEEP_RC -eq 0 ] && echo "  sweep       : PASS" || echo "  sweep       : FAIL (exit $SWEEP_RC)"
else
    echo "  sweep       : SKIP (per-leg selection)"
fi
if $RUN_PYTHON; then
    [ $PYTHON_RC -eq 0 ] && echo "  python      : PASS" || echo "  python      : FAIL (exit $PYTHON_RC)"
else
    echo "  python      : SKIP (per-case selection)"
fi
echo "  results     : $RESULTS_ROOT"
echo "============================================================"

if [ $CAL_RC -ne 0 ] || [ $SWEEP_RC -ne 0 ] || [ $PYTHON_RC -ne 0 ]; then
    exit 1
fi
echo "gpu3gppchan CICD: PASS"
