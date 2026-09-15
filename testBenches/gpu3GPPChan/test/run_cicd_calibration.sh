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

# Calibration regression test for gpu3gppchan.
#
# Runs the reduced-seed 3GPP calibration and gates on KS statistics:
#   - UMa  phase 1/2            seeds 0-4   (needs TR38901_REL18_PARAMS=ON build)
#   - ISAC phase 1/2 tgt+bkgnd  seeds 0-49  (needs default Rel-19 build)
#
# The parameter vintage is a COMPILE-TIME option, so this script maintains two
# builds: the regular build.aarch64 (Rel-19, must have TR38901_REL18_PARAMS=OFF)
# and a dedicated build.aarch64.gpu3gppchan_rel18 for the UMa legs.
#
# Reuses util/run_sls_chan_multiseed.sh via the SLS_CHAN_EX / ANALYSIS_EXTRA_ARGS
# environment overrides; KS statistics are captured in <leg>/analysis.log and
# checked against test/ks_thresholds.yaml by check_ks_gate.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
UTIL_DIR="$SDK_ROOT/testBenches/gpu3GPPChan/util"
CONFIG_DIR="$SDK_ROOT/testBenches/gpu3GPPChan/config"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if ! PYTHON_BIN="$(command -v python3)"; then
        echo "ERROR: python3 not found; set PYTHON_BIN to an executable Python interpreter." >&2
        exit 1
    fi
fi

REL19_BUILD="$SDK_ROOT/build.aarch64"
REL18_BUILD="$SDK_ROOT/build.aarch64.gpu3gppchan_rel18"
BIN_SUBPATH="testBenches/gpu3GPPChan/examples/sls_chan/sls_chan_ex"

OUTPUT_DIR="$PWD/gpu3gppchan_cicd_results/calibration"
THRESHOLDS="$SCRIPT_DIR/ks_thresholds.yaml"
RUN_UMA=true
RUN_ISAC=true
SKIP_BUILD=false
KEEP_H5=false
GATE_MODE="gate"   # gate | derive | none
UMA_SEED_START=0
UMA_SEED_END=4
ISAC_SEED_START=0
ISAC_SEED_END=49
SELECT_LEGS=()          # empty => run all legs (unchanged default)
LIST_LEGS=false
GATE_EXPLICIT=false     # true once the user picks a gate mode explicitly
readonly ALL_LEGS=(uma_phase1 uma_phase2 \
    isac_phase1_target isac_phase1_background \
    isac_phase2_target isac_phase2_background)

usage() {
    cat <<EOF
gpu3gppchan calibration test (UMa + ISAC, reduced seeds, KS gate)

Usage: $0 [options]
  --output-dir DIR       Results root (default: ./gpu3gppchan_cicd_results/calibration)
  --uma-only             Run only the UMa legs (REL18 build)
  --isac-only            Run only the ISAC legs (Rel-19 build)
  --leg NAME             Run only this leg (repeatable); overrides --uma/isac-only.
                         Skips the KS gate unless --no-gate/--derive-thresholds given.
  --list-legs            List calibration leg names and exit
  --uma-seed-end N       UMa last seed (default: $UMA_SEED_END)
  --isac-seed-end N      ISAC last seed (default: $ISAC_SEED_END)
  --skip-build           Do not (re)build; use existing binaries
  --keep-h5              Keep per-leg work dirs with H5 dumps (large!)
  --derive-thresholds    Write ks_thresholds.yaml from this run instead of gating
  --no-gate              Run calibrations only, skip the KS check entirely
  --help                 This message
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
            require_option_value "$@"
            OUTPUT_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"
            shift 2
            ;;
        --uma-only)        RUN_ISAC=false; shift ;;
        --isac-only)       RUN_UMA=false; shift ;;
        --leg)
            require_option_value "$@"
            SELECT_LEGS+=("$2")
            shift 2
            ;;
        --list-legs)       LIST_LEGS=true; shift ;;
        --uma-seed-end)
            require_option_value "$@"
            UMA_SEED_END="$2"
            shift 2
            ;;
        --isac-seed-end)
            require_option_value "$@"
            ISAC_SEED_END="$2"
            shift 2
            ;;
        --skip-build)      SKIP_BUILD=true; shift ;;
        --keep-h5)         KEEP_H5=true; shift ;;
        --derive-thresholds) GATE_MODE="derive"; GATE_EXPLICIT=true; shift ;;
        --no-gate)         GATE_MODE="none"; GATE_EXPLICIT=true; shift ;;
        --help|-h)         usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if ! $RUN_UMA && ! $RUN_ISAC; then
    echo "ERROR: --uma-only and --isac-only cannot be used together" >&2
    usage >&2
    exit 1
fi

#=============================================================================
# Optional per-leg selection (opt-in; absent flags => run all legs)
#=============================================================================
if $LIST_LEGS; then
    printf '%s\n' "${ALL_LEGS[@]}"
    exit 0
fi

if [ "${#SELECT_LEGS[@]}" -gt 0 ]; then
    # Validate every requested leg name against the known set.
    for want in "${SELECT_LEGS[@]}"; do
        found=false
        for known in "${ALL_LEGS[@]}"; do
            [ "$want" = "$known" ] && { found=true; break; }
        done
        $found || { echo "ERROR: unknown leg '$want' (see --list-legs)" >&2; exit 1; }
    done
    # Derive which builds are needed from the selection (overrides --uma/isac-only).
    RUN_UMA=false
    RUN_ISAC=false
    for want in "${SELECT_LEGS[@]}"; do
        case $want in
            uma_*) RUN_UMA=true ;;
            *)     RUN_ISAC=true ;;
        esac
    done
    # A subset run cannot satisfy the full-suite KS gate; skip it unless the
    # user explicitly chose a gate mode.
    if ! $GATE_EXPLICIT; then
        GATE_MODE="none"
        echo "Per-leg selection: KS gate skipped (pass --no-gate/--derive-thresholds to override)"
    fi
    echo "Per-leg selection: ${SELECT_LEGS[*]}"
fi

# True when <leg> should run: no selection => all legs; otherwise only if listed.
leg_selected() {
    [ "${#SELECT_LEGS[@]}" -eq 0 ] && return 0
    local l
    for l in "${SELECT_LEGS[@]}"; do
        [ "$1" = "$l" ] && return 0
    done
    return 1
}

mkdir -p "$OUTPUT_DIR"
echo "Results root: $OUTPUT_DIR"

[ -x "$PYTHON_BIN" ] || {
    echo "ERROR: Python interpreter not found at $PYTHON_BIN"
    echo "Set PYTHON_BIN to an executable Python interpreter."
    exit 1
}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

#=============================================================================
# Builds
#=============================================================================
check_flag() {  # <build_dir> <expected ON|OFF>
    grep -q "TR38901_REL18_PARAMS:BOOL=$2" "$1/CMakeCache.txt" || {
        echo "ERROR: $1 is not built with TR38901_REL18_PARAMS=$2"; exit 1; }
}

if $RUN_ISAC; then
    if [ ! -f "$REL19_BUILD/CMakeCache.txt" ]; then
        echo "ERROR: $REL19_BUILD not configured. Configure the regular SDK build first."
        exit 1
    fi
    check_flag "$REL19_BUILD" OFF
    if ! $SKIP_BUILD; then
        echo ">>> Building sls_chan_ex (Rel-19 default build)"
        cmake --build "$REL19_BUILD" -t sls_chan_ex
    fi
    [ -f "$REL19_BUILD/$BIN_SUBPATH" ] || { echo "ERROR: missing $REL19_BUILD/$BIN_SUBPATH"; exit 1; }
fi

if $RUN_UMA; then
    if [ ! -f "$REL18_BUILD/CMakeCache.txt" ]; then
        if $SKIP_BUILD; then
            echo "ERROR: $REL18_BUILD not configured. Re-run without --skip-build."
            exit 1
        fi
        echo ">>> Configuring REL18 build: $REL18_BUILD"
        cmake -S "$SDK_ROOT" -B "$REL18_BUILD" -G Ninja \
            -DCMAKE_TOOLCHAIN_FILE="$SDK_ROOT/cuPHY/cmake/toolchains/grace-cross" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES=native \
            -DTR38901_REL18_PARAMS=ON
    fi
    check_flag "$REL18_BUILD" ON
    if ! $SKIP_BUILD; then
        echo ">>> Building sls_chan_ex (TR38901_REL18_PARAMS=ON build)"
        cmake --build "$REL18_BUILD" -t sls_chan_ex
    fi
    [ -f "$REL18_BUILD/$BIN_SUBPATH" ] || { echo "ERROR: missing $REL18_BUILD/$BIN_SUBPATH"; exit 1; }
fi

#=============================================================================
# Calibration legs
#=============================================================================
run_leg() {  # <leg> <build_dir> <config> <dataset> <ref_json> <phase> <isac_channel|-> <seed_start> <seed_end>
    local leg=$1 build_dir=$2 config=$3 dataset=$4 ref_json=$5 phase=$6 isac_channel=$7
    local seed_start=$8 seed_end=$9
    local leg_dir="$OUTPUT_DIR/$leg"
    local work_dir="$OUTPUT_DIR/work/$leg"

    echo ""
    echo "========================================================"
    echo ">>> Leg: $leg (seeds $seed_start-$seed_end)"
    echo "========================================================"
    mkdir -p "$leg_dir" "$work_dir"

    # Work on a private copy of the YAML so the tracked config/ dir is never
    # mutated (run_sls_chan_multiseed.sh seds the seed into the file in place).
    local config_copy="$work_dir/$(basename "$config")"
    cp "$CONFIG_DIR/$config" "$config_copy"

    local isac_args=()
    [ "$isac_channel" != "-" ] && isac_args=(--isac-channel "$isac_channel")

    local leg_start
    leg_start=$(date +%s)
    local leg_rc=0
    (
        cd "$work_dir"
        SLS_CHAN_EX="$build_dir/$BIN_SUBPATH" \
        ANALYSIS_EXTRA_ARGS="--log-level INFO --log-file $leg_dir/analysis.log" \
        "$UTIL_DIR/run_sls_chan_multiseed.sh" \
            --seed-start "$seed_start" \
            --seed-end "$seed_end" \
            --config "$config_copy" \
            --dataset "$dataset" \
            --reference-json "testBenches/gpu3GPPChan/util/$ref_json" \
            --phase "$phase" \
            ${isac_args[@]+"${isac_args[@]}"} \
            --output-dir "$leg_dir"
    ) 2>&1 | tee "$leg_dir/run.log" || leg_rc=$?

    local leg_elapsed=$(( $(date +%s) - leg_start ))
    local leg_verdict="PASS"
    [ $leg_rc -eq 0 ] || leg_verdict="FAIL (exit $leg_rc)"
    echo "<<< Leg: $leg $leg_verdict in $((leg_elapsed / 60))m $((leg_elapsed % 60))s"

    if ! $KEEP_H5; then
        rm -rf "$work_dir"
    fi

    return $leg_rc
}

START_TIME=$(date +%s)

if $RUN_UMA; then
    leg_selected uma_phase1 && \
    run_leg uma_phase1 "$REL18_BUILD" statistic_channel_config_phase1.yaml \
        uma_6ghz 3gpp_calibration_phase1.json 1 - "$UMA_SEED_START" "$UMA_SEED_END"
    leg_selected uma_phase2 && \
    run_leg uma_phase2 "$REL18_BUILD" statistic_channel_config_phase2.yaml \
        uma_6ghz 3gpp_calibration_phase2.json 2 - "$UMA_SEED_START" "$UMA_SEED_END"
fi

if $RUN_ISAC; then
    leg_selected isac_phase1_target && \
    run_leg isac_phase1_target "$REL19_BUILD" statistic_channel_config_isac_phase1.yaml \
        isac_uav_phase1_target 3gpp_calibration_isac_uav_phase1.json 1 target \
        "$ISAC_SEED_START" "$ISAC_SEED_END"
    leg_selected isac_phase1_background && \
    run_leg isac_phase1_background "$REL19_BUILD" statistic_channel_config_isac_phase1_background.yaml \
        isac_uav_phase1_background 3gpp_calibration_isac_uav_phase1.json 1 background \
        "$ISAC_SEED_START" "$ISAC_SEED_END"
    leg_selected isac_phase2_target && \
    run_leg isac_phase2_target "$REL19_BUILD" statistic_channel_config_isac_phase2.yaml \
        isac_uav_phase2_target 3gpp_calibration_isac_uav_phase2.json 2 target \
        "$ISAC_SEED_START" "$ISAC_SEED_END"
    leg_selected isac_phase2_background && \
    run_leg isac_phase2_background "$REL19_BUILD" statistic_channel_config_isac_phase2_background.yaml \
        isac_uav_phase2_background 3gpp_calibration_isac_uav_phase2.json 2 background \
        "$ISAC_SEED_START" "$ISAC_SEED_END"
fi

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "All calibration legs finished in $((ELAPSED / 60))m $((ELAPSED % 60))s"

#=============================================================================
# KS gate
#=============================================================================
case $GATE_MODE in
    derive)
        "$PYTHON_BIN" "$SCRIPT_DIR/check_ks_gate.py" \
            --results "$OUTPUT_DIR" --thresholds "$THRESHOLDS" --derive
        ;;
    gate)
        echo ""
        echo ">>> KS gate"
        "$PYTHON_BIN" "$SCRIPT_DIR/check_ks_gate.py" \
            --results "$OUTPUT_DIR" --thresholds "$THRESHOLDS"
        ;;
    none)
        echo "KS gate skipped (--no-gate)"
        ;;
esac

echo ""
echo "CALIBRATION LEG: PASS"
