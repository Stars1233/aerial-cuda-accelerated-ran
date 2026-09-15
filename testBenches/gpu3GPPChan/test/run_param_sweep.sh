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

# gpu3gppchan parameter-sweep smoke test.
#
# Expands test/sweep_params.yaml into one-parameter-at-a-time variants of
# the UMa 6 GHz baseline config and runs sls_chan_ex on each under
# compute-sanitizer (memcheck by default). PASS means every variant runs to
# completion with zero sanitizer errors. RMa memcheck variants leave the
# prohibitively expensive convolveCRNKernel uninstrumented, but the kernel still
# executes normally. No calibration/reference comparison is performed here —
# see run_cicd_calibration.sh for that.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SWEEP_SPEC="$SCRIPT_DIR/sweep_params.yaml"
OUTPUT_DIR="$PWD/gpu3gppchan_cicd_results/sweep"
TOOL="memcheck"    # memcheck | racecheck | initcheck | synccheck | none
KEEP_ARTIFACTS=false
SELECT_CASES=()    # empty => run every variant (unchanged default)
LIST_CASES=false
TIMEOUT_SECS="${TIMEOUT_SECS:-7200}"
SLS_CHAN_EX="${SLS_CHAN_EX:-$SDK_ROOT/build.aarch64/testBenches/gpu3GPPChan/examples/sls_chan/sls_chan_ex}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if ! PYTHON_BIN="$(command -v python3)"; then
        echo "ERROR: python3 not found; set PYTHON_BIN to an executable Python interpreter." >&2
        exit 1
    fi
fi

usage() {
    cat <<EOF
gpu3gppchan parameter sweep (compute-sanitizer smoke test)

Usage: $0 [options]
  --spec FILE        Sweep specification (default: test/sweep_params.yaml)
  --output-dir DIR   Results root (default: ./gpu3gppchan_cicd_results/sweep)
  --tool NAME        compute-sanitizer tool: memcheck (default), racecheck,
                     initcheck, synccheck, or 'none' for a plain run
  --timeout SECONDS  Per-variant timeout (default: $TIMEOUT_SECS)
  --keep-artifacts   Keep per-variant work dirs (H5 dumps etc.)
  --case NAME        Run only this variant (repeatable). Default: all variants.
  --list-cases       Print the expandable variant names and exit.
  --help             This message

The binary can be overridden via the SLS_CHAN_EX environment variable.
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
        --spec)          SWEEP_SPEC="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
        --tool)          TOOL="$2"; shift 2 ;;
        --timeout)       TIMEOUT_SECS="$2"; shift 2 ;;
        --keep-artifacts) KEEP_ARTIFACTS=true; shift ;;
        --case)          require_option_value "$@"; SELECT_CASES+=("$2"); shift 2 ;;
        --list-cases)    LIST_CASES=true; shift ;;
        --help|-h)       usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

case $TOOL in
    memcheck|racecheck|initcheck|synccheck|none) ;;
    *) echo "ERROR: invalid --tool '$TOOL'"; exit 1 ;;
esac

[[ "$TIMEOUT_SECS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: --timeout must be a positive integer (got '$TIMEOUT_SECS')"
    exit 1
}

command -v timeout >/dev/null || {
    echo "ERROR: timeout not in PATH"
    exit 1
}

[ -x "$PYTHON_BIN" ] || {
    echo "ERROR: Python interpreter not found at $PYTHON_BIN"
    echo "Set PYTHON_BIN to an executable Python interpreter."
    exit 1
}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

mkdir -p "$OUTPUT_DIR"
echo "============================================================"
echo "gpu3gppchan parameter sweep"
echo "  spec:    $SWEEP_SPEC"
echo "  binary:  $SLS_CHAN_EX"
echo "  tool:    $TOOL"
echo "  timeout: ${TIMEOUT_SECS}s per variant"
echo "  results: $OUTPUT_DIR"
echo "============================================================"

#=============================================================================
# Expand variants
#=============================================================================
MANIFEST="$OUTPUT_DIR/manifest.tsv"
"$PYTHON_BIN" "$SCRIPT_DIR/apply_sweep_params.py" \
    --spec "$SWEEP_SPEC" --out "$OUTPUT_DIR/configs" --sdk-root "$SDK_ROOT" \
    > "$MANIFEST"

N_VARIANTS=$(wc -l < "$MANIFEST")
echo "Expanded $N_VARIANTS variant config(s)"
[ "$N_VARIANTS" -gt 0 ] || { echo "ERROR: sweep spec produced no variants"; exit 1; }

#=============================================================================
# Optional per-case selection (opt-in; absent flags => run every variant)
#=============================================================================
if $LIST_CASES; then
    echo "Available sweep cases:"
    cut -f1 "$MANIFEST" | sed 's/^/  /'
    exit 0
fi

if [ "${#SELECT_CASES[@]}" -gt 0 ]; then
    all_names="$(cut -f1 "$MANIFEST")"
    missing=()
    for want in "${SELECT_CASES[@]}"; do
        grep -qxF "$want" <<<"$all_names" || missing+=("$want")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "ERROR: requested case(s) not in this sweep: ${missing[*]}" >&2
        echo "Run with --list-cases to see valid names." >&2
        exit 1
    fi
    selected="$OUTPUT_DIR/manifest.selected.tsv"
    : > "$selected"
    while IFS=$'\t' read -r name config desc; do
        for want in "${SELECT_CASES[@]}"; do
            [ "$name" = "$want" ] && {
                printf '%s\t%s\t%s\n' "$name" "$config" "$desc" >> "$selected"
                break
            }
        done
    done < "$MANIFEST"
    MANIFEST="$selected"
    N_VARIANTS=$(wc -l < "$MANIFEST")
    echo "Per-case selection: running $N_VARIANTS variant(s) — ${SELECT_CASES[*]}"
fi

#=============================================================================
# Runtime prerequisites (only needed once we are actually going to run)
#=============================================================================
[ -f "$SLS_CHAN_EX" ] || {
    echo "ERROR: sls_chan_ex not found at $SLS_CHAN_EX"
    echo "Build it with: cmake --build build.aarch64 -t sls_chan_ex"
    exit 1
}

if [ "$TOOL" != "none" ] && ! command -v compute-sanitizer >/dev/null; then
    echo "ERROR: compute-sanitizer not in PATH"
    exit 1
fi

#=============================================================================
# Run each variant
#=============================================================================
declare -a RESULTS
FAILURES=0

while IFS=$'\t' read -r name config desc; do
    run_dir="$OUTPUT_DIR/runs/$name"
    mkdir -p "$run_dir"
    log="$run_dir/run.log"

    n_site=$(awk '$1 == "n_site:" { print $2; exit }' "$config")
    [[ "$n_site" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: invalid or missing system_level.n_site in $config"
        exit 1
    }
    scenario=$(awk '$1 == "scenario:" { print $2; exit }' "$config")
    [ -n "$scenario" ] || {
        echo "ERROR: missing system_level.scenario in $config"
        exit 1
    }

    # CPU-only variants never make an instrumentable CUDA call, which
    # compute-sanitizer treats as an error. Large site-count variants are too
    # expensive under instrumentation, so sanitize only configurations with
    # at most three sites.
    variant_tool="$TOOL"
    tool_reason=""
    if grep -qE "^\s*cpu_only_mode:\s*1\s*(#.*)?$" "$config"; then
        variant_tool="none"
        tool_reason="CPU-only"
    elif [ "$n_site" -gt 3 ]; then
        variant_tool="none"
        tool_reason="n_site=$n_site > 3"
    fi

    cmd=("$SLS_CHAN_EX" "$config" -d "sweep_$name")
    excluded_kernel=""
    if [ "$variant_tool" != "none" ]; then
        sanitizer_args=(--tool "$variant_tool" --error-exitcode 99)
        if [ "$variant_tool" = "memcheck" ] && [ "$scenario" = "RMa" ]; then
            excluded_kernel="convolveCRNKernel"
            sanitizer_args+=(--kernel-name-exclude "kns=$excluded_kernel")
        fi
        cmd=(compute-sanitizer "${sanitizer_args[@]}" "${cmd[@]}")
    fi

    echo ""
    echo ">>> [$name] $desc"
    if [ -n "$tool_reason" ]; then
        echo "    tool: none ($tool_reason)"
    elif [ -n "$excluded_kernel" ]; then
        echo "    tool: $variant_tool ($excluded_kernel executes without instrumentation)"
    else
        echo "    tool: $variant_tool"
    fi
    rc=0
    (cd "$run_dir" && timeout --kill-after=30s "$TIMEOUT_SECS" "${cmd[@]}") \
        > "$log" 2>&1 || rc=$?

    verdict="PASS"
    if [ $rc -eq 124 ]; then
        verdict="TIMEOUT(${TIMEOUT_SECS}s)"
    elif [ $rc -ne 0 ]; then
        verdict="FAIL(exit=$rc)"
    elif [ "$variant_tool" != "none" ] && ! grep -q "ERROR SUMMARY: 0 errors" "$log"; then
        # Belt and suspenders: sanitizer must have printed a clean summary.
        verdict="FAIL(no-clean-summary)"
    fi

    if [ "$verdict" != "PASS" ]; then
        FAILURES=$((FAILURES + 1))
        echo "    $verdict — last log lines:"
        tail -5 "$log" | sed 's/^/    | /'
    else
        echo "    PASS"
        # Keep logs, drop bulky artifacts (H5 dumps) unless asked otherwise
        if ! $KEEP_ARTIFACTS; then
            find "$run_dir" -name "*.h5" -delete
        fi
    fi
    RESULTS+=("$verdict"$'\t'"$name"$'\t'"$desc")
done < "$MANIFEST"

#=============================================================================
# Summary
#=============================================================================
echo ""
echo "============================================================"
echo "SWEEP SUMMARY ($TOOL): $((N_VARIANTS - FAILURES))/$N_VARIANTS passed"
echo "============================================================"
if command -v column >/dev/null 2>&1; then
    printf '%s\n' "${RESULTS[@]}" | column -t -s $'\t'
else
    printf '%s\n' "${RESULTS[@]}"
fi
{
    echo "tool: $TOOL"
    printf '%s\n' "${RESULTS[@]}"
} > "$OUTPUT_DIR/summary.tsv"

if [ $FAILURES -gt 0 ]; then
    echo ""
    echo "SWEEP LEG: FAIL ($FAILURES failure(s)) — logs under $OUTPUT_DIR/runs/"
    exit 1
fi

echo ""
echo "SWEEP LEG: PASS"
