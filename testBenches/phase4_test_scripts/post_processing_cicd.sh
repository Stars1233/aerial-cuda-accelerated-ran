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

#-----------------------------------------------------------------------------------
# post_processing_cicd.sh - CICD wrapper for full post-processing sequence
#
# This script orchestrates the full CICD post-processing sequence by calling
# post_processing_parse.sh and post_processing_analyze.sh in the appropriate order.
#
# Return codes:
#   0 = All steps passed
#   1 = One or more steps failed (all steps run to completion before returning)
#   2 = All steps passed, but warning threshold exceeded (for Slack notification)
#-----------------------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}

MMIMO_FLAG=""
FRAMEWORK_CPLANE_FLAG=""
LABEL=""
IGNORE_UL_CHANNELS=()
IGNORE_DL_CHANNELS=()

# Timing tracking arrays
declare -a STEP_NAMES
declare -a STEP_START_TIMES
declare -a STEP_END_TIMES
declare -a STEP_DURATIONS

# Get current timestamp in seconds with milliseconds
get_timestamp() {
    date +%s.%3N
}

# Format timestamp for display
format_timestamp() {
    date -d "@$1" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -r "${1%.*}" "+%Y-%m-%d %H:%M:%S"
}

# Calculate duration between two timestamps and format as human-readable
calc_duration() {
    local start=$1
    local end=$2
    local duration=$(echo "$end - $start" | bc)
    local minutes=$(echo "$duration / 60" | bc)
    local seconds=$(echo "$duration - ($minutes * 60)" | bc)
    printf "%dm %.1fs" "$minutes" "$seconds"
}

# Record step timing and print result
record_step_timing() {
    local step_name="$1"
    local start_time="$2"
    local end_time="$3"
    
    local duration=$(calc_duration "$start_time" "$end_time")
    local start_fmt=$(format_timestamp "$start_time")
    local end_fmt=$(format_timestamp "$end_time")
    
    # Store for summary
    STEP_NAMES+=("$step_name")
    STEP_START_TIMES+=("$start_time")
    STEP_END_TIMES+=("$end_time")
    STEP_DURATIONS+=("$duration")
    
    echo "    Timing: $start_fmt -> $end_fmt ($duration)"
}

# Required/optional threshold files
GATING_THRESHOLD=""
WARNING_THRESHOLD=""
ABSOLUTE_THRESHOLD=""
LATENCY_SUMMARY_ENABLED=0
SKIP_LOG_ANALYSIS=0
PERF_TRACE_DIR=""
BPFTRACE_DIR=""
FUTEX_CUDA_THRESHOLD=""
FUTEX_CUDA_MIN_CALLS=10
# Maximum acceptable total page-fault count for the page-fault gate (Step 11c).
# Defaults to 0: a pinned/pre-faulted real-time pipeline should take no page
# faults during the steady-state trace window, so any fault fails the test.
# Overridable with --page-fault-threshold.
PAGE_FAULT_THRESHOLD=0
PAGE_FAULT_THRESHOLD_SET=0

show_usage() {
    echo "Usage: $0 <phy_log> <testmac_log> <ru_log> <output_folder> [options]"
    echo
    echo "CICD wrapper that runs the full post-processing sequence with proper return codes."
    echo "Steps 1-4 (parse, metrics, visualizations) run unless --skip-log-analysis is set."
    echo "Threshold checks run if their files are provided."
    echo "Perf trace analysis runs if --perf-trace-dir is provided."
    echo
    echo "Positional Arguments (4 required):"
    echo "  phy_log                Path to phy.log file"
    echo "  testmac_log            Path to testmac.log file"
    echo "  ru_log                 Path to ru.log file (can be blank placeholder)"
    echo "  output_folder          Directory for output files"
    echo
    echo "Threshold Arguments (all optional, gating is mandatory unless --perf-trace-dir is set):"
    echo "  --gating-threshold <file>    Gating perf_requirements file"
    echo "  --warning-threshold <file>   Warning perf_requirements file"
    echo "  --absolute-threshold <file>  Absolute perf_requirements file"
    echo "  --latency-summary            Also run latency summary (NICD)"
    echo "  --mmimo                      Enable mMIMO mode"
    echo "  --framework-cplane           Framework C-plane mode (soft-skip metrics arch. absent on framework path)"
    echo "  -c, --ignore-ul-channels <names...>  UL channels to ignore (e.g. PUCCH PRACH SRS PUSCH)"
    echo "  -d, --ignore-dl-channels <names...>  DL channels to ignore (e.g. PDCCH CSIRS PBCH)"
    echo "  --label <name>               Label for compare_logs output"
    echo "  --skip-log-analysis          Skip steps 1-4 (log parsing, metrics, visualizations)"
    echo "  --perf-trace-dir <dir>       Directory containing perf record data files. When set, runs"
    echo "                               futex/CUDA analysis steps after the normal pipeline."
    echo "  --futex-cuda-threshold <pct> Pass/fail gate on futex/CUDA ratios (per-API and aggregate)."
    echo "                               Requires --perf-trace-dir and a cuda_api_tracer.log in it."
    echo "  --futex-cuda-min-calls <N>   Minimum tracer-log total for an API to be gated (default: 10)."
    echo "  --page-fault-threshold <N>   Max total page faults (user+kernel) allowed before failing."
    echo "                               Requires --perf-trace-dir; runs whenever it is set (default: 0)."
    echo "  --bpftrace-dir <dir>         Directory containing the bpftrace all-syscall capture. When set,"
    echo "                               runs the bpftrace gate: FAIL if the capture is empty (no syscalls)."
    echo "  -h, --help                   Show this help message"
    echo
    echo "Timing defaults are mode-dependent, configured in post_processing_defaults.cfg."
    echo "post_processing_analyze.sh applies them automatically per mode."
    echo
    echo "Sequence Executed:"
    echo "  1-4. Log parsing and visualization (skipped if --skip-log-analysis)"
    echo "  5-8. Threshold checks (skipped if no threshold files provided)"
    echo "  9-10. Latency analysis (if --latency-summary enabled)"
    echo "  11-12. Perf trace analysis (if --perf-trace-dir provided)"
    echo
    echo "Return Codes:"
    echo "  0 = All steps passed"
    echo "  1 = One or more steps failed (all steps run to completion before returning)"
    echo "  2 = All steps passed, but warning threshold exceeded"
    echo
    echo "Example (normal):"
    echo "  $0 phy.log testmac.log ru.log ./output \\"
    echo "      --gating-threshold /path/to/gating_perf_requirements.csv \\"
    echo "      --warning-threshold /path/to/warning_perf_requirements.csv \\"
    echo "      --absolute-threshold /path/to/perf_requirements.csv \\"
    echo "      --mmimo --label my_test"
    echo
    echo "Example (PUSCH-only test case, ignore inactive channels):"
    echo "  $0 phy.log testmac.log ru.log ./output \\"
    echo "      --gating-threshold /path/to/gating.csv \\"
    echo "      -c PUCCH -d PDCCH CSIRS PBCH --label my_test"
    echo
    echo "Example (perf trace analysis, skip log processing):"
    echo "  $0 phy.log testmac.log ru.log ./output \\"
    echo "      --skip-log-analysis --perf-trace-dir /opt/nvidia/cuBB/perf"
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --gating-threshold)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing threshold file for --gating-threshold option"
                exit 1
            fi
            GATING_THRESHOLD="$2"
            shift 2
            ;;
        --warning-threshold)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing threshold file for --warning-threshold option"
                exit 1
            fi
            WARNING_THRESHOLD="$2"
            shift 2
            ;;
        --absolute-threshold)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing threshold file for --absolute-threshold option"
                exit 1
            fi
            ABSOLUTE_THRESHOLD="$2"
            shift 2
            ;;
        --latency-summary)
            LATENCY_SUMMARY_ENABLED=1
            shift
            ;;
        --mmimo)
            MMIMO_FLAG="--mmimo"
            shift
            ;;
        --framework-cplane)
            FRAMEWORK_CPLANE_FLAG="--framework-cplane"
            shift
            ;;
        --label)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --label option"
                exit 1
            fi
            LABEL="$2"
            shift 2
            ;;
        -c|--ignore-ul-channels)
            shift
            IGNORE_UL_CHANNELS=()
            if [[ $# -eq 0 || "$1" == -* ]]; then
                echo "Error: -c / --ignore-ul-channels requires at least one channel name (e.g. PUCCH PRACH)"
                exit 1
            fi
            while [[ $# -gt 0 && "$1" != -* ]]; do
                IGNORE_UL_CHANNELS+=("$1")
                shift
            done
            ;;
        -d|--ignore-dl-channels)
            shift
            IGNORE_DL_CHANNELS=()
            if [[ $# -eq 0 || "$1" == -* ]]; then
                echo "Error: -d / --ignore-dl-channels requires at least one channel name (e.g. PDCCH CSIRS PBCH)"
                exit 1
            fi
            while [[ $# -gt 0 && "$1" != -* ]]; do
                IGNORE_DL_CHANNELS+=("$1")
                shift
            done
            ;;
        --skip-log-analysis)
            SKIP_LOG_ANALYSIS=1
            shift
            ;;
        --perf-trace-dir)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --perf-trace-dir option"
                exit 1
            fi
            PERF_TRACE_DIR="$2"
            shift 2
            ;;
        --bpftrace-dir)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --bpftrace-dir option"
                exit 1
            fi
            BPFTRACE_DIR="$2"
            shift 2
            ;;
        --futex-cuda-threshold)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --futex-cuda-threshold option"
                exit 1
            fi
            FUTEX_CUDA_THRESHOLD="$2"
            shift 2
            ;;
        --futex-cuda-min-calls)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --futex-cuda-min-calls option"
                exit 1
            fi
            FUTEX_CUDA_MIN_CALLS="$2"
            shift 2
            ;;
        --page-fault-threshold)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --page-fault-threshold option"
                exit 1
            fi
            PAGE_FAULT_THRESHOLD="$2"
            PAGE_FAULT_THRESHOLD_SET=1
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# If no arguments provided, exit cleanly (supports empty params case)
if [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]]; then
    echo "No arguments provided. Use --help for usage information."
    exit 0
fi

# Validate positional arguments
if [[ ${#POSITIONAL_ARGS[@]} -ne 4 ]]; then
    echo "Error: Expected 4 positional arguments (phy_log, testmac_log, ru_log, output_folder)"
    echo "Got ${#POSITIONAL_ARGS[@]} arguments: ${POSITIONAL_ARGS[*]}"
    show_usage
    exit 1
fi

PHY_LOG="${POSITIONAL_ARGS[0]}"
TESTMAC_LOG="${POSITIONAL_ARGS[1]}"
RU_LOG="${POSITIONAL_ARGS[2]}"
OUTPUT_FOLDER="${POSITIONAL_ARGS[3]}"

# Ensure the output folder exists. In --skip-log-analysis (perf-trace) mode,
# Step 1 (post_processing_parse.sh) is skipped, and it is otherwise the only
# step that creates this directory. Without it, Step 11 fails to write the
# futex summary and the tracer-log copy fails with "No such file or directory".
mkdir -p "$OUTPUT_FOLDER"

# Validate threshold files exist if provided
if [[ -n "$GATING_THRESHOLD" && ! -f "$GATING_THRESHOLD" ]]; then
    echo "Error: Gating threshold file not found: $GATING_THRESHOLD"
    exit 1
fi

if [[ -n "$WARNING_THRESHOLD" && ! -f "$WARNING_THRESHOLD" ]]; then
    echo "Error: Warning threshold file not found: $WARNING_THRESHOLD"
    exit 1
fi

if [[ -n "$ABSOLUTE_THRESHOLD" && ! -f "$ABSOLUTE_THRESHOLD" ]]; then
    echo "Error: Absolute threshold file not found: $ABSOLUTE_THRESHOLD"
    exit 1
fi

if [[ -n "$FUTEX_CUDA_THRESHOLD" && -z "$PERF_TRACE_DIR" ]]; then
    echo "Error: --futex-cuda-threshold requires --perf-trace-dir"
    exit 1
fi

if [[ $PAGE_FAULT_THRESHOLD_SET -eq 1 && -z "$PERF_TRACE_DIR" ]]; then
    echo "Error: --page-fault-threshold requires --perf-trace-dir"
    exit 1
fi

if ! [[ "$PAGE_FAULT_THRESHOLD" =~ ^[0-9]+$ ]]; then
    echo "Error: --page-fault-threshold must be a non-negative integer (got: $PAGE_FAULT_THRESHOLD)"
    exit 1
fi

# Script paths
PARSE_LOGS="$SCRIPT_DIR/post_processing_parse.sh"
POST_PROCESSING="$SCRIPT_DIR/post_processing_analyze.sh"

LABEL_OPT=""
if [[ -n "$LABEL" ]]; then
    LABEL_OPT="--label $LABEL"
fi

OVERALL_RESULT=0
WARNING_THRESHOLD_FAILED=0
PIPELINE_START=$(get_timestamp)

echo "=============================================================="
echo "CICD Post-Processing Pipeline"
echo "=============================================================="
echo "PHY log:           $PHY_LOG"
echo "testMAC log:       $TESTMAC_LOG"
echo "RU log:            $RU_LOG"
echo "Output folder:     $OUTPUT_FOLDER"
echo "Gating threshold:  $GATING_THRESHOLD"
if [[ -n "$WARNING_THRESHOLD" ]]; then
    echo "Warning threshold: $WARNING_THRESHOLD"
fi
if [[ -n "$ABSOLUTE_THRESHOLD" ]]; then
    echo "Absolute threshold: $ABSOLUTE_THRESHOLD"
fi
if [[ $LATENCY_SUMMARY_ENABLED -eq 1 ]]; then
    echo "Latency summary:   enabled"
fi
if [[ -n "$PERF_TRACE_DIR" ]]; then
    echo "Perf trace dir:    $PERF_TRACE_DIR"
fi
if [[ -n "$FUTEX_CUDA_THRESHOLD" ]]; then
    echo "Futex/CUDA gate:   ${FUTEX_CUDA_THRESHOLD}% (min calls: ${FUTEX_CUDA_MIN_CALLS})"
fi
if [[ ${#IGNORE_UL_CHANNELS[@]} -gt 0 ]]; then
    echo "Ignore UL channels: ${IGNORE_UL_CHANNELS[*]}"
fi
if [[ ${#IGNORE_DL_CHANNELS[@]} -gt 0 ]]; then
    echo "Ignore DL channels: ${IGNORE_DL_CHANNELS[*]}"
fi
if [[ $SKIP_LOG_ANALYSIS -eq 1 ]]; then
    echo "Log analysis:      SKIPPED"
fi
echo "=============================================================="
echo

if [[ ! -f "$POST_PROCESSING" ]]; then
    echo "Error: post_processing_analyze.sh not found at $POST_PROCESSING"
    exit 1
fi

if [[ $SKIP_LOG_ANALYSIS -eq 0 ]]; then

if [[ ! -f "$PARSE_LOGS" ]]; then
    echo "Error: post_processing_parse.sh not found at $PARSE_LOGS"
    exit 1
fi

# Step 1: Parse logs
echo ">>> Step 1: Parsing logs..."
STEP1_START=$(get_timestamp)
PARSE_OPTS="--perf-metrics"
if [[ $LATENCY_SUMMARY_ENABLED -eq 1 ]]; then
    PARSE_OPTS="$PARSE_OPTS --latency-summary"
fi

"$PARSE_LOGS" "$PHY_LOG" "$TESTMAC_LOG" "$RU_LOG" "$OUTPUT_FOLDER" \
    $PARSE_OPTS \
    $MMIMO_FLAG

PARSE_RESULT=$?
STEP1_END=$(get_timestamp)
if [[ $PARSE_RESULT -ne 0 ]]; then
    echo ">>> Step 1: FAILED"
    OVERALL_RESULT=1
else
    echo ">>> Step 1: COMPLETE"
fi
record_step_timing "Parse logs" "$STEP1_START" "$STEP1_END"
echo

# Step 2: Performance metrics extraction
echo ">>> Step 2: Extracting performance metrics..."
STEP2_START=$(get_timestamp)
IGNORE_UL_ARGS=()
if [[ ${#IGNORE_UL_CHANNELS[@]} -gt 0 ]]; then
    IGNORE_UL_ARGS=(-c "${IGNORE_UL_CHANNELS[@]}")
fi
IGNORE_DL_ARGS=()
if [[ ${#IGNORE_DL_CHANNELS[@]} -gt 0 ]]; then
    IGNORE_DL_ARGS=(-d "${IGNORE_DL_CHANNELS[@]}")
fi
"$POST_PROCESSING" "$OUTPUT_FOLDER/binary" "$OUTPUT_FOLDER" \
    --perf-metrics \
    $MMIMO_FLAG \
    $FRAMEWORK_CPLANE_FLAG \
    "${IGNORE_UL_ARGS[@]}" \
    "${IGNORE_DL_ARGS[@]}"

PERF_RESULT=$?
STEP2_END=$(get_timestamp)
if [[ $PERF_RESULT -ne 0 ]]; then
    echo ">>> Step 2: FAILED"
    OVERALL_RESULT=1
else
    echo ">>> Step 2: COMPLETE"
fi
record_step_timing "Performance metrics" "$STEP2_START" "$STEP2_END"
echo

# Step 3: Compare logs visualization
echo ">>> Step 3: Generating compare_logs.html..."
STEP3_START=$(get_timestamp)
"$POST_PROCESSING" "$OUTPUT_FOLDER/binary" "$OUTPUT_FOLDER" \
    --compare-logs \
    $LABEL_OPT \
    $MMIMO_FLAG

COMPARE_RESULT=$?
STEP3_END=$(get_timestamp)
if [[ $COMPARE_RESULT -ne 0 ]]; then
    echo ">>> Step 3: FAILED"
    OVERALL_RESULT=1
else
    echo ">>> Step 3: COMPLETE"
fi
record_step_timing "Compare logs" "$STEP3_START" "$STEP3_END"
echo

# Step 4: CPU timeline visualization
echo ">>> Step 4: Generating cpu_timeline.html..."
STEP4_START=$(get_timestamp)
"$POST_PROCESSING" "$OUTPUT_FOLDER/binary" "$OUTPUT_FOLDER" \
    --cpu-timeline \
    $LABEL_OPT

CPU_TIMELINE_RESULT=$?
STEP4_END=$(get_timestamp)
if [[ $CPU_TIMELINE_RESULT -ne 0 ]]; then
    echo ">>> Step 4: FAILED"
    OVERALL_RESULT=1
else
    echo ">>> Step 4: COMPLETE"
fi
record_step_timing "CPU timeline" "$STEP4_START" "$STEP4_END"
echo

fi # SKIP_LOG_ANALYSIS

# Step 5: Threshold summary (informational, does not affect exit code)
SUMMARY_FILES=()
SUMMARY_LABELS=()
if [[ -n "$ABSOLUTE_THRESHOLD" ]]; then
    SUMMARY_FILES+=("$ABSOLUTE_THRESHOLD")
    SUMMARY_LABELS+=("absolute")
fi
if [[ -n "$GATING_THRESHOLD" ]]; then
    SUMMARY_FILES+=("$GATING_THRESHOLD")
    SUMMARY_LABELS+=("gating")
fi
if [[ -n "$WARNING_THRESHOLD" ]]; then
    SUMMARY_FILES+=("$WARNING_THRESHOLD")
    SUMMARY_LABELS+=("warning")
fi

if [[ ${#SUMMARY_FILES[@]} -gt 0 ]]; then
    echo ">>> Step 5: Generating threshold summary..."
    STEP5_START=$(get_timestamp)

    "$POST_PROCESSING" "$OUTPUT_FOLDER/binary" "$OUTPUT_FOLDER" \
        --threshold-summary "${SUMMARY_FILES[@]}" \
        -l "${SUMMARY_LABELS[@]}" \
        -o "$OUTPUT_FOLDER/threshold_summary.csv"

    STEP5_END=$(get_timestamp)
    echo ">>> Step 5: COMPLETE"
    record_step_timing "Threshold summary" "$STEP5_START" "$STEP5_END"
    echo
else
    echo ">>> Step 5: SKIPPED (no threshold files provided)"
    echo
fi

# Step 6: Absolute threshold check (optional)
if [[ -n "$ABSOLUTE_THRESHOLD" ]]; then
    echo ">>> Step 6: Running absolute threshold check..."
    STEP6_START=$(get_timestamp)
    "$POST_PROCESSING" "$OUTPUT_FOLDER/binary" "$OUTPUT_FOLDER" \
        --absolute-threshold "$ABSOLUTE_THRESHOLD"

    ABSOLUTE_RESULT=$?
    STEP6_END=$(get_timestamp)
    if [[ $ABSOLUTE_RESULT -ne 0 ]]; then
        echo ">>> Step 6: ABSOLUTE THRESHOLD FAILED"
        OVERALL_RESULT=1
    else
        echo ">>> Step 6: ABSOLUTE THRESHOLD PASSED"
    fi
    record_step_timing "Absolute threshold" "$STEP6_START" "$STEP6_END"
    echo
fi

# Step 7: Gating threshold check (mandatory for pass)
if [[ -n "$GATING_THRESHOLD" ]]; then
    echo ">>> Step 7: Running gating threshold check..."
    STEP7_START=$(get_timestamp)
    "$POST_PROCESSING" "$OUTPUT_FOLDER/binary" "$OUTPUT_FOLDER" \
        --gating-threshold "$GATING_THRESHOLD"

    GATING_RESULT=$?
    STEP7_END=$(get_timestamp)
    if [[ $GATING_RESULT -ne 0 ]]; then
        echo ">>> Step 7: GATING THRESHOLD FAILED"
        OVERALL_RESULT=1
    else
        echo ">>> Step 7: GATING THRESHOLD PASSED"
    fi
    record_step_timing "Gating threshold" "$STEP7_START" "$STEP7_END"
    echo
elif [[ $SKIP_LOG_ANALYSIS -eq 1 || -n "$PERF_TRACE_DIR" ]]; then
    echo ">>> Step 7: SKIPPED (log analysis disabled or perf-trace-dir mode)"
    echo
else
    echo ">>> Step 7: GATING THRESHOLD NOT PROVIDED - pipeline will fail"
    OVERALL_RESULT=1
    echo
fi

# Step 8: Warning threshold check (optional, failure = warning only)
if [[ -n "$WARNING_THRESHOLD" ]]; then
    echo ">>> Step 8: Running warning threshold check..."
    STEP8_START=$(get_timestamp)
    "$POST_PROCESSING" "$OUTPUT_FOLDER/binary" "$OUTPUT_FOLDER" \
        --warning-threshold "$WARNING_THRESHOLD"

    WARNING_RESULT=$?
    STEP8_END=$(get_timestamp)
    if [[ $WARNING_RESULT -ne 0 ]]; then
        echo ">>> Step 8: WARNING THRESHOLD EXCEEDED"
        echo "WARNING: Performance is approaching threshold limits but has not failed gating"
        WARNING_THRESHOLD_FAILED=1
    else
        echo ">>> Step 8: WARNING THRESHOLD PASSED"
    fi
    record_step_timing "Warning threshold" "$STEP8_START" "$STEP8_END"
    echo
fi

# Step 9: Latency summary (optional, if NICD enabled)
if [[ $LATENCY_SUMMARY_ENABLED -eq 1 ]]; then
    echo ">>> Step 9: Generating latency_summary.html..."
    STEP9_START=$(get_timestamp)
    "$POST_PROCESSING" "$OUTPUT_FOLDER/binary_ls" "$OUTPUT_FOLDER" \
        --latency-summary \
        $LABEL_OPT \
        $MMIMO_FLAG

    LATENCY_SUMMARY_RESULT=$?
    STEP9_END=$(get_timestamp)
    if [[ $LATENCY_SUMMARY_RESULT -ne 0 ]]; then
        echo ">>> Step 9: FAILED"
        OVERALL_RESULT=1
    else
        echo ">>> Step 9: COMPLETE"
    fi
    record_step_timing "Latency summary" "$STEP9_START" "$STEP9_END"
    echo

    # Step 10: Latency timeline visualization
    echo ">>> Step 10: Generating latency_timeline.html..."
    STEP10_START=$(get_timestamp)
    "$POST_PROCESSING" "$OUTPUT_FOLDER/binary_ls" "$OUTPUT_FOLDER" \
        --latency-timeline \
        $LABEL_OPT \
        $MMIMO_FLAG

    LATENCY_TIMELINE_RESULT=$?
    STEP10_END=$(get_timestamp)
    if [[ $LATENCY_TIMELINE_RESULT -ne 0 ]]; then
        echo ">>> Step 10: FAILED"
        OVERALL_RESULT=1
    else
        echo ">>> Step 10: COMPLETE"
    fi
    record_step_timing "Latency timeline" "$STEP10_START" "$STEP10_END"
    echo
fi

# Step 11-13: Perf trace analysis (optional, when --perf-trace-dir is provided)
if [[ -n "$PERF_TRACE_DIR" ]]; then
    FUTEX_SUMMARY_SCRIPT="$SCRIPT_DIR/syscall_tracer/futex_cuda_summary_multi_thread.py"
    PAGE_FAULT_SUMMARY_SCRIPT="$SCRIPT_DIR/syscall_tracer/page_fault_summary.py"
    RUNS_SUMMARY_SCRIPT="$SCRIPT_DIR/syscall_tracer/futex_cuda_runs_summary.py"
    FUTEX_THRESHOLD_SCRIPT="$SCRIPT_DIR/syscall_tracer/futex_cuda_threshold_check.py"
    PAGE_FAULT_THRESHOLD_SCRIPT="$SCRIPT_DIR/syscall_tracer/page_fault_threshold_check.py"
    TRACER_LOG_NAME="cuda_api_tracer.log"
    TRACER_LOG_IN_OUTPUT="$OUTPUT_FOLDER/$TRACER_LOG_NAME"

    if [[ ! -d "$PERF_TRACE_DIR" ]]; then
        echo ">>> Step 11: FAILED (perf trace directory not found: $PERF_TRACE_DIR)"
        OVERALL_RESULT=1
    else
        FUTEX_SUMMARY_OK=0
        PAGE_FAULT_SUMMARY_OK=0
        TRACER_LOG_OK=0
        rm -f "$OUTPUT_FOLDER/futex_cuda_summary.txt" \
              "$OUTPUT_FOLDER/page_fault_summary.txt" \
              "$OUTPUT_FOLDER/page_fault_gate_report.txt" \
              "$OUTPUT_FOLDER/futex_cuda_runs_summary.txt" \
              "$OUTPUT_FOLDER/futex_cuda_gate_report.txt" \
              "$TRACER_LOG_IN_OUTPUT"

        echo ">>> Step 11: Running futex_cuda_summary_multi_thread.py..."
        STEP11_START=$(get_timestamp)
        if [[ ! -f "$FUTEX_SUMMARY_SCRIPT" ]]; then
            echo ">>> Step 11: FAILED (script not found: $FUTEX_SUMMARY_SCRIPT)"
            OVERALL_RESULT=1
        else
            python3 "$FUTEX_SUMMARY_SCRIPT" \
                -i "$PERF_TRACE_DIR" \
                -o "$OUTPUT_FOLDER/futex_cuda_summary.txt" \
                --no-sudo
            FUTEX_RESULT=$?
            if [[ $FUTEX_RESULT -ne 0 ]]; then
                echo ">>> Step 11: FAILED (exit code $FUTEX_RESULT)"
                OVERALL_RESULT=1
            else
                FUTEX_SUMMARY_OK=1
                echo ">>> Step 11: COMPLETE"
            fi
        fi

        # If a CUDA API tracer log was produced alongside the perf data, mirror
        # it into the output folder so futex_cuda_runs_summary.py (which scans
        # the parent of the summary file) can pick it up via --include-tracer-log.
        TRACER_LOG_SRC="$PERF_TRACE_DIR/$TRACER_LOG_NAME"
        if [[ -s "$TRACER_LOG_SRC" ]]; then
            # Atomic copy: write to a temp file in the same directory, then rename.
            # Prevents a failed cp (disk full, NFS I/O error, signal) from leaving a
            # truncated destination that still passes existence checks and gets
            # parsed as an empty tracer log by Step 13.
            TRACER_LOG_TMP="${TRACER_LOG_IN_OUTPUT}.tmp.$$"
            if cp -f "$TRACER_LOG_SRC" "$TRACER_LOG_TMP" \
                && mv -f "$TRACER_LOG_TMP" "$TRACER_LOG_IN_OUTPUT"; then
                TRACER_LOG_OK=1
                echo "Copied CUDA API tracer log: $TRACER_LOG_SRC -> $TRACER_LOG_IN_OUTPUT"
            else
                rm -f "$TRACER_LOG_TMP" "$TRACER_LOG_IN_OUTPUT"
                echo ">>> Step 11: FAILED (could not copy CUDA API tracer log)"
                OVERALL_RESULT=1
            fi
        elif [[ -f "$TRACER_LOG_SRC" ]]; then
            # The log file exists but is zero bytes. run2_cuPHYcontroller.sh
            # pre-creates the file, so an empty one means the tracer was enabled
            # but captured nothing -- treat it as a hard failure here with a
            # clear message rather than letting it set TRACER_LOG_OK=1 and fail
            # obscurely as a parse error in the Step 13 gate.
            rm -f "$TRACER_LOG_IN_OUTPUT"
            echo ">>> Step 11: FAILED (CUDA API tracer log is empty: $TRACER_LOG_SRC)"
            OVERALL_RESULT=1
        fi

        STEP11_END=$(get_timestamp)
        record_step_timing "Futex/CUDA summary" "$STEP11_START" "$STEP11_END"
        echo

        echo ">>> Step 11b: Running page_fault_summary.py..."
        STEP11B_START=$(get_timestamp)
        if [[ ! -f "$PAGE_FAULT_SUMMARY_SCRIPT" ]]; then
            echo ">>> Step 11b: FAILED (script not found: $PAGE_FAULT_SUMMARY_SCRIPT)"
            OVERALL_RESULT=1
        else
            python3 "$PAGE_FAULT_SUMMARY_SCRIPT" \
                -i "$PERF_TRACE_DIR" \
                -o "$OUTPUT_FOLDER/page_fault_summary.txt" \
                --no-sudo
            PAGE_FAULT_RESULT=$?
            if [[ $PAGE_FAULT_RESULT -ne 0 ]]; then
                echo ">>> Step 11b: FAILED (exit code $PAGE_FAULT_RESULT)"
                OVERALL_RESULT=1
            else
                PAGE_FAULT_SUMMARY_OK=1
                echo ">>> Step 11b: COMPLETE"
            fi
        fi
        STEP11B_END=$(get_timestamp)
        record_step_timing "Page fault summary" "$STEP11B_START" "$STEP11B_END"
        echo

        # Step 11c: Page-fault threshold gate. Always runs when --perf-trace-dir
        # is set; with the default threshold of 0 any page fault fails the test.
        echo ">>> Step 11c: Running page_fault_threshold_check.py (threshold ${PAGE_FAULT_THRESHOLD})..."
        STEP11C_START=$(get_timestamp)
        if [[ $PAGE_FAULT_SUMMARY_OK -eq 0 ]]; then
            echo ">>> Step 11c: FAILED (prerequisite missing: page fault summary did not succeed)"
            OVERALL_RESULT=1
        elif [[ ! -f "$PAGE_FAULT_THRESHOLD_SCRIPT" ]]; then
            echo ">>> Step 11c: FAILED (script not found: $PAGE_FAULT_THRESHOLD_SCRIPT)"
            OVERALL_RESULT=1
        else
            python3 "$PAGE_FAULT_THRESHOLD_SCRIPT" \
                -i "$OUTPUT_FOLDER/page_fault_summary.txt" \
                --threshold "$PAGE_FAULT_THRESHOLD" \
                -o "$OUTPUT_FOLDER/page_fault_gate_report.txt"
            PAGE_FAULT_GATE_RESULT=$?
            if [[ $PAGE_FAULT_GATE_RESULT -ne 0 ]]; then
                echo ">>> Step 11c: FAILED (exit code $PAGE_FAULT_GATE_RESULT -- page fault threshold exceeded)"
                OVERALL_RESULT=1
            else
                echo ">>> Step 11c: COMPLETE"
            fi
        fi
        STEP11C_END=$(get_timestamp)
        record_step_timing "Page fault threshold gate" "$STEP11C_START" "$STEP11C_END"
        echo

        echo ">>> Step 12: Running futex_cuda_runs_summary.py..."
        STEP12_START=$(get_timestamp)
        if [[ $FUTEX_SUMMARY_OK -eq 0 ]]; then
            echo ">>> Step 12: SKIPPED (Step 11 futex summary did not succeed)"
        elif [[ ! -f "$RUNS_SUMMARY_SCRIPT" ]]; then
            echo ">>> Step 12: FAILED (script not found: $RUNS_SUMMARY_SCRIPT)"
            OVERALL_RESULT=1
        else
            RUNS_OPTS=(-i "$OUTPUT_FOLDER" --summary-name futex_cuda_summary.txt \
                       -o "$OUTPUT_FOLDER/futex_cuda_runs_summary.txt")
            if [[ -f "$TRACER_LOG_IN_OUTPUT" ]]; then
                RUNS_OPTS+=(--include-tracer-log --tracer-log-name "$TRACER_LOG_NAME")
            fi
            python3 "$RUNS_SUMMARY_SCRIPT" "${RUNS_OPTS[@]}"
            RUNS_RESULT=$?
            if [[ $RUNS_RESULT -ne 0 ]]; then
                echo ">>> Step 12: FAILED (exit code $RUNS_RESULT)"
                OVERALL_RESULT=1
            else
                echo ">>> Step 12: COMPLETE"
            fi
        fi
        STEP12_END=$(get_timestamp)
        record_step_timing "Runs summary" "$STEP12_START" "$STEP12_END"
        echo

        # Step 13: Per-API / aggregate futex-vs-CUDA threshold gate
        if [[ -n "$FUTEX_CUDA_THRESHOLD" ]]; then
            echo ">>> Step 13: Running futex_cuda_threshold_check.py (threshold ${FUTEX_CUDA_THRESHOLD}%)..."
            STEP13_START=$(get_timestamp)
            if [[ $FUTEX_SUMMARY_OK -eq 0 || $TRACER_LOG_OK -eq 0 ]]; then
                echo ">>> Step 13: FAILED (prerequisite missing: futex_summary_ok=$FUTEX_SUMMARY_OK, tracer_log_ok=$TRACER_LOG_OK)"
                echo "             (the CUDA API tracer may have been disabled, e.g. by --nsys; see parse_test_config_params.py guard)"
                OVERALL_RESULT=1
            elif [[ ! -f "$FUTEX_THRESHOLD_SCRIPT" ]]; then
                echo ">>> Step 13: FAILED (script not found: $FUTEX_THRESHOLD_SCRIPT)"
                OVERALL_RESULT=1
            else
                python3 "$FUTEX_THRESHOLD_SCRIPT" \
                    --futex-summary "$OUTPUT_FOLDER/futex_cuda_summary.txt" \
                    --tracer-log "$TRACER_LOG_IN_OUTPUT" \
                    --threshold "$FUTEX_CUDA_THRESHOLD" \
                    --min-calls "$FUTEX_CUDA_MIN_CALLS" \
                    -o "$OUTPUT_FOLDER/futex_cuda_gate_report.txt"
                GATE_RESULT=$?
                if [[ $GATE_RESULT -ne 0 ]]; then
                    echo ">>> Step 13: FAILED (exit code $GATE_RESULT -- threshold exceeded)"
                    OVERALL_RESULT=1
                else
                    echo ">>> Step 13: COMPLETE"
                fi
            fi
            STEP13_END=$(get_timestamp)
            record_step_timing "Futex/CUDA threshold gate" "$STEP13_START" "$STEP13_END"
            echo
        fi
    fi
fi

# Step 14: bpftrace all-syscall capture gate (optional, when --bpftrace-dir is
# provided). Runs bpftrace_syscall_summary.py, which writes the per-syscall/
# per-worker table and exits non-zero if the capture is empty (no syscalls) --
# that is the pass/fail criterion. Independent of the perf (--perf-trace-dir)
# gates above, so both can run in the same PERF-modifier post-processing pass.
if [[ -n "$BPFTRACE_DIR" ]]; then
    BPFTRACE_SUMMARY_SCRIPT="$SCRIPT_DIR/syscall_tracer/bpftrace_syscall_summary.py"
    rm -f "$OUTPUT_FOLDER/bpftrace_syscall_summary.txt"

    echo ">>> Step 14: Running bpftrace_syscall_summary.py (gate: FAIL if empty capture)..."
    STEP14_START=$(get_timestamp)
    if [[ ! -d "$BPFTRACE_DIR" ]]; then
        echo ">>> Step 14: FAILED (bpftrace directory not found: $BPFTRACE_DIR)"
        OVERALL_RESULT=1
    elif [[ ! -f "$BPFTRACE_SUMMARY_SCRIPT" ]]; then
        echo ">>> Step 14: FAILED (script not found: $BPFTRACE_SUMMARY_SCRIPT)"
        OVERALL_RESULT=1
    else
        # Gate on dumps discovered in BPFTRACE_DIR (supports custom -o output names).
        # The script exits 1 when zero syscalls were captured (empty dump).
        python3 "$BPFTRACE_SUMMARY_SCRIPT" \
            -i "$BPFTRACE_DIR" \
            -o "$OUTPUT_FOLDER/bpftrace_syscall_summary.txt"
        BPFTRACE_GATE_RESULT=$?
        if [[ $BPFTRACE_GATE_RESULT -ne 0 ]]; then
            echo ">>> Step 14: FAILED (exit code $BPFTRACE_GATE_RESULT -- empty bpftrace capture, no syscalls recorded)"
            OVERALL_RESULT=1
        else
            echo ">>> Step 14: COMPLETE"
        fi
    fi
    STEP14_END=$(get_timestamp)
    record_step_timing "bpftrace syscall gate" "$STEP14_START" "$STEP14_END"
    echo
fi

# Final summary
PIPELINE_END=$(get_timestamp)
PIPELINE_DURATION=$(calc_duration "$PIPELINE_START" "$PIPELINE_END")

echo "=============================================================="
echo "CICD Post-Processing Summary"
echo "=============================================================="
echo "Output files:"
if [[ $SKIP_LOG_ANALYSIS -eq 0 ]]; then
    echo "  - $OUTPUT_FOLDER/perf.csv"
    echo "  - $OUTPUT_FOLDER/compare_logs.html"
    echo "  - $OUTPUT_FOLDER/cpu_timeline.html"
fi
if [[ -n "$GATING_THRESHOLD" || -n "$WARNING_THRESHOLD" || -n "$ABSOLUTE_THRESHOLD" ]]; then
    echo "  - $OUTPUT_FOLDER/threshold_summary.csv"
fi
if [[ -n "$GATING_THRESHOLD" ]]; then
    echo "  - $OUTPUT_FOLDER/gating_threshold_results.csv"
fi
if [[ -n "$WARNING_THRESHOLD" ]]; then
    echo "  - $OUTPUT_FOLDER/warning_threshold_results.csv"
fi
if [[ -n "$ABSOLUTE_THRESHOLD" ]]; then
    echo "  - $OUTPUT_FOLDER/absolute_threshold_results.csv"
fi
if [[ $LATENCY_SUMMARY_ENABLED -eq 1 ]]; then
    echo "  - $OUTPUT_FOLDER/latency_summary.html"
    echo "  - $OUTPUT_FOLDER/latency_timeline.html"
fi
if [[ -n "$PERF_TRACE_DIR" ]]; then
    echo "  - $OUTPUT_FOLDER/futex_cuda_summary.txt"
    if [[ -f "$OUTPUT_FOLDER/page_fault_summary.txt" ]]; then
        echo "  - $OUTPUT_FOLDER/page_fault_summary.txt"
    fi
    if [[ -f "$OUTPUT_FOLDER/page_fault_gate_report.txt" ]]; then
        echo "  - $OUTPUT_FOLDER/page_fault_gate_report.txt"
    fi
    echo "  - $OUTPUT_FOLDER/futex_cuda_runs_summary.txt"
    if [[ -n "$FUTEX_CUDA_THRESHOLD" ]]; then
        echo "  - $OUTPUT_FOLDER/futex_cuda_gate_report.txt"
    fi
fi
if [[ -n "$BPFTRACE_DIR" && -f "$OUTPUT_FOLDER/bpftrace_syscall_summary.txt" ]]; then
    echo "  - $OUTPUT_FOLDER/bpftrace_syscall_summary.txt"
fi
echo
echo "--------------------------------------------------------------"
echo "Step Timing Summary"
echo "--------------------------------------------------------------"
printf "%-22s %s\n" "Step" "Duration"
printf "%-22s %s\n" "----" "--------"
for i in "${!STEP_NAMES[@]}"; do
    printf "%-22s %s\n" "${STEP_NAMES[$i]}" "${STEP_DURATIONS[$i]}"
done
echo "--------------------------------------------------------------"
printf "%-22s %s\n" "TOTAL PIPELINE" "$PIPELINE_DURATION"
echo "--------------------------------------------------------------"
echo
if [[ $OVERALL_RESULT -eq 0 ]]; then
    if [[ $WARNING_THRESHOLD_FAILED -eq 1 ]]; then
        echo "RESULT: PASS (with warning threshold exceeded)"
        echo "=============================================================="
        exit 2
    else
        echo "RESULT: PASS"
        echo "=============================================================="
        exit 0
    fi
else
    echo "RESULT: FAIL"
    echo "=============================================================="
    exit 1
fi
