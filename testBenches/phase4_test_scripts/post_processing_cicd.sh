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

# Default values
NUM_PROC=32
MMIMO_FLAG=""
LABEL=""

# Internal timing parameters (not configurable via CLI)
# Performance metrics: longer window to capture steady-state behavior
PERFMETRICS_MAX_DURATION=300
PERFMETRICS_IGNORE_DURATION=30
# Analysis (compare_logs, latency_summary): shorter window for visualization
ANALYSIS_MAX_DURATION=60
ANALYSIS_IGNORE_DURATION=30
# Timeline plots: very short window for detailed timeline visualization
TIMELINE_MAX_DURATION=30.2
TIMELINE_IGNORE_DURATION=30

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

show_usage() {
    echo "Usage: $0 <phy_log> <testmac_log> <ru_log> <output_folder> [options]"
    echo
    echo "CICD wrapper that runs the full post-processing sequence with proper return codes."
    echo "Steps 1-4 (parse, metrics, visualizations) always run."
    echo "Threshold checks run if their files are provided. Pipeline fails without gating."
    echo
    echo "Positional Arguments (4 required):"
    echo "  phy_log                Path to phy.log file"
    echo "  testmac_log            Path to testmac.log file"
    echo "  ru_log                 Path to ru.log file (can be blank placeholder)"
    echo "  output_folder          Directory for output files"
    echo
    echo "Threshold Arguments (all optional, but gating is mandatory for a PASS):"
    echo "  --gating-threshold <file>    Gating perf_requirements file (pipeline fails without this)"
    echo "  --warning-threshold <file>   Warning perf_requirements file"
    echo "  --absolute-threshold <file>  Absolute perf_requirements file"
    echo "  --latency-summary            Also run latency summary (NICD)"
    echo "  --mmimo                      Enable mMIMO mode"
    echo "  --num-proc <n>               Number of processing threads (default: $NUM_PROC)"
    echo "  --label <name>               Label for compare_logs output"
    echo "  -h, --help                   Show this help message"
    echo
    echo "Timing Parameters (internally configured):"
    echo "  Perf metrics:                  -i $PERFMETRICS_IGNORE_DURATION -m $PERFMETRICS_MAX_DURATION"
    echo "  Compare logs, latency summary: -i $ANALYSIS_IGNORE_DURATION -m $ANALYSIS_MAX_DURATION"
    echo "  Timeline plots:                -i $TIMELINE_IGNORE_DURATION -m $TIMELINE_MAX_DURATION"
    echo "  Thresholds:                    (no timing params, uses perf.csv)"
    echo
    echo "Sequence Executed:"
    echo "  1. post_processing_parse.sh (--perf-metrics, optionally --latency-summary)"
    echo "  2. post_processing_analyze.sh --perf-metrics"
    echo "  3. post_processing_analyze.sh --compare-logs"
    echo "  4. post_processing_analyze.sh --cpu-timeline"
    echo "  5. post_processing_analyze.sh --threshold-summary (informational)"
    echo "  6. post_processing_analyze.sh --absolute-threshold (if provided)"
    echo "  7. post_processing_analyze.sh --gating-threshold"
    echo "  8. post_processing_analyze.sh --warning-threshold (if provided, failure = warning only)"
    echo "  9. post_processing_analyze.sh --latency-summary (if --latency-summary enabled)"
    echo " 10. post_processing_analyze.sh --latency-timeline (if --latency-summary enabled)"
    echo
    echo "Return Codes:"
    echo "  0 = All steps passed"
    echo "  1 = One or more steps failed (all steps run to completion before returning)"
    echo "  2 = All steps passed, but warning threshold exceeded"
    echo
    echo "Example:"
    echo "  $0 phy.log testmac.log ru.log ./output \\"
    echo "      --gating-threshold /path/to/gating_perf_requirements.csv \\"
    echo "      --warning-threshold /path/to/warning_perf_requirements.csv \\"
    echo "      --absolute-threshold /path/to/perf_requirements.csv \\"
    echo "      --mmimo --label my_test"
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
        --num-proc)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --num-proc option"
                exit 1
            fi
            NUM_PROC="$2"
            shift 2
            ;;
        --label)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --label option"
                exit 1
            fi
            LABEL="$2"
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

# Script paths
PARSE_LOGS="$SCRIPT_DIR/post_processing_parse.sh"
POST_PROCESSING="$SCRIPT_DIR/post_processing_analyze.sh"

if [[ ! -f "$PARSE_LOGS" ]]; then
    echo "Error: post_processing_parse.sh not found at $PARSE_LOGS"
    exit 1
fi

if [[ ! -f "$POST_PROCESSING" ]]; then
    echo "Error: post_processing_analyze.sh not found at $POST_PROCESSING"
    exit 1
fi

# Build timing arguments (different for perf metrics vs analysis visualizations vs timelines)
PERFMETRICS_OPTS="--max-duration $PERFMETRICS_MAX_DURATION --ignore-duration $PERFMETRICS_IGNORE_DURATION"
ANALYSIS_OPTS="--max-duration $ANALYSIS_MAX_DURATION --ignore-duration $ANALYSIS_IGNORE_DURATION"
TIMELINE_OPTS="--max-duration $TIMELINE_MAX_DURATION --ignore-duration $TIMELINE_IGNORE_DURATION"
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
echo "=============================================================="
echo

# Step 1: Parse logs
echo ">>> Step 1: Parsing logs..."
STEP1_START=$(get_timestamp)
PARSE_OPTS="--perf-metrics"
if [[ $LATENCY_SUMMARY_ENABLED -eq 1 ]]; then
    PARSE_OPTS="$PARSE_OPTS --latency-summary"
fi

"$PARSE_LOGS" "$PHY_LOG" "$TESTMAC_LOG" "$RU_LOG" "$OUTPUT_FOLDER" \
    $PARSE_OPTS \
    --max-duration "$PERFMETRICS_MAX_DURATION" \
    --num-proc "$NUM_PROC" \
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
"$POST_PROCESSING" "$OUTPUT_FOLDER/binary" "$OUTPUT_FOLDER" \
    --perf-metrics \
    $PERFMETRICS_OPTS \
    $MMIMO_FLAG

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
    $ANALYSIS_OPTS \
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
    $TIMELINE_OPTS \
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
        $ANALYSIS_OPTS \
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
        $TIMELINE_OPTS \
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

# Final summary
PIPELINE_END=$(get_timestamp)
PIPELINE_DURATION=$(calc_duration "$PIPELINE_START" "$PIPELINE_END")

echo "=============================================================="
echo "CICD Post-Processing Summary"
echo "=============================================================="
echo "Output files:"
echo "  - $OUTPUT_FOLDER/perf.csv"
echo "  - $OUTPUT_FOLDER/compare_logs.html"
echo "  - $OUTPUT_FOLDER/cpu_timeline.html"
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
