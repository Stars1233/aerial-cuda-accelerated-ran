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
# post_processing_parse.sh - Parse log files using cicd_parse.py
#
# This script runs the expensive cicd_parse.py step to generate binary output folders.
# It supports parsing for performance metrics and/or latency summary (NICD) formats.
#-----------------------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}

# Default values
MAX_DURATION=300
NUM_PROC=32
MMIMO_FLAG=""
PERF_METRICS=0
LATENCY_SUMMARY=0

show_usage() {
    echo "Usage: $0 <phy_log> <testmac_log> <ru_log> <output_folder> [options]"
    echo
    echo "Parse log files to generate binary output folders for post-processing."
    echo
    echo "Positional Arguments (4 required):"
    echo "  phy_log                Path to phy.log file"
    echo "  testmac_log            Path to testmac.log file"
    echo "  ru_log                 Path to ru.log file (can be blank placeholder)"
    echo "  output_folder          Directory for output files"
    echo
    echo "Mode Flags (at least one required):"
    echo "  --perf-metrics         Parse for performance metrics (output: \$output_folder/binary/)"
    echo "  --latency-summary      Parse for latency summary/NICD (output: \$output_folder/binary_ls/)"
    echo
    echo "Optional Arguments:"
    echo "  --mmimo                Enable mMIMO mode (-e flag to cicd_parse.py)"
    echo "  --max-duration <sec>   Max duration to process (default: $MAX_DURATION)"
    echo "  --num-proc <n>         Number of processing threads (default: $NUM_PROC)"
    echo "  -h, --help             Show this help message"
    echo
    echo "Examples:"
    echo "  # Parse for performance metrics only"
    echo "  $0 phy.log testmac.log ru.log ./output --perf-metrics --mmimo"
    echo
    echo "  # Parse for latency summary only (NICD)"
    echo "  $0 phy.log testmac.log ru.log ./output --latency-summary --mmimo"
    echo
    echo "  # Parse for both formats"
    echo "  $0 phy.log testmac.log ru.log ./output --perf-metrics --latency-summary --mmimo"
}

# Parse positional arguments first
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --perf-metrics)
            PERF_METRICS=1
            shift
            ;;
        --latency-summary)
            LATENCY_SUMMARY=1
            shift
            ;;
        --mmimo)
            MMIMO_FLAG="-e"
            shift
            ;;
        --max-duration)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --max-duration option"
                exit 1
            fi
            MAX_DURATION="$2"
            shift 2
            ;;
        --num-proc)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --num-proc option"
                exit 1
            fi
            NUM_PROC="$2"
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
if [[ ${#POSITIONAL_ARGS[@]} -eq 0 && $PERF_METRICS -eq 0 && $LATENCY_SUMMARY -eq 0 ]]; then
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

# Validate that at least one mode flag is specified
if [[ $PERF_METRICS -eq 0 && $LATENCY_SUMMARY -eq 0 ]]; then
    echo "Error: At least one mode flag (--perf-metrics or --latency-summary) is required"
    show_usage
    exit 1
fi

# Validate input files exist
if [[ ! -f "$PHY_LOG" ]]; then
    echo "Error: PHY log file not found: $PHY_LOG"
    exit 1
fi

if [[ ! -f "$TESTMAC_LOG" ]]; then
    echo "Error: testMAC log file not found: $TESTMAC_LOG"
    exit 1
fi

if [[ ! -f "$RU_LOG" ]]; then
    echo "Error: RU log file not found: $RU_LOG"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Activate aerial_postproc virtual environment
source "$SCRIPT_DIR/aerial_postproc/venv_activate.sh" || {
    echo "Warning: Failed to activate aerial_postproc virtual environment"
    echo "Proceeding without virtual environment activation..."
}

# Path to cicd_parse.py
CICD_PARSE="$SCRIPT_DIR/aerial_postproc/scripts/cicd/cicd_parse.py"

if [[ ! -f "$CICD_PARSE" ]]; then
    echo "Error: cicd_parse.py not found at $CICD_PARSE"
    exit 1
fi

OVERALL_EXIT_CODE=0

# Parse for performance metrics if requested
if [[ $PERF_METRICS -eq 1 ]]; then
    echo "=== Parsing for Performance Metrics ==="
    echo "Output folder: $OUTPUT_FOLDER/binary/"

    python3 "$CICD_PARSE" "$PHY_LOG" "$TESTMAC_LOG" "$RU_LOG" \
        -o "$OUTPUT_FOLDER/binary" \
        -f PerfMetrics \
        -n "$NUM_PROC" \
        -m "$MAX_DURATION" \
        $MMIMO_FLAG

    PARSE_EXIT_CODE=$?
    if [[ $PARSE_EXIT_CODE -ne 0 ]]; then
        echo "Error: Performance metrics parsing failed with exit code $PARSE_EXIT_CODE"
        OVERALL_EXIT_CODE=1
    else
        echo "Performance metrics parsing completed successfully"
    fi
fi

# Parse for latency summary if requested
if [[ $LATENCY_SUMMARY -eq 1 ]]; then
    echo "=== Parsing for Latency Summary (NICD) ==="
    echo "Output folder: $OUTPUT_FOLDER/binary_ls/"

    python3 "$CICD_PARSE" "$PHY_LOG" "$TESTMAC_LOG" "$RU_LOG" \
        -o "$OUTPUT_FOLDER/binary_ls" \
        -f LatencySummary \
        -n "$NUM_PROC" \
        -m "$MAX_DURATION" \
        $MMIMO_FLAG

    PARSE_EXIT_CODE=$?
    if [[ $PARSE_EXIT_CODE -ne 0 ]]; then
        echo "Error: Latency summary parsing failed with exit code $PARSE_EXIT_CODE"
        OVERALL_EXIT_CODE=1
    else
        echo "Latency summary parsing completed successfully"
    fi
fi

# Deactivate virtual environment if it was activated
deactivate 2>/dev/null || true

exit $OVERALL_EXIT_CODE
