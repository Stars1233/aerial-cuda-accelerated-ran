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
# post_processing_analyze.sh - Run downstream processing on parsed binary data
#
# This script runs individual post-processing steps on already-parsed binary data.
# Caller should wrap with taskset/chrt if CPU affinity is needed.
#-----------------------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}

# Default values
MAX_DURATION=300
IGNORE_DURATION=30
MMIMO_FLAG=""
LABEL=""

# Mode variables
MODE=""
THRESHOLD_FILE=""

show_usage() {
    echo "Usage: $0 <input_folder> <output_folder> <mode_flag> [options]"
    echo
    echo "Run downstream processing on already-parsed binary data."
    echo
    echo "Positional Arguments (2 required):"
    echo "  input_folder           Directory containing parsed binary data"
    echo "  output_folder          Directory for output products"
    echo
    echo "Mode Flags (exactly one required):"
    echo "  --perf-metrics                  Run cicd_performance_metrics.py (creates perf.csv)"
    echo "  --compare-logs                  Generate compare_logs.html visualization"
    echo "  --cpu-timeline                  Generate cpu_timeline.html visualization"
    echo "  --gating-threshold <file>       Run gating threshold check"
    echo "  --warning-threshold <file>      Run warning threshold check"
    echo "  --absolute-threshold <file>     Run absolute threshold check"
    echo "  --latency-summary               Run latency_summary.py (creates latency_summary.html)"
    echo "  --latency-timeline              Generate latency_timeline.html visualization"
    echo
    echo "Optional Arguments:"
    echo "  --mmimo                 Enable mMIMO mode (-e flag to Python scripts)"
    echo "  --max-duration <sec>    Max duration to process (default: $MAX_DURATION)"
    echo "  --ignore-duration <sec> Initial seconds to skip (default: $IGNORE_DURATION)"
    echo "  --label <name>          Label for compare_logs/latency_summary output"
    echo "  -h, --help              Show this help message"
    echo
    echo "Return Codes:"
    echo "  0 = success (or pass for threshold checks)"
    echo "  1 = failure (or fail for threshold checks)"
    echo
    echo "Examples:"
    echo "  # Performance metrics extraction"
    echo "  $0 ./output/binary ./output --perf-metrics --mmimo"
    echo
    echo "  # Compare logs visualization"
    echo "  $0 ./output/binary ./output --compare-logs --label my_test --mmimo"
    echo
    echo "  # Gating threshold check"
    echo "  $0 ./output/binary ./output --gating-threshold /path/to/gating.csv"
    echo
    echo "  # Latency summary (NICD)"
    echo "  $0 ./output/binary_ls ./output --latency-summary --mmimo"
}

# Parse positional arguments and options
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --perf-metrics)
            MODE="perf-metrics"
            shift
            ;;
        --compare-logs)
            MODE="compare-logs"
            shift
            ;;
        --cpu-timeline)
            MODE="cpu-timeline"
            shift
            ;;
        --gating-threshold)
            MODE="gating-threshold"
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing threshold file for --gating-threshold option"
                exit 1
            fi
            THRESHOLD_FILE="$2"
            shift 2
            ;;
        --warning-threshold)
            MODE="warning-threshold"
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing threshold file for --warning-threshold option"
                exit 1
            fi
            THRESHOLD_FILE="$2"
            shift 2
            ;;
        --absolute-threshold)
            MODE="absolute-threshold"
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing threshold file for --absolute-threshold option"
                exit 1
            fi
            THRESHOLD_FILE="$2"
            shift 2
            ;;
        --threshold-summary)
            MODE="threshold-summary"
            shift
            # Collect remaining args as pass-through to the Python script
            THRESHOLD_SUMMARY_ARGS=()
            while [[ $# -gt 0 ]]; do
                THRESHOLD_SUMMARY_ARGS+=("$1")
                shift
            done
            ;;
        --latency-summary)
            MODE="latency-summary"
            shift
            ;;
        --latency-timeline)
            MODE="latency-timeline"
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
        --ignore-duration)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --ignore-duration option"
                exit 1
            fi
            IGNORE_DURATION="$2"
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
if [[ ${#POSITIONAL_ARGS[@]} -eq 0 && -z "$MODE" ]]; then
    echo "No arguments provided. Use --help for usage information."
    exit 0
fi

# Validate positional arguments
if [[ ${#POSITIONAL_ARGS[@]} -ne 2 ]]; then
    echo "Error: Expected 2 positional arguments (input_folder, output_folder)"
    echo "Got ${#POSITIONAL_ARGS[@]} arguments: ${POSITIONAL_ARGS[*]}"
    show_usage
    exit 1
fi

INPUT_FOLDER="${POSITIONAL_ARGS[0]}"
OUTPUT_FOLDER="${POSITIONAL_ARGS[1]}"

# Validate that a mode flag is specified
if [[ -z "$MODE" ]]; then
    echo "Error: A mode flag is required"
    show_usage
    exit 1
fi

# Validate input folder exists
if [[ ! -d "$INPUT_FOLDER" ]]; then
    echo "Error: Input folder not found: $INPUT_FOLDER"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Activate aerial_postproc virtual environment
source "$SCRIPT_DIR/aerial_postproc/venv_activate.sh" || {
    echo "Warning: Failed to activate aerial_postproc virtual environment"
    echo "Proceeding without virtual environment activation..."
}

# Script paths
CICD_PERF_METRICS="$SCRIPT_DIR/aerial_postproc/scripts/cicd/cicd_performance_metrics.py"
CICD_PERF_THRESHOLD_ABS="$SCRIPT_DIR/aerial_postproc/scripts/cicd/cicd_performance_threshold_absolute.py"
CICD_THRESHOLD_SUMMARY="$SCRIPT_DIR/aerial_postproc/scripts/cicd/cicd_threshold_summary.py"
COMPARE_LOGS="$SCRIPT_DIR/aerial_postproc/scripts/analysis/compare_logs.py"
CPU_TIMELINE="$SCRIPT_DIR/aerial_postproc/scripts/analysis/cpu_timeline_plot.py"
LATENCY_SUMMARY="$SCRIPT_DIR/aerial_postproc/scripts/analysis/latency_summary.py"
LATENCY_TIMELINE="$SCRIPT_DIR/aerial_postproc/scripts/analysis/latency_timeline_plot.py"

EXIT_CODE=0

case $MODE in
    perf-metrics)
        echo "=== Running Performance Metrics Extraction ==="
        echo "Input: $INPUT_FOLDER"
        echo "Output: $OUTPUT_FOLDER/perf.csv"

        if [[ ! -f "$CICD_PERF_METRICS" ]]; then
            echo "Error: cicd_performance_metrics.py not found at $CICD_PERF_METRICS"
            exit 1
        fi

        python3 "$CICD_PERF_METRICS" "$INPUT_FOLDER" \
            -p "$OUTPUT_FOLDER/perf.csv" \
            -m "$MAX_DURATION" \
            -i "$IGNORE_DURATION" \
            $MMIMO_FLAG

        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "Performance metrics extraction completed successfully"
        else
            echo "Error: Performance metrics extraction failed with exit code $EXIT_CODE"
        fi
        ;;

    compare-logs)
        echo "=== Running Compare Logs Visualization ==="
        echo "Input: $INPUT_FOLDER"
        echo "Output: $OUTPUT_FOLDER/compare_logs.html"

        if [[ ! -f "$COMPARE_LOGS" ]]; then
            echo "Error: compare_logs.py not found at $COMPARE_LOGS"
            exit 1
        fi

        LABEL_ARG=""
        if [[ -n "$LABEL" ]]; then
            LABEL_ARG="-l $LABEL"
        fi

        python3 "$COMPARE_LOGS" "$INPUT_FOLDER" \
            -o "$OUTPUT_FOLDER/compare_logs.html" \
            -m "$MAX_DURATION" \
            -i "$IGNORE_DURATION" \
            $LABEL_ARG \
            $MMIMO_FLAG

        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "Compare logs visualization completed successfully"
        else
            echo "Error: Compare logs visualization failed with exit code $EXIT_CODE"
        fi
        ;;

    cpu-timeline)
        echo "=== Running CPU Timeline Visualization ==="
        echo "Input: $INPUT_FOLDER"
        echo "Output: $OUTPUT_FOLDER/cpu_timeline.html"

        if [[ ! -f "$CPU_TIMELINE" ]]; then
            echo "Error: cpu_timeline_plot.py not found at $CPU_TIMELINE"
            exit 1
        fi

        LABEL_ARG=""
        if [[ -n "$LABEL" ]]; then
            LABEL_ARG="-l $LABEL"
        fi

        python3 "$CPU_TIMELINE" "$INPUT_FOLDER" \
            -o "$OUTPUT_FOLDER/cpu_timeline.html" \
            -m "$MAX_DURATION" \
            -i "$IGNORE_DURATION" \
            $LABEL_ARG

        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "CPU timeline visualization completed successfully"
        else
            echo "Error: CPU timeline visualization failed with exit code $EXIT_CODE"
        fi
        ;;

    gating-threshold)
        echo "=== Running Gating Threshold Check ==="
        echo "Performance CSV: $OUTPUT_FOLDER/perf.csv"
        echo "Requirements file: $THRESHOLD_FILE"

        if [[ ! -f "$CICD_PERF_THRESHOLD_ABS" ]]; then
            echo "Error: cicd_performance_threshold_absolute.py not found at $CICD_PERF_THRESHOLD_ABS"
            exit 1
        fi

        # Validate that perf.csv exists
        if [[ ! -f "$OUTPUT_FOLDER/perf.csv" ]]; then
            echo "Error: perf.csv not found at $OUTPUT_FOLDER/perf.csv"
            echo "Run --perf-metrics first to generate perf.csv"
            exit 1
        fi

        # Validate requirements file exists
        if [[ ! -f "$THRESHOLD_FILE" ]]; then
            echo "Error: Requirements file not found: $THRESHOLD_FILE"
            exit 1
        fi

        python3 "$CICD_PERF_THRESHOLD_ABS" "$OUTPUT_FOLDER/perf.csv" "$THRESHOLD_FILE" \
            -o "$OUTPUT_FOLDER/gating_threshold_results.csv"

        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "Gating threshold check: PASS"
        else
            echo "Gating threshold check: FAIL"
        fi
        ;;

    warning-threshold)
        echo "=== Running Warning Threshold Check ==="
        echo "Performance CSV: $OUTPUT_FOLDER/perf.csv"
        echo "Requirements file: $THRESHOLD_FILE"

        if [[ ! -f "$CICD_PERF_THRESHOLD_ABS" ]]; then
            echo "Error: cicd_performance_threshold_absolute.py not found at $CICD_PERF_THRESHOLD_ABS"
            exit 1
        fi

        # Validate that perf.csv exists
        if [[ ! -f "$OUTPUT_FOLDER/perf.csv" ]]; then
            echo "Error: perf.csv not found at $OUTPUT_FOLDER/perf.csv"
            echo "Run --perf-metrics first to generate perf.csv"
            exit 1
        fi

        # Validate requirements file exists
        if [[ ! -f "$THRESHOLD_FILE" ]]; then
            echo "Error: Requirements file not found: $THRESHOLD_FILE"
            exit 1
        fi

        python3 "$CICD_PERF_THRESHOLD_ABS" "$OUTPUT_FOLDER/perf.csv" "$THRESHOLD_FILE" \
            -o "$OUTPUT_FOLDER/warning_threshold_results.csv"

        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "Warning threshold check: PASS"
        else
            echo "Warning threshold check: FAIL"
        fi
        ;;

    absolute-threshold)
        echo "=== Running Absolute Threshold Check ==="
        echo "Performance CSV: $OUTPUT_FOLDER/perf.csv"
        echo "Requirements file: $THRESHOLD_FILE"

        if [[ ! -f "$CICD_PERF_THRESHOLD_ABS" ]]; then
            echo "Error: cicd_performance_threshold_absolute.py not found at $CICD_PERF_THRESHOLD_ABS"
            exit 1
        fi

        # Validate that perf.csv exists
        if [[ ! -f "$OUTPUT_FOLDER/perf.csv" ]]; then
            echo "Error: perf.csv not found at $OUTPUT_FOLDER/perf.csv"
            echo "Run --perf-metrics first to generate perf.csv"
            exit 1
        fi

        # Validate requirements file exists
        if [[ ! -f "$THRESHOLD_FILE" ]]; then
            echo "Error: Requirements file not found: $THRESHOLD_FILE"
            exit 1
        fi

        python3 "$CICD_PERF_THRESHOLD_ABS" "$OUTPUT_FOLDER/perf.csv" "$THRESHOLD_FILE" \
            -o "$OUTPUT_FOLDER/absolute_threshold_results.csv"

        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "Absolute threshold check: PASS"
        else
            echo "Absolute threshold check: FAIL"
        fi
        ;;

    threshold-summary)
        echo "=== Running Threshold Summary ==="

        if [[ ! -f "$CICD_THRESHOLD_SUMMARY" ]]; then
            echo "Error: cicd_threshold_summary.py not found at $CICD_THRESHOLD_SUMMARY"
            exit 1
        fi

        if [[ ! -f "$OUTPUT_FOLDER/perf.csv" ]]; then
            echo "Error: perf.csv not found at $OUTPUT_FOLDER/perf.csv"
            echo "Run --perf-metrics first to generate perf.csv"
            exit 1
        fi

        python3 "$CICD_THRESHOLD_SUMMARY" "$OUTPUT_FOLDER/perf.csv" "${THRESHOLD_SUMMARY_ARGS[@]}"

        EXIT_CODE=$?
        ;;

    latency-summary)
        echo "=== Running Latency Summary Visualization ==="
        echo "Input: $INPUT_FOLDER"
        echo "Output: $OUTPUT_FOLDER/latency_summary.html"

        if [[ ! -f "$LATENCY_SUMMARY" ]]; then
            echo "Error: latency_summary.py not found at $LATENCY_SUMMARY"
            exit 1
        fi

        LABEL_ARG=""
        if [[ -n "$LABEL" ]]; then
            LABEL_ARG="-l $LABEL"
        fi

        python3 "$LATENCY_SUMMARY" "$INPUT_FOLDER" \
            -o "$OUTPUT_FOLDER/latency_summary.html" \
            -m "$MAX_DURATION" \
            -i "$IGNORE_DURATION" \
            $LABEL_ARG \
            $MMIMO_FLAG

        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "Latency summary visualization completed successfully"
        else
            echo "Error: Latency summary visualization failed with exit code $EXIT_CODE"
        fi
        ;;

    latency-timeline)
        echo "=== Running Latency Timeline Visualization ==="
        echo "Input: $INPUT_FOLDER"
        echo "Output: $OUTPUT_FOLDER/latency_timeline.html"

        if [[ ! -f "$LATENCY_TIMELINE" ]]; then
            echo "Error: latency_timeline_plot.py not found at $LATENCY_TIMELINE"
            exit 1
        fi

        LABEL_ARG=""
        if [[ -n "$LABEL" ]]; then
            LABEL_ARG="-l $LABEL"
        fi

        python3 "$LATENCY_TIMELINE" "$INPUT_FOLDER" \
            -o "$OUTPUT_FOLDER/latency_timeline.html" \
            -m "$MAX_DURATION" \
            -i "$IGNORE_DURATION" \
            $LABEL_ARG \
            $MMIMO_FLAG

        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 0 ]]; then
            echo "Latency timeline visualization completed successfully"
        else
            echo "Error: Latency timeline visualization failed with exit code $EXIT_CODE"
        fi
        ;;

    *)
        echo "Error: Unknown mode: $MODE"
        exit 1
        ;;
esac

# Deactivate virtual environment if it was activated
deactivate 2>/dev/null || true

exit $EXIT_CODE
