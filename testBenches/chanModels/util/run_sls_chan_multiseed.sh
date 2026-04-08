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

# Multi-seed SLS channel calibration script
# Runs simulations with different random seeds and combines results for analysis
# Supports both UMa and ISAC calibrations

set -euo pipefail  # Exit on error and fail pipelines

#=============================================================================
# CONFIGURATION - Customize these parameters
#=============================================================================
SEED_START=0
SEED_END=19
YAML_CONFIG="testBenches/chanModels/config/statistic_channel_config_isac_phase2.yaml"
DATASET_NAME="isac_uav_phase2_target"
REFERENCE_JSON="testBenches/chanModels/util/3gpp_calibration_isac_uav_phase2.json"
CALIBRATION_PHASE=2
ISAC_CHANNEL="background"
OUTPUT_DIR="isac_phase2_multiseed_results"
SLS_CHAN_VERBOSE="${SLS_CHAN_VERBOSE:-0}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed-start)
            SEED_START="$2"
            shift 2
            ;;
        --seed-end)
            SEED_END="$2"
            shift 2
            ;;
        --config)
            YAML_CONFIG="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --reference-json)
            REFERENCE_JSON="$2"
            shift 2
            ;;
        --phase)
            CALIBRATION_PHASE="$2"
            shift 2
            ;;
        --isac-channel)
            ISAC_CHANNEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose-sim)
            SLS_CHAN_VERBOSE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seed-start N      Starting seed (default: 0)"
            echo "  --seed-end N        Ending seed (default: 19)"
            echo "  --config FILE       YAML config file path"
            echo "  --dataset NAME      Dataset name for H5 files"
            echo "  --reference-json F  3GPP reference JSON file"
            echo "  --phase N           Calibration phase (1 or 2)"
            echo "  --isac-channel T    ISAC channel type (target or background)"
            echo "  --output-dir DIR    Output directory for results"
            echo "  --verbose-sim       Keep full per-UE simulator logs"
            echo "  --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --seed-start 0 --seed-end 49 --phase 1 --isac-channel target"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

#=============================================================================
# PATHS - Detect SDK root regardless of where script is run from
#=============================================================================
# Get the SDK root directory (3 levels up from util folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SLS_CHAN_EX="$SDK_ROOT/build.aarch64/testBenches/chanModels/examples/sls_chan/sls_chan_ex"
ANALYSIS_SCRIPT="$SDK_ROOT/testBenches/chanModels/util/analysis_channel_stats.py"

# Resolve config paths relative to SDK root
if [[ "$YAML_CONFIG" != /* ]]; then
    YAML_CONFIG="$SDK_ROOT/$YAML_CONFIG"
fi
if [[ "$REFERENCE_JSON" != /* ]]; then
    REFERENCE_JSON="$SDK_ROOT/$REFERENCE_JSON"
fi

#=============================================================================
# FUNCTIONS
#=============================================================================

# Function to update rand_seed in YAML file
update_yaml_seed() {
    local seed=$1
    local yaml_file=$2
    
    # Use sed to replace rand_seed value
    # Handles both "rand_seed: N" and "rand_seed: N  # comment" formats
    sed -i "s/^\(\s*rand_seed:\s*\)[0-9]*/\1${seed}/" "$yaml_file"
    
    echo "Updated rand_seed to $seed in $yaml_file"
}

# Function to get current seed from YAML
get_yaml_seed() {
    local yaml_file=$1
    grep -E "^\s*rand_seed:" "$yaml_file" | sed 's/.*rand_seed:\s*\([0-9]*\).*/\1/'
}

#=============================================================================
# MAIN SCRIPT
#=============================================================================

echo "============================================================"
echo "SLS Channel Multi-Seed Calibration Script"
echo "============================================================"
echo "Seed range: $SEED_START to $SEED_END ($(( SEED_END - SEED_START + 1 )) drops)"
echo "YAML config: $YAML_CONFIG"
echo "Dataset name: $DATASET_NAME"
echo "Reference JSON: $REFERENCE_JSON"
echo "Calibration phase: $CALIBRATION_PHASE"
echo "ISAC channel: $ISAC_CHANNEL"
echo "Output directory: $OUTPUT_DIR"
echo "Verbose simulator logs: $SLS_CHAN_VERBOSE"
echo "============================================================"
echo ""

# Check if executable exists
if [ ! -f "$SLS_CHAN_EX" ]; then
    echo "ERROR: sls_chan_ex not found at $SLS_CHAN_EX"
    echo "Please build with: cmake --build build.aarch64 -t sls_chan_ex"
    exit 1
fi

# Check if YAML config exists
if [ ! -f "$YAML_CONFIG" ]; then
    echo "ERROR: YAML config not found: $YAML_CONFIG"
    exit 1
fi

# Backup original YAML
YAML_BACKUP="${YAML_CONFIG}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$YAML_CONFIG" "$YAML_BACKUP"
echo "Backed up YAML config to: $YAML_BACKUP"

# Track generated H5 files
declare -a H5_FILES

echo ""
echo "============================================================"
echo "Running simulations with seeds $SEED_START to $SEED_END"
echo "============================================================"

for seed in $(seq $SEED_START $SEED_END); do
    echo ""
    echo "--- Seed $seed / $SEED_END ---"
    
    # Update seed in YAML
    update_yaml_seed $seed "$YAML_CONFIG"
    
    # Run simulation. By default, suppress per-UE trajectory spam in logs.
    echo "Running simulation..."
    if [ "$SLS_CHAN_VERBOSE" -eq 1 ]; then
        "$SLS_CHAN_EX" "$YAML_CONFIG" -d "${DATASET_NAME}"
    else
        "$SLS_CHAN_EX" "$YAML_CONFIG" -d "${DATASET_NAME}" 2>&1 | awk '
            /^SHOW_TRAJECTORY:/ { in_trajectory=1; next }
            in_trajectory && /^  UE uid=/ { next }
            in_trajectory { in_trajectory=0 }
            { print }
        '
    fi
    
    # The H5 file will be named: slsChanData_*_seed${seed}.h5
    # Find the most recently created H5 file matching this pattern
    H5_FILE=$(ls -t slsChanData_*_${DATASET_NAME}_seed${seed}.h5 2>/dev/null | head -1)
    
    if [ -n "$H5_FILE" ]; then
        echo "Generated: $H5_FILE"
        H5_FILES+=("$H5_FILE")
    else
        # Try alternate naming pattern (TTI in filename)
        H5_FILE=$(ls -t slsChanData_*_TTI0_${DATASET_NAME}_seed${seed}.h5 2>/dev/null | head -1)
        if [ -n "$H5_FILE" ]; then
            echo "Generated: $H5_FILE"
            H5_FILES+=("$H5_FILE")
        else
            echo "WARNING: Could not find H5 file for seed $seed"
        fi
    fi
done

# Restore original YAML
cp "$YAML_BACKUP" "$YAML_CONFIG"
echo ""
echo "Restored original YAML config from backup"

echo ""
echo "============================================================"
echo "Simulation complete! Generated ${#H5_FILES[@]} H5 files"
echo "============================================================"

if [ ${#H5_FILES[@]} -eq 0 ]; then
    echo "ERROR: No H5 files were generated"
    exit 1
fi

# List all generated files
echo "Generated files:"
for f in "${H5_FILES[@]}"; do
    echo "  - $f"
done

echo ""
echo "============================================================"
echo "Running multi-seed analysis"
echo "============================================================"

# Use the first H5 file as the base - the script will find all matching files
FIRST_H5="${H5_FILES[0]}"

python3 "$ANALYSIS_SCRIPT" \
    "$FIRST_H5" \
    --reference-json "$REFERENCE_JSON" \
    --calibration-phase "$CALIBRATION_PHASE" \
    --isac-channel "$ISAC_CHANNEL" \
    --multi-seed \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "COMPLETE!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "H5 files generated: ${#H5_FILES[@]}"
echo "Total drops (seeds): $(( SEED_END - SEED_START + 1 ))"
echo ""

