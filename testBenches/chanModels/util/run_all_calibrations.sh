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

# Master Calibration Script - Runs both UMa and ISAC calibrations
# This script runs all calibration phases for UMa and ISAC channels

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make sure Python dependencies are installed (use python3 -m pip so pip3-only systems work)
python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"

# Default configuration
# Note: If no seed range provided, UMa uses 0-4 (5 seeds) and ISAC uses 0-199 (200 seeds)
SEED_START=""
SEED_END=""
SEED_START_PROVIDED=false
SEED_END_PROVIDED=false
RUN_UMA=true
RUN_ISAC=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed-start)
            SEED_START="$2"
            SEED_START_PROVIDED=true
            shift 2
            ;;
        --seed-end)
            SEED_END="$2"
            SEED_END_PROVIDED=true
            shift 2
            ;;
        --uma-only)
            RUN_ISAC=false
            shift
            ;;
        --isac-only)
            RUN_UMA=false
            shift
            ;;
        --help)
            echo "Master Calibration Script - Runs both UMa and ISAC calibrations"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --seed-start N    Starting seed (default: 0 for both UMa and ISAC)"
            echo "  --seed-end N      Ending seed (default: 4 for UMa, 199 for ISAC)"
            echo "  --uma-only        Run only UMa calibration (default seeds: 0-4)"
            echo "  --isac-only       Run only ISAC calibration (default seeds: 0-199)"
            echo "  --help            Show this help message"
            echo ""
            echo "Default Behavior:"
            echo "  - UMa calibration uses 5 seeds (0-4) by default"
            echo "  - ISAC calibration uses 200 seeds (0-199) by default"
            echo "  - If you provide --seed-start or --seed-end, it overrides the defaults for ALL calibrations"
            echo ""
            echo "Examples:"
            echo "  # Run all calibrations with default seeds (UMa: 0-4, ISAC: 0-199)"
            echo "  $0"
            echo ""
            echo "  # Quick test with 10 seeds for both UMa and ISAC"
            echo "  $0 --seed-start 0 --seed-end 9"
            echo ""
            echo "  # Run only UMa calibration with default seeds (0-4)"
            echo "  $0 --uma-only"
            echo ""
            echo "  # Run only ISAC calibration with custom seed range"
            echo "  $0 --isac-only --seed-start 0 --seed-end 49"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default seed ranges if not provided
if [ "$SEED_START_PROVIDED" = false ]; then
    SEED_START=0
fi

if [ "$SEED_END_PROVIDED" = false ]; then
    # Use different defaults based on what's being run
    if [ "$RUN_UMA" = true ] && [ "$RUN_ISAC" = false ]; then
        # UMa only: default to 5 seeds (0-4)
        SEED_END=4
    elif [ "$RUN_UMA" = false ] && [ "$RUN_ISAC" = true ]; then
        # ISAC only: default to 200 seeds (0-199)
        SEED_END=199
    else
        # Both: we'll use different ranges per calibration type
        SEED_END=""
    fi
fi

# Print configuration
echo "========================================="
echo "Master Calibration Script"
echo "========================================="
if [ -n "$SEED_END" ]; then
    echo "Seeds: $SEED_START to $SEED_END ($(($SEED_END - $SEED_START + 1)) total)"
else
    echo "Seeds: Using defaults (UMa: 0-4, ISAC: 0-199)"
fi
echo "Run UMa: $RUN_UMA"
echo "Run ISAC: $RUN_ISAC"
echo "========================================="
echo ""

# Track start time
START_TIME=$(date +%s)

#=============================================================================
# UMa Calibration
#=============================================================================
if [ "$RUN_UMA" = true ]; then
    # Determine UMa seed range
    if [ -n "$SEED_END" ]; then
        UMA_SEED_START=$SEED_START
        UMA_SEED_END=$SEED_END
    else
        UMA_SEED_START=0
        UMA_SEED_END=4
    fi
    
    echo ""
    echo "========================================="
    echo "Starting UMa Calibration"
    echo "Seeds: $UMA_SEED_START to $UMA_SEED_END ($(($UMA_SEED_END - $UMA_SEED_START + 1)) seeds)"
    echo "========================================="
    
    # Phase 1: Large Scale Calibration
    echo ""
    echo ">>> UMa Phase 1: Large Scale Calibration"
    "$SCRIPT_DIR/run_sls_chan_multiseed.sh" \
        --seed-start $UMA_SEED_START \
        --seed-end $UMA_SEED_END \
        --config testBenches/chanModels/config/statistic_channel_config_phase1.yaml \
        --dataset uma_6ghz \
        --reference-json testBenches/chanModels/util/3gpp_calibration_phase1.json \
        --phase 1 \
        --output-dir uma_6ghz_phase1_results
    
    echo ""
    echo ">>> UMa Phase 1: Complete ✓"
    
    # Phase 2: Full Calibration
    echo ""
    echo ">>> UMa Phase 2: Full Calibration"
    "$SCRIPT_DIR/run_sls_chan_multiseed.sh" \
        --seed-start $UMA_SEED_START \
        --seed-end $UMA_SEED_END \
        --config testBenches/chanModels/config/statistic_channel_config_phase2.yaml \
        --dataset uma_6ghz \
        --reference-json testBenches/chanModels/util/3gpp_calibration_phase2.json \
        --phase 2 \
        --output-dir uma_6ghz_phase2_results
    
    echo ""
    echo ">>> UMa Phase 2: Complete ✓"
    echo ""
    echo "========================================="
    echo "UMa Calibration Complete"
    echo "========================================="
fi

#=============================================================================
# ISAC Calibration
#=============================================================================
if [ "$RUN_ISAC" = true ]; then
    # Determine ISAC seed range
    if [ -n "$SEED_END" ]; then
        ISAC_SEED_START=$SEED_START
        ISAC_SEED_END=$SEED_END
    else
        ISAC_SEED_START=0
        ISAC_SEED_END=199
    fi
    
    echo ""
    echo "========================================="
    echo "Starting ISAC Calibration"
    echo "Seeds: $ISAC_SEED_START to $ISAC_SEED_END ($(($ISAC_SEED_END - $ISAC_SEED_START + 1)) seeds)"
    echo "========================================="
    
    # Phase 1 Target Channel
    echo ""
    echo ">>> ISAC Phase 1 Target Channel"
    "$SCRIPT_DIR/run_sls_chan_multiseed.sh" \
        --seed-start $ISAC_SEED_START \
        --seed-end $ISAC_SEED_END \
        --config testBenches/chanModels/config/statistic_channel_config_isac_phase1.yaml \
        --dataset isac_uav_phase1_target \
        --reference-json testBenches/chanModels/util/3gpp_calibration_isac_uav_phase1.json \
        --phase 1 \
        --isac-channel target \
        --output-dir isac_phase1_target_results
    
    echo ""
    echo ">>> ISAC Phase 1 Target: Complete ✓"
    
    # Phase 1 Background Channel
    echo ""
    echo ">>> ISAC Phase 1 Background Channel"
    "$SCRIPT_DIR/run_sls_chan_multiseed.sh" \
        --seed-start $ISAC_SEED_START \
        --seed-end $ISAC_SEED_END \
        --config testBenches/chanModels/config/statistic_channel_config_isac_phase1_background.yaml \
        --dataset isac_uav_phase1_background \
        --reference-json testBenches/chanModels/util/3gpp_calibration_isac_uav_phase1.json \
        --phase 1 \
        --isac-channel background \
        --output-dir isac_phase1_background_results
    
    echo ""
    echo ">>> ISAC Phase 1 Background: Complete ✓"
    
    # Phase 2 Target Channel
    echo ""
    echo ">>> ISAC Phase 2 Target Channel"
    "$SCRIPT_DIR/run_sls_chan_multiseed.sh" \
        --seed-start $ISAC_SEED_START \
        --seed-end $ISAC_SEED_END \
        --config testBenches/chanModels/config/statistic_channel_config_isac_phase2.yaml \
        --dataset isac_uav_phase2_target \
        --reference-json testBenches/chanModels/util/3gpp_calibration_isac_uav_phase2.json \
        --phase 2 \
        --isac-channel target \
        --output-dir isac_phase2_target_results
    
    echo ""
    echo ">>> ISAC Phase 2 Target: Complete ✓"
    
    # Phase 2 Background Channel
    echo ""
    echo ">>> ISAC Phase 2 Background Channel"
    "$SCRIPT_DIR/run_sls_chan_multiseed.sh" \
        --seed-start $ISAC_SEED_START \
        --seed-end $ISAC_SEED_END \
        --config testBenches/chanModels/config/statistic_channel_config_isac_phase2_background.yaml \
        --dataset isac_uav_phase2_background \
        --reference-json testBenches/chanModels/util/3gpp_calibration_isac_uav_phase2.json \
        --phase 2 \
        --isac-channel background \
        --output-dir isac_phase2_background_results
    
    echo ""
    echo ">>> ISAC Phase 2 Background: Complete ✓"
    echo ""
    echo "========================================="
    echo "ISAC Calibration Complete"
    echo "========================================="
fi

#=============================================================================
# Summary
#=============================================================================
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo ""
echo "========================================="
echo "ALL CALIBRATIONS COMPLETE!"
echo "========================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

if [ "$RUN_UMA" = true ]; then
    echo "UMa Results:"
    echo "  - uma_6ghz_phase1_results/"
    echo "  - uma_6ghz_phase2_results/"
fi

if [ "$RUN_ISAC" = true ]; then
    echo "ISAC Results:"
    echo "  - isac_phase1_target_results/"
    echo "  - isac_phase1_background_results/"
    echo "  - isac_phase2_target_results/"
    echo "  - isac_phase2_background_results/"
fi

echo ""
echo "Check the CDF plots in each directory to verify calibration quality."
echo "========================================="
