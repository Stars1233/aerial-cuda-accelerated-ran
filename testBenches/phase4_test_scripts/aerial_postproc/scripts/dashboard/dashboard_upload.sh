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

#
# Unified dashboard upload script
#
# Takes an NCU report directory and uploads metrics to the NVDF dashboard.
# Handles uv dependencies, metadata creation, extraction, and upload.
#
# Usage:
#   ./dashboard_upload.sh <ncu_report_dir> [options]
#
# Examples:
#   ./dashboard_upload.sh /tmp/my_results --pipeline cuphy_bench_local --channel 0 --tv TVnr_7201_PUSCH
#   ./dashboard_upload.sh /tmp/my_results -p my_pipeline -c 1 -t TVnr_7201_PUSCH --upload
#

set -e

# =============================================================================
# Path constants (auto-detected from script location)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AERIAL_POSTPROC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CUBB_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Python scripts (alongside this script)
METRICS_EXTRACTION="$SCRIPT_DIR/metrics_extraction.py"
METRICS_UPLOAD="$SCRIPT_DIR/metrics_upload.py"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# NVIDIA internal PyPI index
NVIDIA_PYPI="https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-gpuwa-pypi/simple"

# =============================================================================
# Default settings
# =============================================================================
PIPELINE="${DASHBOARD_PIPELINE:-cuphy_bench_local}"
CHANNEL="${DASHBOARD_CHANNEL:-0}"
TEST_VECTOR="${DASHBOARD_TV:-}"
DO_UPLOAD=false
INDEX="${DASHBOARD_INDEX:-swgpu-aerial-perflab-cicd}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    echo "Usage: $0 <ncu_report_dir> [OPTIONS]"
    echo ""
    echo "Upload NCU profiling results to the NVDF dashboard."
    echo ""
    echo "Arguments:"
    echo "  ncu_report_dir       Directory containing report.ncu-rep"
    echo ""
    echo "Options:"
    echo "  -p, --pipeline NAME  Pipeline name (default: cuphy_bench_local)"
    echo "  -c, --channel N      Channel number: 0, 1, 2, ... (default: 0)"
    echo "  -t, --tv NAME        Test vector name (e.g., TVnr_7201_PUSCH)"
    echo "                       If not provided, extracted from directory name"
    echo "  -u, --upload         Actually upload to NVDF (default: extract only)"
    echo "  -i, --index NAME     NVDF index (default: swgpu-aerial-perflab-cicd)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Environment variables:"
    echo "  DASHBOARD_PIPELINE   Default pipeline name"
    echo "  DASHBOARD_CHANNEL    Default channel number"
    echo "  DASHBOARD_TV         Default test vector"
    echo "  DASHBOARD_INDEX      Default NVDF index"
    echo ""
    echo "Examples:"
    echo "  # Extract only (dry run)"
    echo "  $0 /tmp/my_results --pipeline cuphy_bench_local --channel 0 --tv TVnr_7201_PUSCH"
    echo ""
    echo "  # Extract and upload"
    echo "  $0 /tmp/my_results -p cuphy_bench_local -c 1 -t TVnr_7201_PUSCH --upload"
    echo ""
    echo "Test case format:"
    echo "  The test_case in NVDF is: run{channel}_{test_vector}"
    echo "  Example: run0_TVnr_7201_PUSCH (channel 0, TV 7201 PUSCH)"
    echo ""
    echo "View results at:"
    echo "  https://grafana.nvidia.com/d/db_phase2_engineering/engineering-view?orgId=146"
    exit 0
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pipeline)
            PIPELINE="$2"
            shift 2
            ;;
        -c|--channel)
            CHANNEL="$2"
            shift 2
            ;;
        -t|--tv)
            TEST_VECTOR="$2"
            shift 2
            ;;
        -u|--upload)
            DO_UPLOAD=true
            shift
            ;;
        -i|--index)
            INDEX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo -e "${RED}ERROR: Unknown option: $1${NC}"
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate arguments
if [ ${#POSITIONAL_ARGS[@]} -eq 0 ]; then
    echo -e "${RED}ERROR: No NCU report directory specified${NC}"
    echo ""
    usage
fi

NCU_DIR="${POSITIONAL_ARGS[0]}"

if [ ! -d "$NCU_DIR" ]; then
    echo -e "${RED}ERROR: Directory not found: $NCU_DIR${NC}"
    exit 1
fi

if [ ! -f "$NCU_DIR/report.ncu-rep" ]; then
    echo -e "${RED}ERROR: No report.ncu-rep found in: $NCU_DIR${NC}"
    echo "Make sure you ran: ncu --set full -o $NCU_DIR/report ..."
    exit 1
fi

# Try to extract test vector from directory name if not provided
if [ -z "$TEST_VECTOR" ]; then
    # Try to extract from directory name (e.g., run0_TVnr_7201_PUSCH_20260205)
    DIR_NAME=$(basename "$NCU_DIR")
    if [[ "$DIR_NAME" =~ (TVnr_[0-9]+_[A-Z]+) ]]; then
        TEST_VECTOR="${BASH_REMATCH[1]}"
        echo -e "${YELLOW}Auto-detected test vector: $TEST_VECTOR${NC}"
    else
        echo -e "${RED}ERROR: Could not detect test vector. Please specify with --tv${NC}"
        exit 1
    fi
fi

# Build test case name: run{channel}_{test_vector}
TEST_CASE="run${CHANNEL}_${TEST_VECTOR}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
# Use simple numeric ID to avoid parsing issues with hostnames containing hyphens
JENKINS_ID="1-${TIMESTAMP}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Dashboard Upload${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  NCU Directory:  $NCU_DIR"
echo "  Pipeline:       $PIPELINE"
echo "  Channel:        $CHANNEL"
echo "  Test Vector:    $TEST_VECTOR"
echo "  Test Case:      $TEST_CASE"
echo "  Upload:         $DO_UPLOAD"
if [ "$DO_UPLOAD" = true ]; then
    echo "  Index:          $INDEX"
fi
echo ""

# Step 1: Create/update metadata
echo -e "${CYAN}[1/3] Creating metadata...${NC}"
cat > "$NCU_DIR/metadata.txt" << EOF
jenkins_pipeline=${PIPELINE}
test_case=${TEST_CASE}
jenkins_id=${JENKINS_ID}
jenkins_test_result=PASS
EOF
echo "  Created: $NCU_DIR/metadata.txt"
cat "$NCU_DIR/metadata.txt" | sed 's/^/    /'
echo ""

# Step 2: Run extraction
echo -e "${CYAN}[2/3] Extracting metrics...${NC}"
PYTHONPATH="$AERIAL_POSTPROC_DIR:$PYTHONPATH" \
uv run --index "$NVIDIA_PYPI" --with-requirements "$REQUIREMENTS" \
    "$METRICS_EXTRACTION" "$NCU_DIR" --test_phase phase_2

if [ ! -f "$NCU_DIR/dashboard_data.json" ]; then
    echo -e "${RED}ERROR: Extraction failed${NC}"
    exit 1
fi
echo -e "  ${GREEN}Generated: $NCU_DIR/dashboard_data.json${NC}"
echo -e "  ${GREEN}Generated: $NCU_DIR/kernel_metrics.csv${NC}"
echo ""

# Step 3: Upload (if requested)
if [ "$DO_UPLOAD" = true ]; then
    echo -e "${CYAN}[3/3] Uploading to NVDF...${NC}"
    PYTHONPATH="$AERIAL_POSTPROC_DIR:$PYTHONPATH" \
    uv run --index "$NVIDIA_PYPI" --with-requirements "$REQUIREMENTS" \
        "$METRICS_UPLOAD" "$NCU_DIR" --test_phase phase_2 --index "$INDEX" --upload_opensearch
    
    echo -e "  ${GREEN}Uploaded to index: $INDEX${NC}"
else
    echo -e "${YELLOW}[3/3] Skipping upload (use --upload to enable)${NC}"
    echo ""
    echo "To upload manually:"
    echo "  $0 $NCU_DIR -p $PIPELINE -c $CHANNEL -t $TEST_VECTOR --upload"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output files:"
ls -la "$NCU_DIR"/*.json "$NCU_DIR"/*.csv "$NCU_DIR"/*.txt 2>/dev/null | sed 's/^/  /'
echo ""

if [ "$DO_UPLOAD" = true ]; then
    echo "View results at:"
    echo "  https://grafana.nvidia.com/d/db_phase2_engineering/engineering-view?orgId=146"
    echo ""
    echo "Select:"
    echo "  Pipeline: $PIPELINE"
    echo "  Channel:  $CHANNEL"
    echo "  Test Vector: $TEST_VECTOR"
fi
