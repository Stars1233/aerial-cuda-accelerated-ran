#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Script to run parse_test_config_params.py on all major patterns and display the results

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Array of test patterns (removing F prefix and CCDF as requested)
patterns=(
    "F08_20C_59c_BFP9_STT455000_EH_GC_1P"
    "F08_20C_59c_BFP9_STT455000_EH_1P"
    "F08_20C_59e_BFP9_STT455000_EH_1P"
    "F08_2C_59c_BFP14_STT455000_1P"
    "F08_2C_59b_BFP14_STT455000_EH_1P"
    "F08_3C_66c_BFP9_STT480000_EH_1P"
    "F08_3C_67c_BFP9_STT480000_EH_1P"
    "F08_6C_69_MODCOMP_STT480000_1P"
    "F08_6C_79_MODCOMP_STT480000_EH_1P"
    "F08_6C_69b_MODCOMP_STT480000_EH_1P"
    "F08_6C_79_MODCOMP_STT480000_EH_GC_1P"
    "F08_6C_69_MODCOMP_STT480000_NICD_EH_1P"
    "F08_6C_79_MODCOMP_STT480000_NOPOST_1P"
    "F08_6C_69_MODCOMP_STT480000_NICD_EH_TYPO_1P"
)

# Default host configuration
HOST_CONFIG="${1:-CG1_R750}"

echo "Running parse_test_config_params.py on all major patterns with host config: $HOST_CONFIG"
echo "================================================================================"
echo ""

# Create a temporary directory for output files
TEMP_DIR=$(mktemp -d)
echo "Temporary output directory: $TEMP_DIR"
echo ""

# Process each pattern
for pattern in "${patterns[@]}"; do
    echo "=================================================================================="
    echo "PATTERN: $pattern"
    echo "=================================================================================="
    
    # Generate output filename based on pattern
    output_file="$TEMP_DIR/${pattern//\//_}.sh"
    
    # Run the parser
    echo "Running: ./parse_test_config_params.sh \"$pattern\" \"$HOST_CONFIG\" \"$output_file\""
    "$SCRIPT_DIR/parse_test_config_params.sh" "$pattern" "$HOST_CONFIG" "$output_file" 2>&1
    
    # Display the generated configuration
    if [ -f "$output_file" ]; then
        echo ""
        echo "Generated configuration:"
        echo "------------------------"
        cat "$output_file"
        echo ""
        
        # Clear any existing parameter variables to ensure clean state
        unset COPY_TEST_FILES_PARAMS BUILD_AERIAL_PARAMS SETUP1_DU_PARAMS SETUP2_RU_PARAMS
        unset TEST_CONFIG_PARAMS RUN1_RU_PARAMS RUN2_CUPHYCONTROLLER_PARAMS RUN3_TESTMAC_PARAMS
        
        # Source the file and show key parameters
        source "$output_file"
        echo "Key parameters:"
        echo "  Pattern: $(echo "$TEST_CONFIG_PARAMS" | awk '{print $1}')"
        echo "  Cells: $(echo "$TEST_CONFIG_PARAMS" | grep -o -- '--num-cells=[0-9]*' | cut -d= -f2)"
        echo "  Compression: $(echo "$TEST_CONFIG_PARAMS" | grep -o -- '--compression=[0-9]*' | cut -d= -f2)"
        echo "  BFP: $(echo "$TEST_CONFIG_PARAMS" | grep -o -- '--BFP=[0-9]*' | cut -d= -f2 || echo "N/A")"
        echo "  Early HARQ: $(echo "$TEST_CONFIG_PARAMS" | grep -o -- '--ehq=[0-9]*' | cut -d= -f2)"
        echo "  Green Context: $(echo "$TEST_CONFIG_PARAMS" | grep -o -- '--green-ctx=[0-9]*' | cut -d= -f2)"
        echo "  STT: $(echo "$TEST_CONFIG_PARAMS" | grep -o -- '--STT=[0-9]*' | cut -d= -f2)"
        echo "  MUMIMO: $(echo "$SETUP1_DU_PARAMS" | grep -q -- '--mumimo 1' && echo "1" || echo "0")"
        echo "  Build MODCOMP: $(echo "$BUILD_AERIAL_PARAMS" | grep -q -- '--modcomp 1' && echo "1" || echo "0")"
    else
        echo "ERROR: Failed to generate configuration file"
    fi
    
    echo ""
done

echo "=================================================================================="
echo "SUMMARY"
echo "=================================================================================="
echo "Total patterns processed: ${#patterns[@]}"
echo "Output files saved in: $TEMP_DIR"
echo ""
echo "To clean up temporary files, run: rm -rf $TEMP_DIR"
echo ""

# Option to compare all generated TEST_CONFIG_PARAMS
echo "All TEST_CONFIG_PARAMS values:"
echo "------------------------------"
for pattern in "${patterns[@]}"; do
    output_file="$TEMP_DIR/${pattern//\//_}.sh"
    if [ -f "$output_file" ]; then
        # Clear any existing parameter variables to ensure clean state
        unset COPY_TEST_FILES_PARAMS BUILD_AERIAL_PARAMS SETUP1_DU_PARAMS SETUP2_RU_PARAMS
        unset TEST_CONFIG_PARAMS RUN1_RU_PARAMS RUN2_CUPHYCONTROLLER_PARAMS RUN3_TESTMAC_PARAMS
        
        source "$output_file"
        printf "%-50s: %s\n" "$pattern" "$TEST_CONFIG_PARAMS"
    else
        printf "%-50s: %s\n" "$pattern" "FAILED - No output file generated"
    fi
done
