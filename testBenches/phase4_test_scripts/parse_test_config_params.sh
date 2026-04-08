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

# Shell wrapper for parse_test_config_params.py
# This script generates a parameter file that can be sourced to set environment variables
# Usage: ./parse_test_config_params.sh <test_case> <host_config> <output_file> [options]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if at least 3 arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <test_case_string> <host_config> <output_file> [python_script_options]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 \"F08_6C_69_MODCOMP_STT480000_1P\" \"CG1_R750\" test_params.sh" >&2
    echo "  $0 \"F08_20C_59c_BFP14_EH_GC_2P\" \"GL4_R750\" test_params.sh --custom-build-dir build.debug"  >&2
    echo "" >&2
    echo "This script generates a parameter file that must be sourced to set environment variables:" >&2
    echo "  source test_params.sh" >&2
    echo "" >&2
    echo "After sourcing the file, you can use:" >&2
    echo "  ./copy_test_files.sh \$COPY_TEST_FILES_PARAMS" >&2
    echo "  ./build_aerial_sdk.sh \$BUILD_AERIAL_PARAMS" >&2
    echo "  ./setup1_DU.sh \$SETUP1_DU_PARAMS" >&2
    echo "  ./setup2_RU.sh \$SETUP2_RU_PARAMS" >&2
    echo "  ./test_config.sh \$TEST_CONFIG_PARAMS" >&2
    echo "  ./run1_RU.sh \$RUN1_RU_PARAMS" >&2
    echo "  ./run2_cuPHYcontroller.sh \$RUN2_CUPHYCONTROLLER_PARAMS" >&2
    echo "  ./run3_testMAC.sh \$RUN3_TESTMAC_PARAMS" >&2
    exit 1
fi

# Extract the first three arguments
TEST_CASE="$1"
HOST_CONFIG="$2"
OUTPUT_FILE="$3"
shift 3

# Call the Python script with output file option
OUTPUT=$(python3 "$SCRIPT_DIR/parse_test_config_params.py" "$TEST_CASE" "$HOST_CONFIG" -o "$OUTPUT_FILE" "$@" 2>&1)

# Check if the Python script succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to parse test configuration" >&2
    echo "$OUTPUT" >&2
    exit 1
fi

# Print success message
echo "$OUTPUT"
echo "" >&2
echo "Successfully generated parameter file: $OUTPUT_FILE" >&2
echo "To use these parameters, run: source $OUTPUT_FILE" >&2
echo "Note: Consider clearing any existing variables before sourcing to ensure a clean state" >&2
