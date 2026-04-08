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

# Default settings
BUILD_DIR=${BUILD_DIR:-"build.$(uname -m)"}
PYTHON_SCRIPT="cuPHY-CP/cuphyoam/examples/aerial_ul_zero_uplane.py"
SERVER_IP="localhost"
PORT=50052
CELL_ID=""
CELL_MASK=""
CHANNEL_ID="1"  # Default channel set to 1

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --build_dir DIR     Specify build directory (default: build.$(uname -m))"
    echo "  --server_ip IP      OAM server IP address (default: localhost)"
    echo "  --port PORT         OAM server port (default: 50051)"
    echo "  --cell_id ID        Cell ID to send zero U-plane [0-19]"
    echo "  --cell_mask MASK    Cell bitmask to set"
    echo "  --channel_id ID     Channel ID [1-4] (default: 1)"
    echo "  --help              Show this help message"
    echo
    echo "Example:"
    echo "  $0 --cell_id 0                                  # Send zero U-plane for cell 0, channel 1"
    echo "  $0 --cell_mask 3                                # Send zero U-plane for cells 0 and 1, channel 1"
    echo "  $0 --cell_id 0 --channel_id 2                  # Send zero U-plane for cell 0, channel 2"
    echo "  $0 --server_ip 192.168.1.100 --cell_id 1       # Use specific server"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build_dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --server_ip)
            SERVER_IP="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --cell_id)
            CELL_ID="$2"
            shift 2
            ;;
        --cell_mask)
            CELL_MASK="$2"
            shift 2
            ;;
        --channel_id)
            # Validate channel_id range
            if ! [[ "$2" =~ ^[1-4]$ ]]; then
                echo "Error: channel_id must be between 1 and 4"
                exit 1
            fi
            CHANNEL_ID="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check Python3 availability
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required but not found"
    exit 1
fi

# Build Python command arguments
PYTHON_ARGS="--server_ip ${SERVER_IP} --port ${PORT}"

# Always include channel_id (now has default of 1)
PYTHON_ARGS="${PYTHON_ARGS} --channel_id ${CHANNEL_ID}"

# Add mutually exclusive cell_id or cell_mask
if [ ! -z "$CELL_MASK" ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --cell_mask ${CELL_MASK}"
elif [ ! -z "$CELL_ID" ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --cell_id ${CELL_ID}"
else
    echo "Error: Either --cell_id or --cell_mask must be specified"
    usage
    exit 1
fi

# Set up environment
export PYTHONPATH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/cuphyoam:${PYTHONPATH}"
export LD_LIBRARY_PATH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/cuphyoam:${LD_LIBRARY_PATH}"

# Execute Python script
echo "Starting zero U-plane command..."
echo "Running: python3 ${cuBB_SDK}/${PYTHON_SCRIPT} ${PYTHON_ARGS}"
python3 "${cuBB_SDK}/${PYTHON_SCRIPT}" ${PYTHON_ARGS}

# Check execution status
case $? in
    0)
        echo "Zero U-plane command completed successfully"
        ;;
    1)
        echo "Error: Server is unavailable"
        exit 1
        ;;
    2)
        echo "Error: Invalid arguments provided"
        exit 2
        ;;
    3)
        echo "Error: RPC communication failed"
        exit 3
        ;;
    4)
        echo "Error: Unexpected error occurred"
        exit 4
        ;;
    *)
        echo "Error: Unknown error occurred"
        exit 1
        ;;
esac