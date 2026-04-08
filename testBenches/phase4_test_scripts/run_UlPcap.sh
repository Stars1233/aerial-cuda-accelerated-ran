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
PYTHON_SCRIPT="cuPHY-CP/cuphyoam/examples/aerial_ul_pcap_capture_next_crc.py"
SERVER_IP="localhost"
PORT=50051
CELL_ID=""
CMD=""
CELL_MASK=""

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --build_dir DIR     Specify build directory (default: build.$(uname -m))"
    echo "  --server_ip IP      OAM server IP address (default: localhost)"
    echo "  --port PORT         OAM server port (default: 50051)"
    echo "  --cell_id ID        Cell ID to send command to [0-19]"
    echo "  --cmd CMD           0: disable, 1: enable"
    echo "  --cell_mask MASK    Cell bitmask to set"
    echo "  --help              Show this help message"
    echo
    echo "Example:"
    echo "  $0 --cell_id 0 --cmd 1                  # Enable for cell 0"
    echo "  $0 --cell_mask 3                        # Set bitmask for cells 0 and 1"
    echo "  $0 --server_ip 192.168.1.100 --cmd 1    # Use specific server"
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
        --cmd)
            CMD="$2"
            shift 2
            ;;
        --cell_mask)
            CELL_MASK="$2"
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

if [ ! -z "$CELL_MASK" ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --cell_mask ${CELL_MASK}"
else
    if [ ! -z "$CELL_ID" ]; then
        PYTHON_ARGS="${PYTHON_ARGS} --cell_id ${CELL_ID}"
    fi
    if [ ! -z "$CMD" ]; then
        PYTHON_ARGS="${PYTHON_ARGS} --cmd ${CMD}"
    fi
fi

# Set up environment
export PYTHONPATH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/cuphyoam:${PYTHONPATH}"
export LD_LIBRARY_PATH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/cuphyoam:${LD_LIBRARY_PATH}"

# Execute Python script
echo "Starting UL PCAP capture..."
echo "Running: python3 ${cuBB_SDK}/${PYTHON_SCRIPT} ${PYTHON_ARGS}"
python3 "${cuBB_SDK}/${PYTHON_SCRIPT}" ${PYTHON_ARGS}

# Check execution status
if [ $? -ne 0 ]; then
    echo "Error: UL PCAP capture command send failed"
    exit 1
fi

echo "UL PCAP capture command sent successfully"