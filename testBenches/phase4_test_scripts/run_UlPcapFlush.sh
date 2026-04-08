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
PYTHON_SCRIPT="cuPHY-CP/cuphyoam/examples/aerial_ul_pcap_flush.py"
SERVER_IP="localhost"
PORT="50051"    # Default port
CELL_ID=""

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --build_dir DIR     Specify build directory (default: build.$(uname -m))"
    echo "  --server_ip IP      OAM server IP address (default: localhost)"
    echo "  --port PORT         OAM server port (default: 50051)"
    echo "  --cell_id ID        Cell ID to send command to [0-19]"
    echo "  --help              Show this help message"
    echo
    echo "Example:"
    echo "  $0 --cell_id 0                          # Flush PCAP for cell 0"
    echo "  $0 --server_ip 192.168.1.100 --cell_id 0 --port 50052  # Custom server and port"
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

# Validate required parameters
if [ -z "$CELL_ID" ]; then
    echo "Error: --cell_id is required"
    usage
    exit 1
fi

# Build Python command arguments
PYTHON_ARGS="--server_ip ${SERVER_IP} --port ${PORT} --cell_id ${CELL_ID}"

# Set up environment
export PYTHONPATH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/cuphyoam:${PYTHONPATH}"
export LD_LIBRARY_PATH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/cuphyoam:${LD_LIBRARY_PATH}"

# Execute Python script
echo "Starting UL PCAP flush command..."
echo "Running: python3 ${cuBB_SDK}/${PYTHON_SCRIPT} ${PYTHON_ARGS}"
python3 "${cuBB_SDK}/${PYTHON_SCRIPT}" ${PYTHON_ARGS}

# Check execution status
case $? in
    0)
        echo "UL PCAP flush command completed successfully"
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
        echo "Error: RPC communication failed, PCAP bitmask for the cell is not set"
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