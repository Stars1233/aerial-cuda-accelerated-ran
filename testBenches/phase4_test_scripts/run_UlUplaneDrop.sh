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
PYTHON_SCRIPT="cuPHY-CP/cuphyoam/examples/aerial_ul_u_plane_drop.py"
SERVER_IP="localhost"
PORT="50052"    # Default port
CELL_ID=""
CHANNEL_ID="1"  # Default to PUSCH
DROP_RATE="0"   # Default to 0 (disabled)
SINGLE_DROP="0" # Default to 0 (continuous mode)
DROP_SLOT="0"   # Default to 0 (symbol mode)
FRAME_ID="0"    # Default frame ID [0-255]
SUBFRAME_ID="0" # Default subframe ID [0-9]
SLOT_ID="0"     # Default slot ID [0-1]

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --build_dir DIR     Specify build directory (default: build.$(uname -m))"
    echo "  --server_ip IP      OAM server IP address (default: localhost)"
    echo "  --port PORT         OAM server port (default: 50052)"
    echo "  --cell_id ID        Cell ID to send command to [0-19]"
    echo "  --channel_id ID     Channel ID [1-4] (default: 1)"
    echo "                      1: PUSCH"
    echo "                      2: PRACH"
    echo "                      3: PUCCH"
    echo "                      4: SRS"
    echo "  --drop_rate RATE    Drop rate [0-50] (default: 0, disabled)"
    echo "  --single_drop VAL   Single packet drop [0-1] (default: 0)"
    echo "  --drop_slot VAL     Drop entire slot [0-1] (default: 0)"
    echo "  --frame ID          Frame ID [0-255] (default: 0)"
    echo "  --subframe ID       Subframe ID [0-9] (default: 0)"
    echo "  --slot ID           Slot ID [0-1] (default: 0)"
    echo "  --help              Show this help message"
    echo
    echo "Example:"
    echo "  $0 --cell_id 0 --channel_id 1 --drop_rate 10     # Drop 10% of PUSCH packets for cell 0"
    echo "  $0 --cell_id 1 --channel_id 4 --single_drop 1    # Drop single SRS packet for cell 1"
    echo "  $0 --cell_id 0 --drop_rate 0                     # Disable packet dropping for cell 0"
    echo "  $0 --cell_id 0 --drop_slot 1 --drop_rate 20      # Drop 20% of random slots for cell 0"
    echo "  $0 --cell_id 0 --drop_slot 1 --frame 100 --subframe 5 --slot 1  # Drop specific slot"
    echo "  $0 --server_ip 192.168.1.100 --port 50052 --cell_id 0 --drop_rate 20  # Custom server and port"
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
        --channel_id)
            if ! [[ "$2" =~ ^[1-4]$ ]]; then
                echo "Error: channel_id must be between 1 and 4"
                exit 1
            fi
            CHANNEL_ID="$2"
            shift 2
            ;;
        --drop_rate)
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [ "$2" -gt 50 ]; then
                echo "Error: drop_rate must be between 0 and 50"
                exit 1
            fi
            DROP_RATE="$2"
            shift 2
            ;;
        --single_drop)
            if ! [[ "$2" =~ ^[0-1]$ ]]; then
                echo "Error: single_drop must be 0 or 1"
                exit 1
            fi
            SINGLE_DROP="$2"
            shift 2
            ;;
        --drop_slot)
            if ! [[ "$2" =~ ^[0-1]$ ]]; then
                echo "Error: drop_slot must be 0 or 1"
                exit 1
            fi
            DROP_SLOT="$2"
            shift 2
            ;;
        --frame)
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [ "$2" -gt 255 ]; then
                echo "Error: frame must be between 0 and 255"
                exit 1
            fi
            FRAME_ID="$2"
            shift 2
            ;;
        --subframe)
            if ! [[ "$2" =~ ^[0-9]+$ ]] || [ "$2" -gt 9 ]; then
                echo "Error: subframe must be between 0 and 9"
                exit 1
            fi
            SUBFRAME_ID="$2"
            shift 2
            ;;
        --slot)
            if ! [[ "$2" =~ ^[0-1]$ ]]; then
                echo "Error: slot must be 0 or 1"
                exit 1
            fi
            SLOT_ID="$2"
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
PYTHON_ARGS="--server_ip ${SERVER_IP} --port ${PORT} --cell_id ${CELL_ID} --channel_id ${CHANNEL_ID} \
             --drop_rate ${DROP_RATE} --single_drop ${SINGLE_DROP} --drop_slot ${DROP_SLOT} \
             --frame ${FRAME_ID} --subframe ${SUBFRAME_ID} --slot ${SLOT_ID}"

# Set up environment
export PYTHONPATH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/cuphyoam:${PYTHONPATH}"
export LD_LIBRARY_PATH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/cuphyoam:${LD_LIBRARY_PATH}"

# Execute Python script
echo "Starting UL U-plane drop command..."
echo "Running: python3 ${cuBB_SDK}/${PYTHON_SCRIPT} ${PYTHON_ARGS}"
python3 "${cuBB_SDK}/${PYTHON_SCRIPT}" ${PYTHON_ARGS}

# Check execution status
case $? in
    0)
        echo "UL U-plane drop command completed successfully"
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