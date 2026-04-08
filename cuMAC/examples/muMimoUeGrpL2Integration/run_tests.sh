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

# Script to run muMimoUeGrp tests in separate tmux panes
# Usage: sudo ./run_tests.sh [build_dir]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default build directory
BUILD_DIR="${1:-/opt/nvidia/cuBB/build}"

# Session name
TMUX_SESSION="muMimoUeGrp_tests"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}ERROR: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}ERROR: tmux is not installed${NC}"
    echo -e "${YELLOW}Install with: sudo apt-get install tmux${NC}"
    exit 1
fi

# Function to print section headers
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Function to check if executable exists
check_executable() {
    local exe_path="$1"
    local exe_name="$2"
    
    if [ ! -f "$exe_path" ]; then
        echo -e "${RED}ERROR: $exe_name not found at: $exe_path${NC}"
        echo -e "${YELLOW}Please build the project first or specify correct build directory${NC}"
        echo -e "${YELLOW}Usage: sudo $0 [build_dir]${NC}"
        exit 1
    fi
    
    if [ ! -x "$exe_path" ]; then
        echo -e "${RED}ERROR: $exe_name is not executable: $exe_path${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Found: $exe_name${NC}"
}

# Paths to executables
CUMAC_TEST="$BUILD_DIR/cuMAC/examples/muMimoUeGrpL2Integration/cumac_muUeGrp_test"
L1_TEST="$BUILD_DIR/cuMAC/examples/muMimoUeGrpL2Integration/l1_muUeGrp_test"
L2_TEST="$BUILD_DIR/cuMAC/examples/muMimoUeGrpL2Integration/l2_muUeGrp_test"

# Check for config file
CONFIG_DIR="/opt/nvidia/cuBB/cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles"
CONFIG_FILE="$CONFIG_DIR/config.yaml"

print_header "Checking Prerequisites"

echo "Build directory: $BUILD_DIR"
echo "Config directory: $CONFIG_DIR"
echo "Tmux session: $TMUX_SESSION"
echo ""

# Check all executables exist
check_executable "$CUMAC_TEST" "cumac_muUeGrp_test"
check_executable "$L1_TEST" "l1_muUeGrp_test"
check_executable "$L2_TEST" "l2_muUeGrp_test"

# Check config file
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}✓ Found: config.yaml${NC}"
else
    echo -e "${YELLOW}WARNING: config.yaml not found at: $CONFIG_FILE${NC}"
fi

echo ""
echo -e "${GREEN}All prerequisites met!${NC}"

# Kill existing session if it exists
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo ""
    echo -e "${YELLOW}Killing existing tmux session: $TMUX_SESSION${NC}"
    tmux kill-session -t "$TMUX_SESSION"
    sleep 1
fi

print_header "Creating Tmux Session"

echo "This script will create a tmux session with three vertical panes:"
echo "  Left:   l1_muUeGrp_test     - L1 MU-MIMO UE Group Test (starts first - PRIMARY)"
echo "  Center: cumac_muUeGrp_test  - cuMAC MU-MIMO UE Group Test"
echo "  Right:  l2_muUeGrp_test     - L2 MU-MIMO UE Group Test"
echo ""
echo "Starting in 2 seconds..."
sleep 2

# Create new tmux session with first pane (cumac test)
echo -e "${BLUE}Creating tmux session...${NC}"
tmux new-session -d -s "$TMUX_SESSION" -n "muMimoUeGrp"

# Set pane titles (requires tmux 3.0+)
tmux set-option -t "$TMUX_SESSION" pane-border-status top
tmux set-option -t "$TMUX_SESSION" pane-border-format "#{pane_index}: #{pane_title}"

# Set the first pane title and prepare command (L1 runs first - PRIMARY for L1-cuMAC resources)
tmux select-pane -t "$TMUX_SESSION:0.0" -T "L1 Test"
tmux send-keys -t "$TMUX_SESSION:0.0" "echo '========================================'" C-m
tmux send-keys -t "$TMUX_SESSION:0.0" "echo 'L1 MU-MIMO UE Group Test'" C-m
tmux send-keys -t "$TMUX_SESSION:0.0" "echo '========================================'" C-m
tmux send-keys -t "$TMUX_SESSION:0.0" "echo 'Executable: $L1_TEST'" C-m
tmux send-keys -t "$TMUX_SESSION:0.0" "echo 'Start time: \$(date)'" C-m
tmux send-keys -t "$TMUX_SESSION:0.0" "echo ''" C-m

# Split window vertically for cuMAC test (creates pane 1)
tmux split-window -h -t "$TMUX_SESSION:0"
tmux select-pane -t "$TMUX_SESSION:0.1" -T "cuMAC Test"
tmux send-keys -t "$TMUX_SESSION:0.1" "echo '========================================'" C-m
tmux send-keys -t "$TMUX_SESSION:0.1" "echo 'cuMAC MU-MIMO UE Group Test'" C-m
tmux send-keys -t "$TMUX_SESSION:0.1" "echo '========================================'" C-m
tmux send-keys -t "$TMUX_SESSION:0.1" "echo 'Executable: $CUMAC_TEST'" C-m
tmux send-keys -t "$TMUX_SESSION:0.1" "echo 'Start time: \$(date)'" C-m
tmux send-keys -t "$TMUX_SESSION:0.1" "echo ''" C-m

# Split the right pane vertically for L2 test (creates pane 2)
tmux split-window -h -t "$TMUX_SESSION:0.1"
tmux select-pane -t "$TMUX_SESSION:0.2" -T "L2 Test"
tmux send-keys -t "$TMUX_SESSION:0.2" "echo '========================================'" C-m
tmux send-keys -t "$TMUX_SESSION:0.2" "echo 'L2 MU-MIMO UE Group Test'" C-m
tmux send-keys -t "$TMUX_SESSION:0.2" "echo '========================================'" C-m
tmux send-keys -t "$TMUX_SESSION:0.2" "echo 'Executable: $L2_TEST'" C-m
tmux send-keys -t "$TMUX_SESSION:0.2" "echo 'Start time: \$(date)'" C-m
tmux send-keys -t "$TMUX_SESSION:0.2" "echo ''" C-m

# Balance the panes to make them equal width
tmux select-layout -t "$TMUX_SESSION:0" even-horizontal

# Wait a moment for layout to settle
sleep 1

print_header "Launching Applications"

echo -e "${YELLOW}Note: Tests will run in tmux panes. Use these commands:${NC}"
echo "  - Attach to session:    tmux attach -t $TMUX_SESSION"
echo "  - Detach from session:  Press Ctrl+b then d"
echo "  - Kill session:         tmux kill-session -t $TMUX_SESSION"
echo "  - Navigate panes:       Ctrl+b then arrow keys"
echo "  - Scroll in pane:       Ctrl+b then [ (press q to exit scroll mode)"
echo ""

# Determine the build root directory (parent of BUILD_DIR if BUILD_DIR ends with /build)
BUILD_ROOT="${BUILD_DIR%/build}"
if [ "$BUILD_ROOT" == "$BUILD_DIR" ]; then
    # BUILD_DIR doesn't end with /build, so use it as is
    BUILD_ROOT="$BUILD_DIR"
fi

# Run the tests in each pane (change to build root for relative paths to work)
# IMPORTANT: Start L1 first - it is PRIMARY for L1-cuMAC semaphore and memory pools.
# cuMAC (SECONDARY) must attach after L1 creates the shared resources to avoid races/segfaults.
echo -e "${GREEN}Starting l1_muUeGrp_test in pane 0...${NC}"
tmux send-keys -t "$TMUX_SESSION:0.0" "cd $BUILD_ROOT && $L1_TEST" C-m

# Brief delay so L1 can create semaphores and memory pools before cuMAC attaches
sleep 2

echo -e "${GREEN}Starting cumac_muUeGrp_test in pane 1...${NC}"
tmux send-keys -t "$TMUX_SESSION:0.1" "cd $BUILD_ROOT && $CUMAC_TEST" C-m

echo -e "${GREEN}Starting l2_muUeGrp_test in pane 2...${NC}"
tmux send-keys -t "$TMUX_SESSION:0.2" "cd $BUILD_ROOT && $L2_TEST" C-m

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All applications launched in tmux!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Attaching to tmux session...${NC}"
echo -e "${YELLOW}(Press Ctrl+b then d to detach)${NC}"
echo ""
sleep 2

# Attach to the session
tmux attach -t "$TMUX_SESSION"

exit 0
