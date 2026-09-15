#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#--------------------------------------------------------------------
#This script is to be run on the RU side
#--------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}

CONFIG_DIR=$cuBB_SDK

BUILD_DIR=build.$(uname -m)


show_usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  --build_dir <build-path>          Specify the relative path to build directory. (default: "$BUILD_DIR")"
  echo "  --config_dir <path>               Specify the path to the directory containing config files. (default: "\$cuBB_SDK")"
  echo "                                    the testBenches scripts will modify configuration files and write output files to this location"
  echo "  --gdb_script <script>             Specify the gdb script to use."
  echo "  --timeout <seconds>               Kill ru_emulator after seconds"
  echo "  --tv-base-path <path>, -t <path>  Specify the full path to the directory where test vectors are stored."
  echo "  --stage-local                     Stage this test's TV working set onto fast local storage (default"
  echo "                                    /dev/shm) before running, so load_tvs() reads locally instead of NFS."
  echo "                                    Idempotent: repeat runs of the same test re-stage nothing."
  echo "  --stage-dir <path>                Local staging directory for --stage-local (default: \${TV_STAGE_DIR:-/dev/shm/cubb_tv_stage})."
  echo "  -h, --help                        Show this help message."
  echo
  echo "Example:"
  echo "  $0 --build_dir build_dbg --tv-base-path /tmp/testVectors"
  echo
  echo "  to run RU-emulator in $cuBB_SDK/build_dbg path."
  echo "  Note that RU-emulator runs for channels set in test_config.sh (by default, all channels)."
  exit 1
}

GDB_SCRIPT=""
TIMEOUT=0
STAGE_LOCAL=0
STAGE_DIR=""

# Parse additional options
while [[ $# -gt 0 ]]; do
  case $1 in
    --build_dir=*)
      BUILD_DIR="${1#*=}"
      shift
      ;;
    --build_dir)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for $1 option"
        show_usage
      fi
      BUILD_DIR="$2"
      shift 2
      ;;
    --config_dir=*)
      CONFIG_DIR="${1#*=}"
      shift
      ;;
    --config_dir)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for $1 option"
        show_usage
        exit 1
      fi
      CONFIG_DIR="$2"
      shift 2
      ;;
    --gdb_script=*)
      GDB_SCRIPT="${1#*=}"
      shift
      ;;
    --gdb_script)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --gdb_script option"
        show_usage
        exit 1
      fi
      GDB_SCRIPT="$2"
      shift 2
      ;;
    --timeout=*)
      TIMEOUT="${1#*=}"
      shift
      ;;
    --timeout)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --timeout option"
        show_usage
        exit 1
      fi
      TIMEOUT="$2"
      shift 2
      ;;
    --tv-base-path=*)
      TV_BASE_PATH="${1#*=}"
      shift
      ;;
    --tv-base-path|-t)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for $1 option"
        show_usage
        exit 1
      fi
      TV_BASE_PATH="$2"
      shift 2
      ;;
    --stage-local)
      STAGE_LOCAL=1
      shift
      ;;
    --stage-dir=*)
      STAGE_DIR="${1#*=}"
      if [[ -z "$STAGE_DIR" ]]; then
        echo "Error: Missing value for --stage-dir option"
        show_usage
        exit 1
      fi
      STAGE_LOCAL=1
      shift
      ;;
    --stage-dir)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --stage-dir option"
        show_usage
        exit 1
      fi
      STAGE_DIR="$2"
      STAGE_LOCAL=1
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ ! -d $cuBB_SDK/"$BUILD_DIR" ]]; then
  echo "Error: Unable to access $cuBB_SDK/"$BUILD_DIR""
  exit 1
fi

# Check if TV_BASE_PATH is set and if 'multi-cell' exists inside it
if [[ -n "$TV_BASE_PATH" ]]; then
  if [[ ! -d "$TV_BASE_PATH/multi-cell" ]]; then
    echo "Error: 'multi-cell' directory for launch pattern files not found under '$TV_BASE_PATH'. You may use copy_test_files.sh to copy the required files to '$TV_BASE_PATH'."
    exit 1
  fi
fi

TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
if [[ ! -f $TEST_CONFIG_FILE ]]; then
    echo "$TEST_CONFIG_FILE is missing. Please run setup1_DU.sh and setup2_RU.sh first"
    exit 1
fi
source $TEST_CONFIG_FILE

#-------------------------------------------------------------------------------------------------------
#verify if test_config.sh has been run before running RU-emulator
if [[ ! -v TEST_CONFIG_DONE ]]; then
    echo "Error: Please run test_config.sh before executing the run scripts."
    exit 1
fi

#-------------------------------------------------------------------------------------------------------
# Optionally stage the TV working set onto fast local storage so ru_emulator's
# load_tvs() reads from local RAM/NVMe instead of the (~1Gbps-capped) NFS mount.
# PATTERN and NUM_CELLS come from test_config_summary.sh sourced above.
if [[ "$STAGE_LOCAL" -eq 1 ]]; then
    STAGE_ARGS=(--pattern "$PATTERN" --num-cells "$NUM_CELLS")
    [[ -n "$STAGE_DIR" ]] && STAGE_ARGS+=(--stage-dir "$STAGE_DIR")
    echo "Staging test vectors to local storage for faster load_tvs()..."
    if ! STAGE_OUT=$("$SCRIPT_DIR/stage_tvs_local.sh" "${STAGE_ARGS[@]}"); then
        echo "$STAGE_OUT"
        echo "Error: TV staging failed."
        exit 1
    fi
    echo "$STAGE_OUT"
    TV_BASE_PATH=$(echo "$STAGE_OUT" | sed -n 's/^TV_BASE_PATH=//p' | tail -1)
    if [[ -z "$TV_BASE_PATH" || ! -d "$TV_BASE_PATH/multi-cell" ]]; then
        echo "Error: staging did not produce a usable TV base path."
        exit 1
    fi
    echo "Using staged TV base path: $TV_BASE_PATH"
fi

#-------------------------------------------------------------------------------------------------------
#verify if setup2_RU.sh has been run before running RU-emulator
for ((p=0; p<${NUM_PORTS:-1}; p++)); do
    iface_var="RU_ETH_INTERFACE_${p}"
    mac_var="RU_MAC_ADDRESS_${p}"
    iface="${!iface_var}"
    expected_mac="${!mac_var}"
    if [[ -z "$iface" || -z "$expected_mac" ]]; then
        echo "Error: RU_ETH_INTERFACE_$p or RU_MAC_ADDRESS_$p not set. Please ensure setup1_DU.sh and setup2_RU.sh completed successfully."
        exit 1
    fi
    actual_mac=$(cat /sys/class/net/"${iface}"/address)
    if [ "$actual_mac" != "$expected_mac" ]; then
        echo "Error: MAC addresses do not match for interface $p. Expected $expected_mac (from config), but interface reports $actual_mac. Please ensure to run setup1_DU.sh and setup2_RU.sh before running run1_RU.sh"
        env | grep ADDR
        exit 1
    fi
done

#-------------------------------------------------------------------------------------------------------
# Pattern, channels, and cell topology from test_config_summary.sh.
CELL_TOPOLOGY="${CELL_TOPOLOGY:-${NUM_CELLS}C}"
IFS='_' read -ra CELL_TOPOLOGY_ARGS <<< "$CELL_TOPOLOGY"
if [ "$CHANNELS" == "all" ]; then
    CHANNELS=()
else
    CHANNELS=("--channels" "$CHANNELS")
fi

if [ $TIMEOUT -gt 0 ]; then
    WITH_TIMEOUT="timeout --kill-after=10 ${TIMEOUT}"
else
    WITH_TIMEOUT=""
fi

#-------------------------------------------------------------------------------------------------------
# use user-defined base path for for test vectors (if specified)
EXTRA_ARGS=()
if [[ -n "$TV_BASE_PATH" ]]; then
  # ru_emulator concatenates --tv verbatim with each TV filename (config_parser.cpp:
  # `user_defined_tv_base_path + tv_path`), so the base path MUST end with a slash or
  # it looks for e.g. /dev/shm/cubb_tv_stageTVnr_*.h5. Normalize to exactly one.
  EXTRA_ARGS+=(--tv "${TV_BASE_PATH%/}/" --lp "${TV_BASE_PATH%/}/multi-cell/")
fi
#-------------------------------------------------------------------------------------------------------
BASE_RU_YAML=$(basename "$RU_YAML")
if [[ "$CONTROLLER_MODE" == *nrSim_SCF* ]]; then
    NRSIM_TC=$(echo "$CONTROLLER_MODE" | sed -E 's/nrSim_SCF_(CG1_|SPRK_|MGX1_)?//')
    echo "$WITH_TIMEOUT $GDB_SCRIPT $cuBB_SDK/$BUILD_DIR/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator nrSim $NRSIM_TC" "${CHANNELS[@]}" "${EXTRA_ARGS[@]}"
    { sudo -E LD_BIND_NOW=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH} $WITH_TIMEOUT $GDB_SCRIPT "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator" nrSim $NRSIM_TC --config $BASE_RU_YAML "${CHANNELS[@]}" "${EXTRA_ARGS[@]}"; RET=$?; } || true
else
    echo "$WITH_TIMEOUT $GDB_SCRIPT $cuBB_SDK/$BUILD_DIR/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator F08 ${CELL_TOPOLOGY_ARGS[*]} $PATTERN --config $BASE_RU_YAML" "${CHANNELS[@]}" "${EXTRA_ARGS[@]}"
    # shellcheck disable=SC1073
    { sudo -E LD_BIND_NOW=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH} $WITH_TIMEOUT $GDB_SCRIPT "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator" F08 "${CELL_TOPOLOGY_ARGS[@]}" "$PATTERN" --config "$BASE_RU_YAML" "${CHANNELS[@]}" "${EXTRA_ARGS[@]}"; RET=$?; } || true
fi
exit $RET
