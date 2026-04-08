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
#This script is to be run on DU side for testMAC
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
  echo "  --build_dir <build-path>  Specify the relative path to build directory. (default: "$BUILD_DIR")"
  echo "  --config_dir <path>       Specify the path to the directory containing config files. (default: "\$cuBB_SDK")"
  echo "                            the testBenches scripts will modify configuration files and write output files to this location"
  echo "  --timeout <seconds>       Kill test_mac after seconds"
  echo "  --gdb_script <script>     Specify the gdb script to use."
  echo "  --ml2 <0|1>               Select Multi-L2 instance: 0 for first instance (ML2_CELL_MASK0), 1 for second (ML2_CELL_MASK1 + TESTMAC1_YAML)"
  echo "  -h, --help                Show this help message."
  echo
  echo "Example:"
  echo "  $0 --build_dir build_dbg"
  echo "  $0 --ml2 0"
  echo "  $0 --ml2 1"
  echo
  echo "  to run test_mac in $cuBB_SDK/build_dbg path."
  echo "  Note that test_mac runs for channels set in test_config.sh (by default, all channels)."
  exit 1
}

TIMEOUT=0
GDB_SCRIPT=""
CELL_MASK=""
CONFIG_YAML=""
ML2_INSTANCE=""

# Parse additional options
while [[ $# -gt 0 ]]; do
  case $1 in
    --build_dir=*)
      BUILD_DIR="${1#*=}"
      shift
      ;;
    --build_dir)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --build_dir option"
        show_usage
        exit 1
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
    --ml2=*)
      ML2_INSTANCE="${1#*=}"
      if [[ "$ML2_INSTANCE" != "0" && "$ML2_INSTANCE" != "1" ]]; then
        echo "Error: Invalid value for --ml2 option. Must be 0 or 1"
        show_usage
        exit 1
      fi
      shift
      ;;
    --ml2)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --ml2 option"
        show_usage
        exit 1
      fi
      ML2_INSTANCE="$2"
      if [[ "$ML2_INSTANCE" != "0" && "$ML2_INSTANCE" != "1" ]]; then
        echo "Error: Invalid value for --ml2 option. Must be 0 or 1"
        show_usage
        exit 1
      fi
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
  echo "Error: Unable to access $cuBB_SDK/$BUILD_DIR"
  exit 1
fi

TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
if [[ ! -f $TEST_CONFIG_FILE ]]; then
    echo "$TEST_CONFIG_FILE is missing. Please run setup1_DU.sh and setup2_RU.sh first"
fi
[ -f $TEST_CONFIG_FILE ] && source $TEST_CONFIG_FILE
#verify if setup1_DU.sh has been run before running testMAC
while [[ ! -v TEST_CONFIG_DONE ]]; do
    echo "Error: Please run test_config.sh before executing the run scripts. Retrying in 5 seconds."
    sleep 5
    [ -f $TEST_CONFIG_FILE ] && source $TEST_CONFIG_FILE
done

# Apply Multi-L2 configuration if --ml2 option is specified
if [[ -n "$ML2_INSTANCE" ]]; then
    if [[ "$MULTI_L2" == false ]]; then
        echo "Error: Multi-L2 mode is disabled. Please run test_config.sh with --ml2 to enable Multi-L2 mode."
        exit 1
    fi
    if [[ "$ML2_INSTANCE" == "0" ]]; then
        if [[ -n "$ML2_CELL_MASK0" ]]; then
            CELL_MASK="--cells $ML2_CELL_MASK0"
            echo "Using Multi-L2 instance 0: CELL_MASK=$ML2_CELL_MASK0"
        else
            echo "Error: ML2_CELL_MASK0 not found in test_config_summary.sh. Multi-L2 mode may not be configured."
            exit 1
        fi
    elif [[ "$ML2_INSTANCE" == "1" ]]; then
        if [[ -n "$ML2_CELL_MASK1" ]]; then
            CELL_MASK="--cells $ML2_CELL_MASK1"
            echo "Using Multi-L2 instance 1: CELL_MASK=$ML2_CELL_MASK1"
        else
            echo "Error: ML2_CELL_MASK1 not found in test_config_summary.sh. Multi-L2 mode may not be configured."
            exit 1
        fi
        if [[ -n "$TESTMAC1_YAML" ]]; then
            mac_yaml_file="${TESTMAC1_YAML##*/}"
            CONFIG_YAML="--config $mac_yaml_file"
            echo "Using Multi-L2 instance 1: CONFIG=$mac_yaml_file"
        else
            echo "Error: TESTMAC1_YAML not found in test_config_summary.sh. Multi-L2 mode may not be configured."
            exit 1
        fi
    else
        echo "Error: Invalid value for --ml2 option. Must be 0 or 1"
        show_usage
        exit 1
    fi
else
    if [[ "$MULTI_L2" == true ]]; then
        echo "Error: Multi-L2 mode is enabled. --ml2 option is required to select L2 instance"
        show_usage
        exit 1
    fi
fi

#-------------------------------------------------------------------------------------------------------
# Check first interface (always required)
ACTUAL_DU_MAC_ADDRESS_0=$(cat /sys/class/net/"${DU_ETH_INTERFACE_0}"/address)
if [ "$ACTUAL_DU_MAC_ADDRESS_0" != "$DU_MAC_ADDRESS_0" ]; then
    echo "Error: MAC addresses do not match for interface 0. Expected $ACTUAL_DU_MAC_ADDRESS_0, but reading $DU_MAC_ADDRESS_0 from logs. Please ensure to run setup1_DU.sh and setup2_RU.sh before running run3_testMAC.sh"
    exit 1
fi

# Check second interface if running in 2-port mode
if [ "${NUM_PORTS:-1}" -eq 2 ]; then
    ACTUAL_DU_MAC_ADDRESS_1=$(cat /sys/class/net/"${DU_ETH_INTERFACE_1}"/address)
    if [ "$ACTUAL_DU_MAC_ADDRESS_1" != "$DU_MAC_ADDRESS_1" ]; then
        echo "Error: MAC addresses do not match for interface 1. Expected $ACTUAL_DU_MAC_ADDRESS_1, but reading $DU_MAC_ADDRESS_1 from logs. Please ensure to run setup1_DU.sh and setup2_RU.sh before running run3_testMAC.sh"
        exit 1
    fi
fi

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# pattern, channels and number of cells from test_config_summary.sh
NUM_CELLS="${NUM_CELLS}C"
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
if [[ "$CONTROLLER_MODE" == *nrSim_SCF* ]]; then
    NRSIM_TC=$(echo "$CONTROLLER_MODE" | sed -E 's/nrSim_SCF_(CG1_)?//')
    echo "$WITH_TIMEOUT stdbuf --output=L $GDB_SCRIPT $cuBB_SDK/$BUILD_DIR/cuPHY-CP/testMAC/testMAC/test_mac nrSim $NRSIM_TC" "${CHANNELS[@]}" "$CELL_MASK" "$CONFIG_YAML"
    { sudo -E LD_BIND_NOW=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH} $WITH_TIMEOUT stdbuf --output=L $GDB_SCRIPT "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/testMAC/testMAC/test_mac" nrSim $NRSIM_TC "${CHANNELS[@]}" $CELL_MASK $CONFIG_YAML; RET=$?; } || true
else
    echo "$WITH_TIMEOUT stdbuf --output=L $GDB_SCRIPT $cuBB_SDK/$BUILD_DIR/cuPHY-CP/testMAC/testMAC/test_mac F08 $NUM_CELLS $PATTERN" "${CHANNELS[@]}" "$CELL_MASK" "$CONFIG_YAML"
    { sudo -E LD_BIND_NOW=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH} $WITH_TIMEOUT stdbuf --output=L $GDB_SCRIPT "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/testMAC/testMAC/test_mac" F08 $NUM_CELLS $PATTERN "${CHANNELS[@]}" $CELL_MASK $CONFIG_YAML; RET=$?; } || true
fi
exit $RET

