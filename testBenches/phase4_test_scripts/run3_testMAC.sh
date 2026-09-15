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

valid_channels=("PUSCH" "PDSCH" "PDCCH_UL" "PDCCH_DL" "PBCH" "PUCCH" "PRACH" "CSI_RS" "SRS" "BFW_DL" "BFW_UL" "all")

normalize_channels() {
  local channels="$1"

  if [ "$channels" == "all" ]; then
    echo "all"
    return 0
  fi

  if [[ "$channels" =~ ^0x[0-9A-Fa-f]+$ ]]; then
    local channels_dec
    channels_dec=$(printf "%d" "$channels")
    if [[ $channels_dec -gt 2047 ]]; then
      echo "Error: Invalid channel bitmask '$channels'. Max value 0x7FF" >&2
      return 1
    fi
    echo "$channels"
    return 0
  fi

  IFS='+,' read -ra channel_list <<< "$channels"
  local new_channels=""
  local channel

  for channel in "${channel_list[@]}"; do
    if [[ ! " ${valid_channels[*]} " =~ " $channel " ]]; then
      echo "Error: Invalid channel '$channel' found in --channels." >&2
      echo "List of valid channels: ${valid_channels[*]}" >&2
      return 1
    fi
    if [ -n "$new_channels" ]; then
      new_channels="$new_channels"+"$channel"
    else
      new_channels="$channel"
    fi
  done

  echo "$new_channels"
}

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
  echo "  --ru_emulator_host <host> Hostname of the RU emulator; if set, test_mac waits for RU readiness before first cell_init"
  echo "  --channels <channel_names> OR <bit_mask>"
  echo "                            Override channels from test_config.sh for this run only."
  echo "                            Channel names may be separated by ',' or '+' (e.g. PDSCH+PDCCH_DL+PBCH+CSI_RS)."
  echo "                            Alternatively, specify a hex bit-mask (e.g. 0x29A)."
  echo "                            Use 'all' to run all channels."
  echo "  -h, --help                Show this help message."
  echo
  echo "Example:"
  echo "  $0 --build_dir build_dbg"
  echo "  $0 --ml2 0"
  echo "  $0 --ml2 1"
  echo "  $0 --channels PDSCH+PDCCH_DL+PBCH+CSI_RS+BFW_DL"
  echo
  echo "  to run test_mac in $cuBB_SDK/build_dbg path."
  echo "  By default, test_mac runs for channels set in test_config.sh (all channels if unset there)."
  exit 1
}

TIMEOUT=0
GDB_SCRIPT=""
CELL_MASK=()
CONFIG_YAML=()
ML2_INSTANCE=""
RU_EMULATOR_HOST=""
CHANNELS_ARG=""

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
    --ru_emulator_host=*)
      RU_EMULATOR_HOST="${1#*=}"
      shift
      ;;
    --ru_emulator_host)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --ru_emulator_host option"
        show_usage
        exit 1
      fi
      RU_EMULATOR_HOST="$2"
      shift 2
      ;;
    --channels=*)
      CHANNELS_ARG="${1#*=}"
      shift
      ;;
    --channels)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --channels option"
        show_usage
        exit 1
      fi
      CHANNELS_ARG="$2"
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
            CELL_MASK=(--cells "$ML2_CELL_MASK0")
            echo "Using Multi-L2 instance 0: CELL_MASK=$ML2_CELL_MASK0"
        else
            echo "Error: ML2_CELL_MASK0 not found in test_config_summary.sh. Multi-L2 mode may not be configured."
            exit 1
        fi
    elif [[ "$ML2_INSTANCE" == "1" ]]; then
        if [[ -n "$ML2_CELL_MASK1" ]]; then
            CELL_MASK=(--cells "$ML2_CELL_MASK1")
            echo "Using Multi-L2 instance 1: CELL_MASK=$ML2_CELL_MASK1"
        else
            echo "Error: ML2_CELL_MASK1 not found in test_config_summary.sh. Multi-L2 mode may not be configured."
            exit 1
        fi
        if [[ -n "$TESTMAC1_YAML" ]]; then
            mac_yaml_file="${TESTMAC1_YAML##*/}"
            CONFIG_YAML=(--config "$mac_yaml_file")
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
# Verify interface MAC addresses match config
for ((p=0; p<${NUM_PORTS:-1}; p++)); do
    iface_var="DU_ETH_INTERFACE_${p}"
    mac_var="DU_MAC_ADDRESS_${p}"
    iface="${!iface_var}"
    expected_mac="${!mac_var}"
    if [[ -z "$iface" || -z "$expected_mac" ]]; then
        echo "Error: DU_ETH_INTERFACE_$p or DU_MAC_ADDRESS_$p not set. Please ensure setup1_DU.sh and setup2_RU.sh completed successfully."
        exit 1
    fi
    actual_mac=$(cat /sys/class/net/"${iface}"/address)
    if [ "$actual_mac" != "$expected_mac" ]; then
        echo "Error: MAC addresses do not match for interface $p. Expected $expected_mac (from config), but interface reports $actual_mac. Please ensure to run setup1_DU.sh and setup2_RU.sh before running run3_testMAC.sh"
        exit 1
    fi
done

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
# Pattern, channels, and cell topology from test_config_summary.sh.
if [[ -n "$CHANNELS_ARG" ]]; then
    CHANNELS=$(normalize_channels "$CHANNELS_ARG") || exit 1
    echo "Using command-line channel override: $CHANNELS"
fi

CELL_TOPOLOGY="${CELL_TOPOLOGY:-${NUM_CELLS}C}"
IFS='_' read -ra CELL_TOPOLOGY_ARGS <<< "$CELL_TOPOLOGY"
if [ "$CHANNELS" == "all" ]; then
    CHANNELS=()
else
    CHANNELS=("--channels" "$CHANNELS")
fi


if [ "$TIMEOUT" -gt 0 ]; then
    WITH_TIMEOUT=(timeout --kill-after=10 "$TIMEOUT")
else
    WITH_TIMEOUT=()
fi

GDB_SCRIPT_ARGS=()
if [[ -n "$GDB_SCRIPT" ]]; then
    GDB_SCRIPT_ARGS=("$GDB_SCRIPT")
fi

RU_HOST_ARG=()
if [[ -n "$RU_EMULATOR_HOST" ]]; then
    RU_HOST_ARG=(--ru_emulator_host "$RU_EMULATOR_HOST")
fi

print_command() {
    printf '%q ' "$@"
    echo
}

#-------------------------------------------------------------------------------------------------------
if [[ "$CONTROLLER_MODE" == *nrSim_SCF* ]]; then
    NRSIM_TC=$(echo "$CONTROLLER_MODE" | sed -E 's/nrSim_SCF_(CG1_|SPRK_|MGX1_)?//')
    print_command "${WITH_TIMEOUT[@]}" stdbuf --output=L "${GDB_SCRIPT_ARGS[@]}" "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/testMAC/testMAC/test_mac" nrSim "$NRSIM_TC" "${CHANNELS[@]}" "${CELL_MASK[@]}" "${CONFIG_YAML[@]}" "${RU_HOST_ARG[@]}"
    { sudo -E LD_BIND_NOW=1 "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" "${WITH_TIMEOUT[@]}" stdbuf --output=L "${GDB_SCRIPT_ARGS[@]}" "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/testMAC/testMAC/test_mac" nrSim "$NRSIM_TC" "${CHANNELS[@]}" "${CELL_MASK[@]}" "${CONFIG_YAML[@]}" "${RU_HOST_ARG[@]}"; RET=$?; } || true
else
    print_command "${WITH_TIMEOUT[@]}" stdbuf --output=L "${GDB_SCRIPT_ARGS[@]}" "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/testMAC/testMAC/test_mac" F08 "${CELL_TOPOLOGY_ARGS[@]}" "$PATTERN" "${CHANNELS[@]}" "${CELL_MASK[@]}" "${CONFIG_YAML[@]}" "${RU_HOST_ARG[@]}"
    { sudo -E LD_BIND_NOW=1 "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" "${WITH_TIMEOUT[@]}" stdbuf --output=L "${GDB_SCRIPT_ARGS[@]}" "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/testMAC/testMAC/test_mac" F08 "${CELL_TOPOLOGY_ARGS[@]}" "$PATTERN" "${CHANNELS[@]}" "${CELL_MASK[@]}" "${CONFIG_YAML[@]}" "${RU_HOST_ARG[@]}"; RET=$?; } || true
fi
exit $RET
