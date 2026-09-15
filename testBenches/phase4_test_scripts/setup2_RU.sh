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

# Set cuBB SDK path
cuBB_SDK=$(realpath $SCRIPT_DIR/../..)
CONFIG_DIR=$cuBB_SDK
CONFIG_DIR_SET=false

# Maximum number of NIC ports supported (must match setup1_DU.sh / MAX_NUM_OF_NIC_PORT_SUPPORTED).
# MGX ARC Pro: up to 3x CX8 NICs × 8 ports/NIC = 24 ports.
MAX_PORTS=24

#==============================================================
# Default Values
#==============================================================
# Network interface defaults (supports up to MAX_PORTS)
RU_ETH_INTERFACE_0="aerial00"
RU_ETH_INTERFACE_1="aerial01"

# Configuration defaults
RU_CFG="config"

show_usage() {
    echo "Usage: $0 [options]"
    echo "Setup script for RU to enable running cuBB test"
    echo
    echo "Options:"
    echo "  --help         , -h       Show this help message and exit"
    echo
    echo "  --ru-ethN=INTERFACE                Set RU network interface for port N (N=0..$((MAX_PORTS-1)))"
    echo "                                     Default: aerial00 (N=0), aerial01 (N=1)"
    echo "                                     e.g. --ru-eth0=aerial00 --ru-eth1=aerial01 --ru-eth2=aerial02"
    echo
    echo "  --cubb-sdk=PATH                     Set cuBB SDK path"
    echo "                                     Default: auto-detect (../../ from script dir)"
    echo
    echo "  --ru-yaml=Y, -y Y        Set ru-emulator YAML file"
    echo "                           Default: $RU_CFG.yaml"
    echo 
    echo "  --core-config-input-override=YAML  Override logical core configuration file"
    echo "                                     example: --core-config-input-override=l1_logical_cores_DU_CG1_RU_CG1.yaml"
    echo "                                     Default: unset"
    echo
    echo "  --config_dir <path>      Specify the path to the directory containing config files. (default: "\$cuBB_SDK")"
    echo "                           the testBenches scripts will modify configuration files and write output files to this location"
    echo
    echo
    echo "Example:"
    echo "  $0 --ru-eth0=aerial00 --ru-eth1=aerial01"
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_usage
            exit 0
            ;;
        --ru-eth[0-9]*)
            if [[ "$1" =~ ^--ru-eth([0-9]+)=(.+)$ ]]; then
                idx=$((10#${BASH_REMATCH[1]}))
                if [[ "$idx" -ge "$MAX_PORTS" ]]; then
                    echo "Error: Port index $idx exceeds maximum ($MAX_PORTS)"
                    exit 1
                fi
                declare "RU_ETH_INTERFACE_${idx}=${BASH_REMATCH[2]}"
            elif [[ "$1" =~ ^--ru-eth([0-9]+)$ ]]; then
                idx=$((10#${BASH_REMATCH[1]}))
                if [[ "$idx" -ge "$MAX_PORTS" ]]; then
                    echo "Error: Port index $idx exceeds maximum ($MAX_PORTS)"
                    exit 1
                fi
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: Missing value for $1 option"
                    show_usage
                    exit 1
                fi
                eval "RU_ETH_INTERFACE_${idx}=\"$2\""
                shift
            else
                echo "Error: Malformed option: $1"
                show_usage
                exit 1

            fi
            shift
            ;;
        --core-config-input-override=*)
            CORE_CONFIG_INPUT_OVERRIDE="${1#*=}"
            shift
            ;;
        --core-config-input-override)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1 option"
                show_usage
                exit 1
            fi
            CORE_CONFIG_INPUT_OVERRIDE="$2"
            shift 2
            ;;
        --ru-yaml*)
            if [[ "$1" == *"="* ]]; then
                # Option with equals sign
                option="${1%%=*}"
                value="${1#*=}"
                if [[ -z "$value" ]]; then
                    echo "Error: Missing value for $option option"
                    exit 1
                fi
            else
                # Option without equals sign
                option="$1"
                if [[ -z "$2" || "$2" == -* ]]; then
                    echo "Error: Missing value for $option option"
                    exit 1
                fi
                value="$2"
                shift
            fi
            case $option in
                --ru-yaml) RU_CFG="$value" ;;
                *) echo "Unknown option: $option"; exit 1 ;;
            esac
            shift
            ;;
        -y)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1 option"
                exit 1
            fi
            RU_CFG="$2"
            shift 2
            ;;
        --config_dir=*)
          CONFIG_DIR="${1#*=}"
          CONFIG_DIR_SET=true
          shift
          ;;
        --config_dir)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for $1 option"
            show_usage
            exit 1
          fi
          CONFIG_DIR="$2"
          CONFIG_DIR_SET=true
          shift 2
          ;;
        --cubb-sdk=*)
          cuBB_SDK="${1#*=}"
          shift
          ;;
        --cubb-sdk)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for $1 option"
            show_usage
            exit 1
          fi
          cuBB_SDK="$2"
          shift 2
          ;;
        *) # unknown option
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Apply CONFIG_DIR logic after all parsing is complete
if [[ "$CONFIG_DIR_SET" == "false" ]]; then
    CONFIG_DIR="$cuBB_SDK"
fi

TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
if [[ ! -f $TEST_CONFIG_FILE ]]; then
    echo "$TEST_CONFIG_FILE is missing. Please run setup1_DU.sh first"
    exit 1
fi
source $TEST_CONFIG_FILE
if [[ -v RU_SETUP_COMPLETE ]]; then
    echo "$0 has already been run, please run setup1_DU.sh first"
    exit 1
fi

# Save the RU_HOST_TYPE from setup1_DU.sh
EXPECTED_RU_HOST_TYPE="$RU_HOST_TYPE"

# Function to detect actual RU host type
detect_ru_host_type() {
    local detected_type

    NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
    detected_type="_DEVKIT"
    if [[ "$NUMA_NODES" == "2" ]]; then
        detected_type="_R750"
    fi
    if [[ "$(arch)" == "aarch64" ]]; then
        if grep -qx "QuantaEdge EGN77C-2U" /sys/devices/virtual/dmi/id/board_name 2>/dev/null; then
            detected_type="_MGX1"
        else
            detected_type="_CG1"
        fi
    fi
    echo "$detected_type"
}

# Detect actual RU host type if not loopback
if [[ "$EXPECTED_RU_HOST_TYPE" != "_LOOPBACK" ]]; then
    ACTUAL_RU_HOST_TYPE=$(detect_ru_host_type)
else
    ACTUAL_RU_HOST_TYPE="$EXPECTED_RU_HOST_TYPE"
fi

# Validate that detected RU host type matches what was configured in setup1_DU.sh
if [[ "$ACTUAL_RU_HOST_TYPE" != "$EXPECTED_RU_HOST_TYPE" ]]; then
    echo "Error: RU host type mismatch detected!"
    echo "  Expected (from setup1_DU.sh): $EXPECTED_RU_HOST_TYPE"
    echo "  Actual (detected on RU):      $ACTUAL_RU_HOST_TYPE"
    echo ""
    echo "This mismatch will cause incorrect core configuration files to be used."
    echo "Please re-run setup1_DU.sh with the correct --ru-host-type parameter:"
    echo ""
    echo "  Example: ./setup1_DU.sh --ru-host-type=$ACTUAL_RU_HOST_TYPE"
    echo ""
    echo "Then re-run setup2_RU.sh"
    exit 1
fi

# Use the validated RU host type
RU_HOST_TYPE="$EXPECTED_RU_HOST_TYPE"

#---------------------------------------------------------------
# Check network interfaces and gather information
#---------------------------------------------------------------
echo "Setting NIC interfaces in RU-emulator config file"

# Determine number of configured interfaces (highest index + 1)
RU_NUM_PORTS=2
for ((i=MAX_PORTS-1; i>=2; i--)); do
    var_name="RU_ETH_INTERFACE_${i}"
    if [[ -n "${!var_name}" ]]; then
        RU_NUM_PORTS=$((i + 1))
        break
    fi
done

# Validate no gaps in port configuration
for ((i=0; i<RU_NUM_PORTS; i++)); do
    var_name="RU_ETH_INTERFACE_${i}"
    if [[ -z "${!var_name}" ]]; then
        echo "Error: Port configuration has a gap at index $i. Please configure ports 0..$((RU_NUM_PORTS-1)) contiguously."
        exit 1
    fi
done

# Check each interface and get info
for ((i=0; i<RU_NUM_PORTS; i++)); do
    var_name="RU_ETH_INTERFACE_${i}"
    iface="${!var_name}"
    if [ ! -d "/sys/class/net/${iface}" ]; then
        echo "Warning: Network interface ${iface} not found in /sys/class/net/"
        echo "         PCIe and MAC address information will not be available for interface $i"
        eval "RU_PCIE_${i}=\"\""
        eval "RU_MAC_ADDRESS_${i}=\"\""
    else
        pcie=$(ethtool -i "${iface}" 2>/dev/null | grep bus-info | awk '{print $2}')
        mac=$(cat /sys/class/net/"${iface}"/address 2>/dev/null)
        eval "RU_PCIE_${i}=\"${pcie}\""
        eval "RU_MAC_ADDRESS_${i}=\"${mac}\""
    fi
done




RU_CFG_PATH="${CONFIG_DIR}/cuPHY-CP/ru-emulator/config"
if [[ "$CONTROLLER_MODE" == *nrSim_SCF* ]]; then
    RU_YAML=$(find "$(realpath "$RU_CFG_PATH")" -maxdepth 1 -type f -name "*$CONTROLLER_MODE*" -print -quit)
    if [ ! -n "$RU_YAML" ]; then
        echo "No file containing '$CONTROLLER_MODE' found in $RU_CFG_PATH"
        exit 1
    fi
elif [[ "$RU_CFG" == *yaml ]]; then
    RU_YAML=${CONFIG_DIR}/cuPHY-CP/ru-emulator/config/$RU_CFG
else
    # Default to single port config (can be overridden with --ru-yaml)
    RU_YAML=${CONFIG_DIR}/cuPHY-CP/ru-emulator/config/config.yaml
fi

#--------------------------------------------------------------
# Mapping logical CPU cores for RU
#--------------------------------------------------------------
# Set opt based on config parsed in setup1_DU.sh
if [[ "$MUMIMO" == "ON" ]]; then
    OPT="_MMIMO"
else
    OPT=
fi

if [[ "$DU_HYPER_THREADED" == "1" ]]; then
    OPT="_HT"
    echo "WARNING!! The logical core mapping does not support both MUMIMO=ON and Hyper threaded DU"
    echo "Configuration files for Hyper threaded DU will be used"
else
    OPT=$OPT
fi

# Append ML2 to OPT if Multi-L2 mode is enabled
if [[ "$MULTI_L2" == true ]]; then
    OPT="${OPT}_ML2"
fi

CORE_CONFIG_OUTPUT=$CONFIG_DIR/testBenches/phase4_test_scripts

# Allow manual selection of CORE_CONFIG_INPUT via optional CLI override
if [[ -n "$CORE_CONFIG_INPUT_OVERRIDE" ]]; then
    CORE_CONFIG_INPUT="${CORE_CONFIG_INPUT_OVERRIDE}"
else
    CORE_CONFIG_INPUT=l1_logical_cores_DU${CUPHY_HOST_TYPE}_RU${RU_HOST_TYPE}${OPT}.yaml
fi

echo "Using CORE_CONFIG_INPUT: $CORE_CONFIG_INPUT"

if [[ -f $cuBB_SDK/cubb_scripts/autoconfig/l1_core_config/$CORE_CONFIG_INPUT ]]; then
    (
    cd $cuBB_SDK/cubb_scripts/autoconfig/l1_core_config;
    out1=$(python3 auto_allocate_physical_cores.py $CORE_CONFIG_INPUT ${CORE_CONFIG_OUTPUT}/physical_cores_ru.yaml -e -q)
    if [ -n "$out1" ]; then
        echo "$out1" | sed 's/^/auto_allocate_physical_cores.py: /'
        echo "Warning: skipping to run auto_override_yaml_cores.py on RU"
    else
        out2=$(python3 auto_override_yaml_cores.py  ${CORE_CONFIG_OUTPUT}/physical_cores_ru.yaml -r $RU_YAML -q)
        if [ -n "$out2" ]; then
            echo "$out2" | sed 's/^/auto_override_yaml_cores.py: /'
        fi
    fi
    )
fi

# Override core assignments for NRSIM_TC 90629 on R750
if [[ "$NRSIM_TC" == "90629" && "$RU_HOST_TYPE" == "_R750" ]]; then
    echo "Overriding core assignments for NRSIM_TC 90629 on R750"
    yq -i ".ru_emulator.ul_core_list = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]" $RU_YAML
    yq -i ".ru_emulator.dl_core_list = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]" $RU_YAML
    echo "Core assignments overridden in $RU_YAML"
fi

if [[ "$NRSIM_TC" == "90629" && "$RU_HOST_TYPE" == "_LOOPBACK" ]]; then
    echo "Overwrite default core assignments for NRSIM_TC 90629 on GH loopback setup"
    yq -i ".ru_emulator.ul_core_list = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]" $RU_YAML
    yq -i ".ru_emulator.ul_srs_core_list = [44, 45, 46]" $RU_YAML
    yq -i ".ru_emulator.dl_core_list = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]" $RU_YAML
    echo "Core assignments overwritten in $RU_YAML"
fi

#==============================================================
# update test_config_summary.sh
# Write variables to the test_config_summary.sh
RU_SETUP_COMPLETE=1
# Build RU port variables dynamically
RU_PORT_VARS=""
for ((i=0; i<RU_NUM_PORTS; i++)); do
    RU_PORT_VARS="$RU_PORT_VARS RU_PCIE_${i} RU_ETH_INTERFACE_${i} RU_MAC_ADDRESS_${i}"
done
VARS="$VARS${RU_PORT_VARS} RU_NUM_PORTS RU_HOST_TYPE RU_YAML RU_SETUP_COMPLETE"
> "$TEST_CONFIG_FILE"  # Clear the file before writing
for var in ${VARS}; do
    echo "$var=\"${!var}\"" >> "$TEST_CONFIG_FILE"
done
chmod +x "$TEST_CONFIG_FILE"
