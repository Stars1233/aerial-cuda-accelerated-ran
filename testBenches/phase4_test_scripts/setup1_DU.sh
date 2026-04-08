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
#This script is to be run on the DU side (cuphy-controller/testMAC)
#--------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

# Set cuBB SDK path
cuBB_SDK=$(realpath $SCRIPT_DIR/../..)
CONFIG_DIR=$cuBB_SDK
CONFIG_DIR_SET=false

valid_yamls=("F08" "F08_BF3" "F08_CG1" "F08_GL4" "F08_R750" "nrSim_SCF_.+" "nrSim_SCF_CG1_.+")
#ToDo: add 2-port yamls to the list once scripts are extended to support 2-port config
#valid_yamls=("F08_BF3" "F08_CG1" "F08_CG1_2_PORT" "F08_GL4" "F08_R750" "F08_R750_2_PORT")

if [[ "$(arch)" == "aarch64" ]]; then
    if ! CPU_MODEL=$(lscpu | grep -s Cortex- ); then
        CPU_MODEL=
    fi
    # BlueField models:
    # * BlueField-2 ==> A72
    # * BlueField-3 ==> A78
    if [[ "$CPU_MODEL" == *"A72"* ]]; then
        CUPHY_HOST_TYPE="bf2-arm"
    elif [[ "$CPU_MODEL" == *"A78"* ]]; then
        CUPHY_HOST_TYPE="_BF3"
    else
        CUPHY_HOST_TYPE="_CG1"
    fi
else
    NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $3}')
    if [[ "$NUMA_NODES" -gt 1 ]]; then
        CUPHY_HOST_TYPE="_R750"
    else
        CUPHY_HOST_TYPE="" # devkit
    fi
fi


#==============================================================
# Default Values
#==============================================================
# Network interface defaults
DU_ETH_INTERFACE_0="aerial00"
DU_ETH_INTERFACE_1="aerial01"

# RU host type default
RU_HOST_TYPE="_R750"

# Test configuration defaults
MUMIMO="OFF"
CONTROLLER_MODE=F08$CUPHY_HOST_TYPE

if [[ "$CUPHY_HOST_TYPE" == "_CG1" ]]; then
    if nvidia-smi | grep -q "NVIDIA L4"; then
        CONTROLLER_MODE=F08_GL4
        CUPHY_HOST_TYPE="_GL4" #Override CUPHY_HOST_TYPE for GL4 case
    fi
fi

# Multi-L2 enable status default
MULTI_L2=false

show_usage() {
    echo "Usage: $0 [options]"
    echo "Setup script for DU to enable running cuBB test"
    echo
    echo "Options:"
    echo "  --help        , -h       Show this help message and exit"
    echo "  --cuphy-yaml=Y, -y Y     Set cuPHY-controller YAML file"
    echo "                           Acceptable values: ${valid_yamls[*]}"
    echo "                           Default: $CONTROLLER_MODE"
    echo
    echo "  --cubb-sdk=PATH                   Set cuBB SDK path"
    echo "                                    Default: auto-detect (../../ from script dir)"
    echo
    echo "  --du-eth0=INTERFACE               Set first DU network interface name"
    echo "                                    Default: aerial00"
    echo
    echo "  --du-eth1=INTERFACE               Set second DU network interface name"
    echo "                                    Default: aerial01"
    echo "  --ru-host-type=TYPE               Set RU host type for core configuration"
    echo "                                    Default: _R750"
    echo
    echo "  --core-config-input-override=YAML  Override logical core configuration file"
    echo "                                     example: --core-config-input-override=l1_logical_cores_DU_CG1_RU_CG1.yaml"
    echo "                                     Default: unset"
    echo
    echo "  --mumimo <0|1>, -m <0|1>    Select the mumimo test mode:"
    echo "                                0 - mmimo test mode OFF (default)"
    echo "                                1 - mmimo test mode ON"
    echo "  --config_dir <path>      Specify the path to the directory containing config files. (default: "\$cuBB_SDK")"
    echo "                           the testBenches scripts will modify configuration files and write output files to this location"
    echo "  --ml2                    Enable Multi-L2 mode for running two testMAC instances"
    echo
    echo
    echo "Example:"
    echo "  $0 -y F08_CG1 -m 1 --du-eth0=aerial00 --du-eth1=aerial01"
}


DU_HYPER_THREADED=$(cat /sys/devices/system/cpu/smt/active)


# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
           show_usage
           exit 0
           ;;
        --cuphy-yaml*)
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
                --cuphy-yaml) CONTROLLER_MODE="$value" ;;
                *) echo "Unknown option: $option"; exit 1 ;;
            esac
            shift
            ;;
        -y)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1 option"
                exit 1
            fi
            case $1 in
                -y) CONTROLLER_MODE="$2" ;;
            esac
            shift 2
            ;;
        --mumimo|-m)
            if [[ "$2" == "0" ]]; then
                MUMIMO="OFF"
            elif [[ "$2" == "1" ]]; then
                MUMIMO="ON"
            else
                echo "Invalid option for --mumimo. Use 0 for OFF or 1 for ON."
                show_usage
                exit 1
            fi
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
        --du-eth0=*)
          DU_ETH_INTERFACE_0="${1#*=}"
          shift
          ;;
        --du-eth0)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for $1 option"
            show_usage
            exit 1
          fi
          DU_ETH_INTERFACE_0="$2"
          shift 2
          ;;
        --du-eth1=*)
          DU_ETH_INTERFACE_1="${1#*=}"
          shift
          ;;
        --du-eth1)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for $1 option"
            show_usage
            exit 1
          fi
          DU_ETH_INTERFACE_1="$2"
          shift 2
          ;;
        --ru-host-type=*)
          RU_HOST_TYPE="${1#*=}"
          shift
          ;;
        --ru-host-type)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for $1 option"
            show_usage
            exit 1
          fi
          RU_HOST_TYPE="$2"
          shift 2
          ;;
        --ml2)
            MULTI_L2=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Apply CONFIG_DIR logic after all parsing is complete
if [[ "$CONFIG_DIR_SET" == "false" ]]; then
    CONFIG_DIR="$cuBB_SDK"
fi

#---------------------------------------------------------------
# Check network interfaces and gather information
#---------------------------------------------------------------
# Check first interface and get info if available
if [ ! -d "/sys/class/net/${DU_ETH_INTERFACE_0}" ]; then
    echo "Warning: Network interface ${DU_ETH_INTERFACE_0} not found in /sys/class/net/"
    echo "         PCIe and MAC address information will not be available for interface 0"
    DU_PCIE_0=""
    DU_MAC_ADDRESS_0=""
else
    DU_PCIE_0=$(ethtool -i ${DU_ETH_INTERFACE_0} | grep bus-info | awk '{print $2}')
    DU_MAC_ADDRESS_0=$(cat /sys/class/net/"${DU_ETH_INTERFACE_0}"/address)
fi

# Check second interface and get info if available
if [ ! -d "/sys/class/net/${DU_ETH_INTERFACE_1}" ]; then
    echo "Warning: Network interface ${DU_ETH_INTERFACE_1} not found in /sys/class/net/"
    echo "         PCIe and MAC address information will not be available for interface 1"
    DU_PCIE_1=""
    DU_MAC_ADDRESS_1=""
else
    DU_PCIE_1=$(ethtool -i ${DU_ETH_INTERFACE_1} | grep bus-info | awk '{print $2}')
    DU_MAC_ADDRESS_1=$(cat /sys/class/net/"${DU_ETH_INTERFACE_1}"/address)
fi

# Check if suffix for cuPHY-controller yaml file name is valid
valid=false
for yaml in "${valid_yamls[@]}"; do
  if [[ "$CONTROLLER_MODE" =~ $yaml ]]; then
    valid=true
    break
  fi
done

if [[ "$valid" == false ]]; then
    echo "Error: Invalid cuPHY-controller yaml value '$CONTROLLER_MODE'. Please provide one of the following values: ${valid_yamls[*]}"
    exit 1
fi

#For nrSim cases, we set MUMIMO flag based on the auto-generated cuphycontroller yaml file
if [[ "$CONTROLLER_MODE" == *nrSim_SCF* ]]; then
    if [[ "$CONTROLLER_MODE" == *SCF_CG1* ]]; then
        NRSIM_TC=${CONTROLLER_MODE#*SCF_CG1_}
    else
        NRSIM_TC=${CONTROLLER_MODE#*SCF_}
    fi
    echo $NRSIM_TC
    if [[ -v NRSIM_TC ]]; then
        pushd $cuBB_SDK > /dev/null
	SCRIPT_ARGS="-c $NRSIM_TC"
	[[ "$CONFIG_DIR_SET" == "true" ]] && SCRIPT_ARGS="$SCRIPT_ARGS -b $CONFIG_DIR"
        ./cubb_scripts/autoconfig/auto_AllConfig.py $SCRIPT_ARGS
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "Warning: Issue generating configuration files: $ret"
        fi
        popd > /dev/null
    fi
fi

CUPHY_YAML=${CONFIG_DIR}/cuPHY-CP/cuphycontroller/config/cuphycontroller_${CONTROLLER_MODE}.yaml
if [ ! -f "$CUPHY_YAML" ]; then
    echo "File $CUPHY_YAML does not exist"
    exit 1
fi

#For nrSim cases, we set MUMIMO flag based on the auto-generated cuphycontroller yaml file
if [[ "$CONTROLLER_MODE" == *nrSim_SCF* ]]; then
    if [ "$(yq '.cuphydriver_config.mMIMO_enable' $CUPHY_YAML)" = "1" ]; then
        MUMIMO="ON"
    else
        MUMIMO="OFF"
    fi
fi

TESTMAC_YAML=${CONFIG_DIR}/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml
l2adapter_filename=$(sed -n 's/^l2adapter_filename: *//p' "$CUPHY_YAML")
L2A_YAML=${CONFIG_DIR}/cuPHY-CP/cuphycontroller/config/$l2adapter_filename
NVLOG_YAML=${CONFIG_DIR}/cuPHY/nvlog/config/nvlog_config.yaml

#--------------------------------------------------------------
# Mapping logical CPU cores for DU
#--------------------------------------------------------------
echo "Logical core assignment on DU"
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
    out1=$(python3 auto_allocate_physical_cores.py $CORE_CONFIG_INPUT ${CORE_CONFIG_OUTPUT}/physical_cores_du.yaml -q)
    if [ -n "$out1" ]; then
        echo "$out1" | sed 's/^/auto_allocate_physical_cores.py: /'
        echo "Warning: skipping to run auto_override_yaml_cores.py on DU"
    else
        out2=$(python3 auto_override_yaml_cores.py  ${CORE_CONFIG_OUTPUT}/physical_cores_du.yaml -c $CUPHY_YAML -l $L2A_YAML -t $TESTMAC_YAML -q)
        if [ -n "$out2" ]; then
            echo "$out2" | sed 's/^/auto_override_yaml_cores.py: /'
        fi
    fi
    )
fi


#==============================================================
# Create test_config_summary.sh
TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
DU_SETUP_COMPLETE=1

# Read NUM_DL_WORKERS from CUPHY YAML to populate in test_config_summary.sh
NUM_DL_WORKERS=$(yq '.cuphydriver_config.workers_dl | length' $CUPHY_YAML)
echo "NUM_DL_WORKERS from CUPHY YAML: $NUM_DL_WORKERS"

# Write variables to the test_config_summary.sh
VARS="VARS CUPHY_HOST_TYPE RU_HOST_TYPE CONTROLLER_MODE DU_PCIE_0 DU_PCIE_1 DU_ETH_INTERFACE_0 DU_ETH_INTERFACE_1 DU_MAC_ADDRESS_0 DU_MAC_ADDRESS_1 DU_HYPER_THREADED MUMIMO CUPHY_YAML TESTMAC_YAML L2A_YAML NVLOG_YAML NRSIM_TC DU_SETUP_COMPLETE MULTI_L2 NUM_DL_WORKERS"
> "$TEST_CONFIG_FILE"  # Clear the file before writing
for var in ${VARS}; do
    echo "$var=\"${!var}\"" >> "$TEST_CONFIG_FILE"
done
chmod +x "$TEST_CONFIG_FILE"
