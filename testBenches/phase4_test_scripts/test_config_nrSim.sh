#!/bin/bash  -e

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

#--------------------------------------------------------------------
#This script sets cuBB nrSim test parameters
#--------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}

CONFIG_DIR=$cuBB_SDK

valid_channels=("PUSCH" "PDSCH" "PDCCH_UL" "PDCCH_DL" "PBCH" "PUCCH" "PRACH" "CSI_RS" "SRS" "BFW_DL" "BFW_UL" "all")

show_usage() {
    echo "Script to set cuBB nrSim test configurations."
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --help         , -h         Show this help message and exit"
    echo "  --channels  <channel_names> OR <bit_mask>   Specify participating channels passed to ru_emulator/test_mac (default: all)"
    echo "                                              Please provide one or more of the following channels. Channel names can be separated by ',' or '+'"
    echo "                                              ${valid_channels[*]}"
    echo "                                              Alternatively, one can specify the channel bit-mask as a hex value (eg:0xF) b0:PUSCH,b1:PDSCH,b2:PDCCH_UL,b3:PDCCH_DL,b4:PBCH,b5:PUCCH,b6:PRACH,b7:CSI_RS,b8:SRS,b9:DL_BFW,b10:UL_BFW "
    echo "  --num-slots=N  , -T N       Set number of test slots (default: 600000)"
    echo "  --ehq=N        , -q N       Enable (1) or disable (0) early-HARQ in PUSCH (default: 1)"
    echo "  --dlc-tb=N     , -t N       Enable (1) or disable (0) # Enable/Disable DLC testbench (default: 0)"
    echo "  --BFP=N        , -B N       Set BFP type (acceptable values: 9,14 | default: 9)"
    echo "  --compression=N  , -o N      Set compression method (acceptable values: 1 (BFP), 4 (mod compression) | default: 1)"
    echo "  --polar-dcdr-list-sz=N  , -p N      Set pusch polar decoder list size (acceptable values: 1 , 2, 4, 8)"
    echo "  --config_dir <path>         Specify the path to the directory containing config files. (default: "\$cuBB_SDK")"
    echo "                              the testBenches scripts will modify configuration files and write output files to this location"
    echo "Example:"
    echo "  $0 --channels PDSCH+CSI_RS -T 300"
}

# Default values
TEST_SLOTS=600000
CHANNELS="all"
CHANNELS_dec=0
EARLY_HARQ_ENABLED=1
DLC_TB_ENABLED=0
COMPRESSION=1
POLAR_DECODER_LIST_SIZE=1

# Parse command-line arguments
BFP_EXPLICITLY_SET=false
COMPRESSION_EXPLICITLY_SET=false
EARLY_HARQ_EXPLICITLY_SET=false
POLAR_DECODER_LIST_SIZE_EXPLICITLY_SET=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --BFP*|-B)
            BFP_EXPLICITLY_SET=true
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
            BFP="$value"
            shift
            ;;
        --channels*|--ehq*|--dlc-tb*|--compression*|--num-ports*|--polar-dcdr-list-sz*|--num-slots*)
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
                --channels) CHANNELS="$value" ;;
                --num-slots) TEST_SLOTS_EXPLICITLY_SET=true; TEST_SLOTS="$value" ;;
                --ehq) EARLY_HARQ_EXPLICITLY_SET=true; EARLY_HARQ_ENABLED="$value" ;;
                --dlc-tb) DLC_TB_ENABLED="$value" ;;
                --compression) COMPRESSION_EXPLICITLY_SET=true; COMPRESSION="$value" ;;
                --polar-dcdr-list-sz) POLAR_DECODER_LIST_SIZE_EXPLICITLY_SET=true; POLAR_DECODER_LIST_SIZE="$value" ;;
                *) echo "Unknown option: $option"; exit 1 ;;
            esac
            shift
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
        -h|--help)
           show_usage
           exit 0
           ;;
        -T|-q|-t|-o|-p)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1 option"
                exit 1
            fi
            case $1 in
                -T) TEST_SLOTS_EXPLICITLY_SET=true; TEST_SLOTS="$2" ;;
                -q) EARLY_HARQ_EXPLICITLY_SET=true; EARLY_HARQ_ENABLED="$2" ;;
                -t) DLC_TB_ENABLED="$2" ;;
                -o) COMPRESSION_EXPLICITLY_SET=true; COMPRESSION="$2" ;;
                -p) POLAR_DECODER_LIST_SIZE_EXPLICITLY_SET=true; POLAR_DECODER_LIST_SIZE="$2" ;;
            esac
            shift 2
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            # Assume the first non-option argument is the pattern name
            pattern_name=${1,,} # Convert the pattern name to lowercase
            shift
            ;;
    esac
done

if [ "$CHANNELS" != "all" ]; then
    if [[ "$CHANNELS" =~ ^0x[0-9A-Fa-f]+$ ]]; then
        echo "Channel list specified as bit mask."
        CHANNELS_dec=$(printf "%d" $CHANNELS)
        if [[ $CHANNELS_dec -gt 2047 ]]; then
            echo "Error: Invalid channel bitmask '$CHANNELS' found in channels. Max value 0x7FF"
            exit 1
        fi
    else
        # Split the string on + and , using parameter expansion
        IFS='+,' read -ra channel_list <<< "$CHANNELS"

        # If channels are separated using , change it to +
        NEW_CHANNELS=""

        # Validate each individual channel name
        for channel in "${channel_list[@]}"; do
            if [[ ! " ${valid_channels[*]} " =~ " $channel " ]]; then
                echo "Error: Invalid channel '$channel' found in channels."
                echo "List of valid channels: ${valid_channels[*]}"
                exit 1
            fi
            if [ -n "$NEW_CHANNELS" ]; then
            NEW_CHANNELS="$NEW_CHANNELS"+"$channel"
            else
            NEW_CHANNELS="$channel"
            fi
        done
        CHANNELS=$NEW_CHANNELS
    fi
fi

TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
if [[ ! -f $TEST_CONFIG_FILE ]]; then
    echo "$TEST_CONFIG_FILE is missing. Please run setup1_DU.sh and setup2_RU.sh first"
    exit 1
fi

source $TEST_CONFIG_FILE
if [[ ! -v DU_SETUP_COMPLETE ]] || [[ ! -v RU_SETUP_COMPLETE ]]; then
    echo "Error: Please run setup1_DU.sh and setup2_RU.sh before executing $0."
    exit 1
fi

if [[ "$FORCE" != true ]] && [[ -v TEST_CONFIG_DONE ]]; then
    echo "Error: $0 has already been run."
    echo "   use --force to regenerate configs"
    exit 1
fi

#--------------------------------------------------------------
# early-HARQ related changes
#--------------------------------------------------------------
if [ "$COMPRESSION_EXPLICITLY_SET" = true ]; then
    if [ "$EARLY_HARQ_ENABLED" -eq 1 ]; then
        yq -i '.indicationPerSlot.uciIndPerSlot = 2' $TESTMAC_YAML
    elif [ "$EARLY_HARQ_ENABLED" -eq 0 ]; then
        # For non-early-HARQ
        yq -i '.indicationPerSlot.uciIndPerSlot = 0' $TESTMAC_YAML
    else
        echo "Error: --ehq|-q must be either 0 or 1"
        exit 1
    fi

    if [ "$EARLY_HARQ_ENABLED" -eq 1 ]; then
        EHQ_STATUS=enabled
    else
        EHQ_STATUS=disabled
    fi
fi

#--------------------------------------------------------------
# BFP and compression method related changes
#--------------------------------------------------------------
if [ "$COMPRESSION_EXPLICITLY_SET" = true ]; then
    if [ "$BFP_EXPLICITLY_SET" = true ] && [ "$COMPRESSION" != "1" ]; then
        echo "Warning: BFP value is set but compression method is not set to BFP (1). This may lead to unexpected behavior."
    fi

    if [ "$COMPRESSION" -eq 1 ] || [ "$COMPRESSION" -eq 4 ]; then
        if [ "$COMPRESSION" -eq 1 ]; then
            # For BFP compression method
            if [ "$BFP" -eq 9 ] || [ "$BFP" -eq 14 ]; then
                yq -i "with(.cuphydriver_config.cells[].dl_iq_data_fmt; .comp_meth = 1 | .bit_width = $BFP)" $CUPHY_YAML
                yq -i "with(.cuphydriver_config.cells[].ul_iq_data_fmt; .comp_meth = 1 | .bit_width = $BFP)" $CUPHY_YAML
                yq -i "with(.ru_emulator.cell_configs[].dl_iq_data_fmt; .comp_meth = 1 | .bit_width = $BFP)" $RU_YAML
                yq -i "with(.ru_emulator.cell_configs[].ul_iq_data_fmt; .comp_meth = 1 | .bit_width = $BFP)" $RU_YAML
            else
                echo "Error: BFP must be either 9 or 14"
                exit 1
            fi
        else
            # For mod compression method
            yq -i "with(.cuphydriver_config.cells[].dl_iq_data_fmt; .comp_meth = 4 | .bit_width = 9)" $CUPHY_YAML
            yq -i "with(.ru_emulator.cell_configs[].dl_iq_data_fmt; .comp_meth = 4 | .bit_width = 9)" $RU_YAML

            #yq -i ".cuphydriver_config.bfw_beta_prescaler = 16384" $CUPHY_YAML
        fi
    else
        echo "Error: Compression method must be either 1 (BFP) or 4 (mod compression)"
        exit 1
    fi
fi

if [ "$POLAR_DECODER_LIST_SIZE_EXPLICITLY_SET" = true ]; then
    if [ "$POLAR_DECODER_LIST_SIZE" -eq 1 ] || [ "$POLAR_DECODER_LIST_SIZE" -eq 2 ] || [ "$POLAR_DECODER_LIST_SIZE" -eq 4 ] || [ "$POLAR_DECODER_LIST_SIZE" -eq 8 ]; then
        yq -i ".cuphydriver_config.puxch_polarDcdrListSz = $POLAR_DECODER_LIST_SIZE" $CUPHY_YAML
    else
        echo "Error: POLAR_DECODER_LIST_SIZE must be 1, 2, 4 or 8"
        exit 1
    fi
fi

if [ "$TEST_SLOTS_EXPLICITLY_SET" ]; then
   yq -i ".test_slots = $TEST_SLOTS" $TESTMAC_YAML
else
   TEST_SLOTS=$(yq '.test_slots' $TESTMAC_YAML)
fi

echo "Setting NIC interface in cuPHY controller config file"
yq -i ".cuphydriver_config.nics[0].nic = \"$DU_PCIE_0\"" $CUPHY_YAML
yq -i ".cuphydriver_config.cells[].src_mac_addr = \"$DU_MAC_ADDRESS_0\"" $CUPHY_YAML
yq -i ".cuphydriver_config.cells[].nic = \"$DU_PCIE_0\"" $CUPHY_YAML

echo "Setting NIC interface and peer mac address in ru-emulator config file"
yq -i ".ru_emulator.peers[0].peerethaddr = \"$DU_MAC_ADDRESS_0\"" $RU_YAML
yq -i ".ru_emulator.nics[0].nic_interface = \"$RU_PCIE_0\"" $RU_YAML

if [[ "$CUPHY_HOST_TYPE" == "_R750" ]]; then
    yq -i ".ru_emulator.aerial_fh_split_rx_tx_mempool = 1" $RU_YAML
fi

# Enable/Disable DLC testbench
if [ "$DLC_TB_ENABLED" -eq 1 ]; then
    yq -i '.ru_emulator.dlc_tb = 1' $RU_YAML
elif [ "$DLC_TB_ENABLED" -eq 0 ]; then
    yq -i '.ru_emulator.dlc_tb = 0' $RU_YAML
else
    echo "Error: --dlc-tb|-t must be either 0 or 1"
    exit 1
fi

# Variables appended to VARS (in TEST_CONFIG_FILE) by this script
TEST_VARS="CHANNELS TEST_SLOTS DLC_TB_ENABLED"

if [ "$COMPRESSION_EXPLICITLY_SET" = true ]; then
    TEST_VARS="$TEST_VARS BFP"
fi

if [ "$EARLY_HARQ_EXPLICITLY_SET" = true ]; then
    TEST_VARS="$TEST_VARS EARLY_HARQ_ENABLED EHQ_STATUS"
fi

if [ "$POLAR_DECODER_LIST_SIZE_EXPLICITLY_SET" = true ]; then
    TEST_VARS="$TEST_VARS POLAR_DECODER_LIST_SIZE"
fi

if [[ "$FORCE" = true ]]; then
    # remove all TEST_VARS
    VARS="${VARS%%CHANNELS*}"
fi

# update test_config_summary.sh
TEST_CONFIG_DONE=1
VARS="$VARS $TEST_VARS TEST_CONFIG_DONE"
> "$TEST_CONFIG_FILE"  # Clear the file before writing
for var in ${VARS}; do
    echo "$var=\"${!var}\"" >> "$TEST_CONFIG_FILE"
    printf "%-25s : %-20s\n" "$var" "${!var}"
done
chmod +x "$TEST_CONFIG_FILE"
