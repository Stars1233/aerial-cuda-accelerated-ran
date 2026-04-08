#!/bin/bash  -e

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
#This script sets cuBB test parameters
#--------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

# Source the valid patterns file
source "$SCRIPT_DIR/valid_perf_patterns.sh"

cuBB_SDK=$(realpath $SCRIPT_DIR/../..)

CONFIG_DIR=$cuBB_SDK
CONFIG_DIR_SET=false

valid_channels=("PUSCH" "PDSCH" "PDCCH_UL" "PDCCH_DL" "PBCH" "PUCCH" "PRACH" "CSI_RS" "SRS" "BFW_DL" "BFW_UL" "all")

show_usage() {
    echo "Script to set cuBB test configurations."
    echo "Usage: $0 <pattern> [options]"
    echo
    echo "Arguments:"
    echo "  pattern_name             Name of the pattern (required)"
    echo "  Please provide one of the following patterns as the argument:"
    echo "  ${valid_perf_patterns[*]}"
    echo
    echo "Options:"
    echo "  --help         , -h         Show this help message and exit"
    echo "  --BFP=N        , -B N       Set BFP type (acceptable values: 9,14 | default: 9)"
    echo "  --compression=N  , -o N      Set compression method (acceptable values: 1 (BFP), 4 (mod compression) | default: 1)"
    echo "  --channels  <channel_names> OR <bit_mask>   Specify participating channels passed to ru_emulator/test_mac (default: all)"
    echo "                                              Please provide one or more of the following channels. Channel names can be separated by ',' or '+'"
    echo "                                              ${valid_channels[*]}"
    echo "                                              Alternatively, one can specify the channel bit-mask as a hex value (eg:0xF) b0:PUSCH,b1:PDSCH,b2:PDCCH_UL,b3:PDCCH_DL,b4:PBCH,b5:PUCCH,b6:PRACH,b7:CSI_RS,b8:SRS,b9:DL_BFW,b10:UL_BFW "
    echo "  --DGL=N        , -d N       Enable (1) or disable (0) device graph launch (default: 1)"
    echo "  --green-ctx=N  , -g N       Enable (1) or disable (0) green-context (default: 0)"
    echo "  --gc-wqs=N                  Enable (1) or disable (0) green context workqueues (default: 1)"
    echo "  --data-lake=N  , -l N       Enable (1) or disable (0) datalake (default: 0)"
    echo "  --dlc-tb=N     , -t N       Enable (1) or disable (0) # Enable/Disable DLC testbench (default: 0)"
    echo "  --ehq=N        , -q N       Enable (1) or disable (0) early-HARQ in PUSCH (default: 1)"
    echo "  --log-nic-timings           Enable additional debug logs related to NIC timings (disabled by default)"
    echo "  --ru-worker-tracing         Enable RU C-plane worker tracing logs (disabled by default)"
    echo "  --reduced-logging           Enable reduced logging mode - disables detailed tracing and processing time logs (disabled by default)"
    echo "  --cupti                     Enable CUPTI tracing (disabled by default)"
    echo "  --num-cells=N  , -c N       Set number of cells (default: 20)"
    echo "  --num-ports=N  , -p N       Set number of NIC ports (default: 1, acceptable values: 1 or 2)"
    echo "  --num-slots=N  , -T N       Set number of test slots (default: 600000)"
    echo "  --STT=N        , -s N       Set schedule total time (default: 455000 for 4T4R and 480000 for MIMO)"
    echo "  --enable_32dl=N, -e N       Enable (1) or disable (0) enable_32dl (default: 0)"
    echo "  --work-cancel=N, -w N       Set work cancellation mode (0: disable, 1: conditional graph nodes, 2: device graph launch | default: 2)"
    echo "  --dlc-packing=N             Set DL C-plane core packing scheme (0: default, 1: fixed per-cell, 2: dynamic workload-based [not yet supported] | default: 0)"
    echo "  --dlc-core-index=<array>    Set DL C-plane core index per cell for scheme=1 (e.g., --dlc-core-index=[0,1,2] for 3 cells)"
    echo "  --pmu=N        , -m N       Set pmu_metrics mode (0: disabled, 1: general counters (platform agnostic), 2: topdown metrics (Grace only), 3: cache metrics (Grace only) | default: 0)"
    echo "  --force        , -f         Force re-generation of configuration"
    echo "  --cubb-sdk=PATH             Set cuBB SDK path"
    echo "                              Default: auto-detect (../../ from script dir)"
    echo "  --config_dir <path>         Specify the path to the directory containing config files. (default: "\$cuBB_SDK")"
    echo "                              the testBenches scripts will modify configuration files and write output files to this location"
    echo "  --cicd <test_case_string>   Parse a test case string (e.g., F08_6C_79_MODCOMP_STT480000_EH_1P)"
    echo "                              This option is primarily for CI/CD integration and will parse the test case"
    echo "                              string to extract pattern, cells, and other parameters automatically"
    echo
    echo "Example:"
    echo "  $0 59c --num-cells=20 -B 9 -T 30000 --ehq=0 --channels PUSCH+PDSCH+CSI_RS"
}

# Helper function to compute the number of DLC tasks/cores available
# This mirrors the logic in cuphydriver_api.cpp get_num_dlc_tasks()
# Arguments:
#   $1 - num_workers: number of DL workers from the logical cores config
#   $2 - commViaCpu: 1 if GL4 mode (gpu_init_comms_via_cpu), 0 otherwise
#   $3 - mMIMO_enable: 1 if MIMO enabled, 0 otherwise
# Returns: number of DLC tasks via echo
get_num_dlc_tasks() {
    local num_workers=$1
    local commViaCpu=$2
    local mMIMO_enable=$3

    # Ensure minimum number of workers required
    if [ "$num_workers" -lt 2 ]; then
        echo 0
        return
    fi

    if [ "$commViaCpu" -eq 1 ]; then
        if [ "$mMIMO_enable" -eq 1 ]; then
            echo $((num_workers - 3))
        else
            echo $((num_workers - 2))
        fi
    elif [ "$mMIMO_enable" -eq 1 ]; then
        if [ "$num_workers" -gt 5 ]; then
            echo $((num_workers - 2))
        else
            echo "$num_workers"
        fi
    else
        echo $((num_workers - 1))
    fi
}

# Helper function to generate DLC core index array for fixed packing scheme
# For every group of 3 cells:
#   - First cell (index 0, 3, 6, ...) gets a dedicated DLC core
#   - Next two cells (indices 1,2 and 4,5 and 7,8, ...) share a DLC core
# This means each group of 3 cells uses 2 DLC cores
# Arguments:
#   $1 - num_cells: total number of cells
#   $2 - num_dlc_cores: total available DLC cores
# Returns: array string like "[0,1,1,2,3,3,4,5,5]" via echo
# Example: 9 cells, 6 DLC cores -> [0,1,1,2,3,3,4,5,5]
generate_dlc_core_index_grouped() {
    local num_cells=$1
    local num_dlc_cores=$2

    # Calculate required cores: each group of 3 cells needs 2 cores
    local num_groups=$(( (num_cells + 2) / 3 ))
    local required_cores=$((num_groups * 2))

    if [ "$num_dlc_cores" -lt "$required_cores" ]; then
        echo "Error: Not enough DLC cores ($num_dlc_cores) for fixed packing scheme with $num_cells cells (need $required_cores cores for $num_groups groups)" >&2
        return 1
    fi

    local result="["

    for ((cell_idx=0; cell_idx<num_cells; cell_idx++)); do
        if [ "$cell_idx" -gt 0 ]; then
            result="${result},"
        fi

        # Determine which group of 3 this cell belongs to
        local group_idx=$((cell_idx / 3))
        local pos_in_group=$((cell_idx % 3))

        if [ "$pos_in_group" -eq 0 ]; then
            # First cell in group gets dedicated core (even core index: 0, 2, 4, ...)
            local core_idx=$((group_idx * 2))
            result="${result}${core_idx}"
        else
            # Second and third cells in group share a core (odd core index: 1, 3, 5, ...)
            local core_idx=$((group_idx * 2 + 1))
            result="${result}${core_idx}"
        fi
    done

    result="${result}]"
    echo "$result"
}

# Default values
NUM_CELLS=20
NUM_PORTS=1
TEST_SLOTS=600000
BFP=9
DEVICE_GRAPH_LAUNCH_ENABLED=1
EARLY_HARQ_ENABLED=1
WORK_CANCEL_MODE=2
USE_GREEN_CONTEXT=0
USE_GC_WQS=1 # Use green contexts work queues (default enabled, but relevant only if USE_GREEN_CONTEXT is set)
STT_DEFAULT=455000
STT=""
LOG_NIC_TIMINGS=false
RU_WORKER_TRACING=false
REDUCED_LOGGING=false
CUPTI_TRACING=false
PMU_METRICS=0
CHANNELS="all"
CHANNELS_dec=0
COMPRESSION=1
DATALAKE=0
DLC_TB_ENABLED=0
ENABLE_32DL=0
DLC_CORE_PACKING_SCHEME=0
DLC_CORE_PACKING_SCHEME_USER_SPECIFIED=false
DLC_CORE_INDEX=""
DLC_CORE_INDEX_USER_SPECIFIED=false
# Parse command-line arguments
BFP_EXPLICITLY_SET=false
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
        --channels*|--DGL*|--dlc-tb*|--ehq*|--green-ctx*|--gc-wqs*|--num-cells*|--num-ports*|--num-slots*|--STT*|--work-cancel*|--pmu*|--compression*|--data-lake*)
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
                --DGL) DEVICE_GRAPH_LAUNCH_ENABLED="$value" ;;
                --dlc-tb) DLC_TB_ENABLED="$value" ;;
                --ehq) EARLY_HARQ_ENABLED="$value" ;;
                --green-ctx) USE_GREEN_CONTEXT="$value" ;;
                --gc-wqs) USE_GC_WQS="$value" ;;
                --num-cells) NUM_CELLS="$value" ;;
                --num-ports) NUM_PORTS="$value" ;;
                --num-slots) TEST_SLOTS="$value" ;;
                --STT) STT="$value" ;;
                --work-cancel) WORK_CANCEL_MODE="$value" ;;
                --pmu) PMU_METRICS="$value" ;;
                --compression) COMPRESSION="$value" ;;
                --data-lake) DATALAKE="$value" ;;
                *) echo "Unknown option: $option"; exit 1 ;;
            esac
            shift
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
        -h|--help)
           show_usage
           exit 0
           ;;
        -d|-g|-q|-c|-p|-T|-s|-w|-m|-o|-l|-t)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1 option"
                exit 1
            fi
            case $1 in
                -d) DEVICE_GRAPH_LAUNCH_ENABLED="$2" ;;
                -q) EARLY_HARQ_ENABLED="$2" ;;
                -g) USE_GREEN_CONTEXT="$2" ;;
                -c) NUM_CELLS="$2" ;;
                -p) NUM_PORTS="$2" ;;
                -T) TEST_SLOTS="$2" ;;
                -s) STT="$2" ;;
                -w) WORK_CANCEL_MODE="$2" ;;
                -m) PMU_METRICS="$2" ;;
                -o) COMPRESSION="$2" ;;
                -l) DATALAKE="$2" ;;
                -t) DLC_TB_ENABLED="$2" ;;
            esac
            shift 2
            ;;
        --cicd)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1 option"
                exit 1
            fi
            CICD_TEST_CASE=$2
            shift
            ;;
        --log-nic-timings)
            LOG_NIC_TIMINGS=true
            shift
            ;;
        --ru-worker-tracing)
            RU_WORKER_TRACING=true
            shift
            ;;
        --reduced-logging)
            REDUCED_LOGGING=true
            shift
            ;;
        --cupti)
            CUPTI_TRACING=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --enable_32dl)
            ENABLE_32DL="$2"
            shift 2
            ;;
        --dlc-packing=*)
            DLC_CORE_PACKING_SCHEME="${1#*=}"
            DLC_CORE_PACKING_SCHEME_USER_SPECIFIED=true
            shift
            ;;
        --dlc-packing)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1 option"
                exit 1
            fi
            DLC_CORE_PACKING_SCHEME="$2"
            DLC_CORE_PACKING_SCHEME_USER_SPECIFIED=true
            shift 2
            ;;
        --dlc-core-index=*)
            DLC_CORE_INDEX="${1#*=}"
            DLC_CORE_INDEX_USER_SPECIFIED=true
            shift
            ;;
        --dlc-core-index)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for $1 option"
                exit 1
            fi
            DLC_CORE_INDEX="$2"
            DLC_CORE_INDEX_USER_SPECIFIED=true
            shift 2
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

# Apply CONFIG_DIR logic after all parsing is complete
if [[ "$CONFIG_DIR_SET" == "false" ]]; then
    CONFIG_DIR="$cuBB_SDK"
fi

# Validate NUM_PORTS value
if [ "$NUM_PORTS" -ne 1 ] && [ "$NUM_PORTS" -ne 2 ] ; then
    echo "Error: NUM_PORTS must be either 1 or 2."
    exit 1
fi

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

if [ -n "$CICD_TEST_CASE" ]; then

    # F08_X_NC_YY_extra 
    #     - X  = pattern (0, A, B, C, D, E, F)
    #     - N  = number of cells - can be 2 digits
    #     - YY = F08 pattern (e.g. 03, 05, 11, 14, etc.)
    #     - Z  = Compression (e.g. 9, 14 - can be 2 digits) NOTE that this is NOT used here
    #     - extra = stuff like "restart" for test with restarting cells
    #         - "restart" - treat it the same for all parameters we parse
    #         - "BFP9" or "BFP14" - compression bits
    #         - Can be chained together. E.g. _BFP14_restart
    #         - 2P for dual port
    if [[ $CICD_TEST_CASE =~ ^[0-9]+$ ]]; then
        echo "$CICD_TEST_CASE"
        exit 0
    fi

    if [[ "$CICD_TEST_CASE" =~ ^F08_([0ABCDEF])_([0-9]{1,2})C_([0-9]{2}[a-z]*)(.*) ]]; then
        f08_pattern="${BASH_REMATCH[1]}"
        num_cells="${BASH_REMATCH[2]}"
        pattern_name="${BASH_REMATCH[3]}"
        extra="${BASH_REMATCH[4]}"
        ehq=0

        # Check if the pattern name is valid
        if [[ ! " ${valid_perf_patterns[*]} " =~ " $pattern_name " ]]; then
            echo "Error: Invalid pattern name '$pattern_name'. Please provide one of the following patterns: ${valid_perf_patterns[*]}"
            exit 1
        fi

        CMD="$pattern_name"

        # Convert the string into an array, split by underscore
        IFS='_' read -ra parts <<< "$extra"

        # Loop over each part and match patterns
        for part in "${parts[@]}"; do
            case "$part" in
                BFP[0-9]*)
                    bfp=${part:3}
                    CMD="$CMD --BFP=$bfp"
                    ;;
                CCDF)
                    ccdf=true
                    ;;
                STT[0-9]*)
                    CMD="$CMD --STT=${part:3}"
                    ;;
                EH)
                    ehq=1
                    ;;
                GC)
                    CMD="$CMD --green-ctx=1"
                    ;;
                DL)
                    CMD="$CMD --data-lake=1"
                    ;;
                1P)
                    num_ports=1
                    ;;
                2P)
                    num_ports=2
                    ;;
            esac
        done
        #CMD="$CMD --BFP=$bfp"
        if [[ $ehq -eq 0 ]]; then
            CMD="$CMD --ehq=0"
        fi
        CMD="$CMD --num-cells=$num_cells"

        if [[ " ${mimo_patterns[*]} " =~ " $pattern_name " ]]; then
            # CICD nees to get this information so that we can pass it to setup1_DU.sh script
            # CICD needs to remove this flag before calling test_config.sh
            CMD="$CMD --mumimo=1"
        fi

        echo $CMD
    else
        exit 1
    fi

    exit 0
fi

# Check if the first argument is empty
if [[ -z "${pattern_name}" ]]; then
  echo "Error: pattern argument missing"
  show_usage
  exit 1
fi

# Check if the pattern name is valid
if [[ ! " ${valid_perf_patterns[*]} " =~ " $pattern_name " ]]; then
    echo "Error: Invalid pattern name '$pattern_name'. Please provide one of the following patterns: ${valid_perf_patterns[*]}"
    exit 1
fi

# Variables appended to VARS (in TEST_CONFIG_FILE) by this script
    TEST_VARS="PATTERN PATTERN_MODE CHANNELS NUM_CELLS NUM_PORTS TEST_SLOTS WORK_CANCEL_MODE WC_MODE BFP EARLY_HARQ_ENABLED EHQ_STATUS DEVICE_GRAPH_LAUNCH_ENABLED DGL_STATUS USE_GREEN_CONTEXT USE_GC_WQS GC_STATUS PMU_METRICS STT DLC_TB_ENABLED ML2_CELL_MASK0 ML2_CELL_MASK1 ML2_CELL_LIST0 ML2_CELL_LIST1 TESTMAC1_YAML"

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

# Validate required variables for port configuration
if [ "$NUM_PORTS" -eq 1 ]; then
    if [[ ! -v DU_MAC_ADDRESS_0 ]] || [[ ! -v RU_PCIE_0 ]]; then
        echo "Error: 1-port configuration requires DU_MAC_ADDRESS_0 and RU_PCIE_0 variables."
        echo "Please ensure setup1_DU.sh and setup2_RU.sh completed successfully."
        exit 1
    fi
elif [ "$NUM_PORTS" -eq 2 ]; then
    if [[ ! -v DU_MAC_ADDRESS_0 ]] || [[ ! -v RU_PCIE_0 ]] || [[ ! -v DU_MAC_ADDRESS_1 ]] || [[ ! -v RU_PCIE_1 ]]; then
        echo "Error: 2-port configuration requires DU_MAC_ADDRESS_0, RU_PCIE_0, DU_MAC_ADDRESS_1 and RU_PCIE_1 variables."
        echo "Please ensure setup1_DU.sh and setup2_RU.sh completed successfully."
        exit 1
    fi
fi

if [[ "$FORCE" != true ]] && [[ -v TEST_CONFIG_DONE ]]; then
    echo "Error: $0 has already been run."
    echo "   use --force to regenerate configs"
    exit 1
fi

if [[ "$FORCE" = true ]]; then
    # remove all TEST_VARS
    VARS="${VARS%%PATTERN*}"
fi






#--------------------------------------------------------------
# Additional considerations for some cases
#--------------------------------------------------------------
if [[ "$USE_GREEN_CONTEXT" -eq 1 && "$WORK_CANCEL_MODE" -ne 0 ]]; then
    if [[ "$pattern_name" == "59c" || "$pattern_name" == "62c" ]]; then
        echo "Warning: --work-cancel is set to $WORK_CANCEL_MODE and Green Context is set to 1. Note that in the CI/CD pipeline for pattern $pattern_name with GC enabled, 'pusch_workCancelMode' is set to 0 as a temporary workaround."
    fi
fi

if [[ "$DATALAKE" -eq 1 ]]; then
    yq -i '.cuphydriver_config.data_core = 30' "$CUPHY_YAML"
    yq -i '.cuphydriver_config.datalake_address = "localhost"' "$CUPHY_YAML"
    yq -i '.cuphydriver_config.datalake_samples = 10000' "$CUPHY_YAML"
elif [[ "$DATALAKE" -eq 0 ]]; then
    yq -i 'del(.cuphydriver_config.data_core)' "$CUPHY_YAML"
    yq -i 'del(.cuphydriver_config.datalake_address)' "$CUPHY_YAML"
    yq -i 'del(.cuphydriver_config.datalake_samples)' "$CUPHY_YAML"
else
    echo "Error: DATALAKE must either be 0 or 1"
    exit 1
fi

if [[ "$pattern_name" == "62c" ]] || [[ "$pattern_name" == "63c" ]]; then
    yq -i '.cuphydriver_config.pusch_aggr_per_ctx = 9' $CUPHY_YAML
    yq -i '.cuphydriver_config.prach_aggr_per_ctx = 4' $CUPHY_YAML
    yq -i '.cuphydriver_config.ul_input_buffer_per_cell = 15' $CUPHY_YAML
    yq -i '.cuphydriver_config.max_harq_pools = 512' $CUPHY_YAML
fi

if [[ "$pattern_name" == "71" ]] || [[ "$pattern_name" == "79a" ]] || [[ "$pattern_name" == "79b" ]] || [[ "$pattern_name" == "81b" ]] || [[ "$pattern_name" == "81d" ]]; then
    yq -i '.cuphydriver_config.pusch_aggr_per_ctx = 4' $CUPHY_YAML
    yq -i '.cuphydriver_config.srs_aggr_per_ctx = 5' $CUPHY_YAML
fi

# Check if pattern is a MIMO case
if [[ " ${mimo_patterns[*]} " =~ " $pattern_name " ]]; then
    if [[ $MUMIMO != "ON" ]]; then
        echo "Error: pattern_name $pattern_name should use MIMO"
        exit 1
    fi

    STT_DEFAULT=480000
fi

# Check if pattern is a DFT-S-OFDM case
if [[ " ${dft_s_ofdm_patterns[*]} " =~ " $pattern_name " ]]; then
    yq -i '.cuphydriver_config.pusch_dftsofdm = 1' $CUPHY_YAML
fi
#--------------------------------------------------------------
# testMAC MIMO configuration
#--------------------------------------------------------------
if [[ "$MUMIMO" == "ON" ]]; then
    yq -i '.indicationPerSlot.srsIndPerSlot = 2' $TESTMAC_YAML
fi

#--------------------------------------------------------------
# testMAC configuration
#--------------------------------------------------------------
if [ -z "$STT" ]; then
    STT="$STT_DEFAULT"
fi
yq -i ".schedule_total_time = $STT" $TESTMAC_YAML
yq -i '.builder_thread_enable = 1' $TESTMAC_YAML
yq -i '.fapi_delay_bit_mask = 0xFF' $TESTMAC_YAML
yq -i ".test_slots = $TEST_SLOTS" $TESTMAC_YAML

#--------------------------------------------------------------
# cuPHY controller configuration
#--------------------------------------------------------------
# Conformance settings
yq -i '.cuphydriver_config.pusch_tdi = 1' $CUPHY_YAML
yq -i '.cuphydriver_config.pusch_cfo = 1' $CUPHY_YAML
yq -i '.cuphydriver_config.pusch_to = 1' $CUPHY_YAML
yq -i '.cuphydriver_config.puxch_polarDcdrListSz = 8' $CUPHY_YAML

PATTERN_MODE=$(get_pattern_mode "$pattern_name")
if [ "$PATTERN_MODE" == "peak" ]; then
    yq -i '.cuphydriver_config.cells[].pusch_nMaxPrb = 273' $CUPHY_YAML
    echo "pusch_nMaxPrb       : 273"
elif [ "$PATTERN_MODE" == "average" ]; then
    yq -i '.cuphydriver_config.cells[].pusch_nMaxPrb = 136' $CUPHY_YAML
    echo "pusch_nMaxPrb       : 136"
else
    echo "Error: PATTERN must be either 'peak' or 'average'"
    exit 1
fi

if [ "$DEVICE_GRAPH_LAUNCH_ENABLED" -eq 1 ] || [ "$DEVICE_GRAPH_LAUNCH_ENABLED" -eq 0 ]; then
    yq -i ".cuphydriver_config.pusch_deviceGraphLaunchEn = $DEVICE_GRAPH_LAUNCH_ENABLED" $CUPHY_YAML
else
    echo "Error: --DGL|-d must be either 0 or 1"
    exit 1
fi

if [ "$WORK_CANCEL_MODE" -ge 0 ] && [ "$WORK_CANCEL_MODE" -le 2 ]; then
    yq -i ".cuphydriver_config.pusch_workCancelMode = $WORK_CANCEL_MODE" $CUPHY_YAML
else
    echo "Error: --work-cancel|-w must be either 0, 1 or 2"
    exit 1
fi

if [ "$USE_GREEN_CONTEXT" -eq 1 ] || [ "$USE_GREEN_CONTEXT" -eq 0 ]; then
    yq -i ".cuphydriver_config.use_green_contexts = $USE_GREEN_CONTEXT" $CUPHY_YAML
else
    echo "Error: --green-ctx|-g must be either 0 or 1"
    exit 1
fi
if [ "$USE_GC_WQS" -eq 1 ] || [ "$USE_GC_WQS" -eq 0 ]; then
    yq -i ".cuphydriver_config.use_gc_workqueues = $USE_GC_WQS" $CUPHY_YAML
else
    echo "Error: --gc-wqs must be either 0 or 1"
    exit 1
fi
yq -i ".cuphydriver_config.pmu_metrics = $PMU_METRICS" $CUPHY_YAML
# Update number of cells
yq -i ".cuphydriver_config.cell_group_num = $NUM_CELLS" $CUPHY_YAML
# Enable DL core affinity by default
yq -i '.cuphydriver_config.enable_dl_core_affinity = 1' $CUPHY_YAML

# DL C-plane core packing scheme configuration
# For pattern 89, default to fixed per-cell packing scheme if not specified by user
if [ "$pattern_name" == "89" ] && [ "$DLC_CORE_PACKING_SCHEME_USER_SPECIFIED" = false ]; then
    DLC_CORE_PACKING_SCHEME=1
    echo "Pattern 89 detected: defaulting to fixed per-cell DLC packing scheme (--dlc-packing=1)"
fi

if [ "$DLC_CORE_PACKING_SCHEME" -eq 2 ]; then
    echo "Error: dlc_core_packing_scheme=2 (dynamic workload-based) is not yet supported"
    exit 1
fi

if [ "$DLC_CORE_PACKING_SCHEME" -ge 0 ] && [ "$DLC_CORE_PACKING_SCHEME" -le 2 ]; then
    yq -i ".cuphydriver_config.dlc_core_packing_scheme = $DLC_CORE_PACKING_SCHEME" $CUPHY_YAML
else
    echo "Error: --dlc-packing must be 0, 1, or 2"
    exit 1
fi

# Compute the number of DLC tasks/cores available for bound checking and auto-generation
# NUM_DL_WORKERS should already be set by setup1_DU.sh in test_config_summary.sh
echo "NUM_DL_WORKERS (from test_config_summary.sh): $NUM_DL_WORKERS"

# Determine commViaCpu (1 for GL4 mode, 0 otherwise)
if [[ "$CONTROLLER_MODE" == "F08_GL4" ]]; then
    COMM_VIA_CPU=1
else
    COMM_VIA_CPU=0
fi

# Determine mMIMO_enable (1 if MUMIMO is ON, 0 otherwise)
if [[ "$MUMIMO" == "ON" ]]; then
    MMIMO_ENABLE=1
else
    MMIMO_ENABLE=0
fi

# Calculate the number of DLC tasks available
NUM_DLC_TASKS=$(get_num_dlc_tasks "$NUM_DL_WORKERS" "$COMM_VIA_CPU" "$MMIMO_ENABLE")
echo "DLC tasks available: $NUM_DLC_TASKS (DL workers: $NUM_DL_WORKERS, commViaCpu: $COMM_VIA_CPU, mMIMO: $MMIMO_ENABLE)"

# Set per-cell dlc_core_index when packing scheme is 1 (fixed per-cell)
if [ "$DLC_CORE_PACKING_SCHEME" -eq 1 ]; then
    # Auto-generate dlc-core-index for pattern 89 if not specified by user
    if [ -z "$DLC_CORE_INDEX" ] && [ "$DLC_CORE_INDEX_USER_SPECIFIED" = false ]; then
        if [ "$pattern_name" == "89" ]; then
            # Auto-generate for fixed packing scheme: cells at index 0,3,6,... get dedicated cores
            DLC_CORE_INDEX=$(generate_dlc_core_index_grouped "$NUM_CELLS" "$NUM_DLC_TASKS")
            if [ $? -ne 0 ]; then
                echo "Error: Failed to auto-generate dlc_core_index for fixed packing scheme"
                exit 1
            fi
            echo "Auto-generated dlc_core_index (fixed packing scheme): $DLC_CORE_INDEX"
        else
            echo "Error: --dlc-core-index must be specified when --dlc-packing=1"
            exit 1
        fi
    fi

    # Parse the dlc_core_index array (e.g., [0,1,2])
    # Remove brackets and split by comma
    DLC_CORE_INDEX_CLEAN=$(echo "$DLC_CORE_INDEX" | tr -d '[]' | tr -d ' ')
    IFS=',' read -ra DLC_CORE_INDEX_ARR <<< "$DLC_CORE_INDEX_CLEAN"
    
    if [ "${#DLC_CORE_INDEX_ARR[@]}" -lt "$NUM_CELLS" ]; then
        echo "Error: dlc_core_index array has ${#DLC_CORE_INDEX_ARR[@]} elements but NUM_CELLS is $NUM_CELLS"
        exit 1
    fi

    # Bound checking: ensure all indices are within the available DLC tasks
    if [ "$DLC_CORE_INDEX_USER_SPECIFIED" = true ]; then
        for ((i=0; i<NUM_CELLS; i++)); do
            core_idx=${DLC_CORE_INDEX_ARR[$i]}
            if [ "$core_idx" -ge "$NUM_DLC_TASKS" ]; then
                echo "Error: dlc_core_index[$i]=$core_idx exceeds available DLC tasks ($NUM_DLC_TASKS)"
                echo "  Maximum valid index is $((NUM_DLC_TASKS - 1))"
                exit 1
            fi
            if [ "$core_idx" -lt 0 ]; then
                echo "Error: dlc_core_index[$i]=$core_idx is negative (must be >= 0)"
                exit 1
            fi
        done
        echo "Bound checking passed: all dlc_core_index values are within range [0, $((NUM_DLC_TASKS - 1))]"
    fi
    
    # Set dlc_core_index for each cell
    CELL_COUNT=$(yq '.cuphydriver_config.cells | length' $CUPHY_YAML)
    for ((i=0; i<CELL_COUNT && i<NUM_CELLS; i++)); do
        yq -i ".cuphydriver_config.cells[$i].dlc_core_index = ${DLC_CORE_INDEX_ARR[$i]}" $CUPHY_YAML
    done
    echo "DL C-plane core packing scheme: fixed per-cell"
    echo "  dlc_core_index mapping: ${DLC_CORE_INDEX_ARR[*]}"
else
    echo "DL C-plane core packing scheme: default"
fi

#--------------------------------------------------------------
# cuPHY controller MIMO configuration
#--------------------------------------------------------------
if [[ "$MUMIMO" == "ON" ]]; then
    echo "Setting mtu size to 8192"
    yq -i '.cuphydriver_config.nics[0].mtu = 8192' $CUPHY_YAML
    yq -i '.cuphydriver_config.fix_beta_dl = 0' $CUPHY_YAML
    yq -i '.cuphydriver_config.mMIMO_enable = 1' $CUPHY_YAML
    yq -i '.cuphydriver_config.enable_srs = 1' $CUPHY_YAML
    # Dispatch on controller mode first
    if [[ $CONTROLLER_MODE == "F08_GL4" ]]; then
        # GL4 mode settings
        yq -i '.cuphydriver_config.mps_sm_ul_order = 6' $CUPHY_YAML
        yq -i '.cuphydriver_config.mps_sm_srs = 14' $CUPHY_YAML
        yq -i '.cuphydriver_config.max_harq_pools = 256' $CUPHY_YAML
        yq -i '.cuphydriver_config.total_num_srs_chest_buffers = 3072' $CUPHY_YAML
        yq -i '.cuphydriver_config.ul_input_buffer_per_cell = 5' $CUPHY_YAML
        yq -i '.cuphydriver_config.bfw_c_plane_chaining_mode = 1' $CUPHY_YAML    
        yq -i '.cuphydriver_config.cells[].srs_prb_stride = 273' $CUPHY_YAML
    else
        yq -i ".cuphydriver_config.total_num_srs_chest_buffers = $((1024 * NUM_CELLS))" $CUPHY_YAML
        # CG1 mode settings - dispatch on green context
        yq -i '.cuphydriver_config.mps_sm_ul_order = 10' $CUPHY_YAML
        if [ "$USE_GREEN_CONTEXT" -eq 1 ]; then
            yq -i '.cuphydriver_config.mps_sm_srs = 48' $CUPHY_YAML
        else
            yq -i '.cuphydriver_config.mps_sm_srs = 32' $CUPHY_YAML
        fi
        # Adjust SM provisioning for RTX 4500-class GPUs (SM limit ~82)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null)
        if echo "$GPU_NAME" | grep -qiE 'RTX[^,]*4500|RTX Pro 4500'; then
            yq -i '.cuphydriver_config.mps_sm_pusch = 40' $CUPHY_YAML
            yq -i '.cuphydriver_config.mps_sm_pdsch = 80' $CUPHY_YAML
        fi
    fi
    yq -i '.cuphydriver_config.dlc_bfw_enable_divide_per_cell = 1' $CUPHY_YAML
    yq -i '.cuphydriver_config.ulc_alloc_cplane_bfw_txq = 1' $CUPHY_YAML
    yq -i '.cuphydriver_config.ul_srs_aggr3_task_launch_offset_ns = 500000' $CUPHY_YAML

    # Update eAxC IDs for MIMO
    MIMO_EAXC_IDS="[32512, 32513, 32514, 32515, 32516, 32517, 32518, 32519, 32520, 32521, 32522, 32523, 32524, 32525, 32526, 32527]"
    if [[ "$ENABLE_32DL" -eq 1 ]]; then
        MIMO_EAXC_IDS_PDSCH="[32512, 32513, 32514, 32515, 32516, 32517, 32518, 32519, 32520, 32521, 32522, 32523, 32524, 32525, 32526, 32527, 32528, 32529, 32530, 32531, 32532, 32533, 32534, 32535, 32536, 32537, 32538, 32539, 32540, 32541, 32542, 32543]"
    else
        MIMO_EAXC_IDS_PDSCH="[32512, 32513, 32514, 32515, 32516, 32517, 32518, 32519, 32520, 32521, 32522, 32523, 32524, 32525, 32526, 32527]"
    fi
    yq -i ".cuphydriver_config.cells[].eAxC_id_ssb_pbch = $MIMO_EAXC_IDS" $CUPHY_YAML
    yq -i ".cuphydriver_config.cells[].eAxC_id_pdcch = $MIMO_EAXC_IDS" $CUPHY_YAML
    yq -i ".cuphydriver_config.cells[].eAxC_id_pdsch = $MIMO_EAXC_IDS_PDSCH" $CUPHY_YAML
    yq -i ".cuphydriver_config.cells[].eAxC_id_csirs = $MIMO_EAXC_IDS" $CUPHY_YAML
    yq -i ".cuphydriver_config.cells[].eAxC_id_pusch = $MIMO_EAXC_IDS" $CUPHY_YAML
    yq -i ".cuphydriver_config.cells[].eAxC_id_pucch = $MIMO_EAXC_IDS" $CUPHY_YAML

    SRS_EAXC_IDS="[32576, 32577, 32578, 32579, 32580, 32581, 32582, 32583, 32584, 32585, 32586, 32587, 32588, 32589, 32590, 32591, 32592, 32593, 32594, 32595, 32596, 32597, 32598, 32599, 32600, 32601, 32602, 32603, 32604, 32605, 32606, 32607, 32608, 32609, 32610, 32611, 32612, 32613, 32614, 32615, 32616, 32617, 32618, 32619, 32620, 32621, 32622, 32623, 32624, 32625, 32626, 32627, 32628, 32629, 32630, 32631, 32632, 32633, 32634, 32635, 32636, 32637, 32638, 32639]"
    yq -i ".cuphydriver_config.cells[].eAxC_id_srs = $SRS_EAXC_IDS" $CUPHY_YAML

    PRACH_EAXC_IDS="[32544, 32545, 32546, 32547, 32548, 32549, 32550, 32551, 32552, 32553, 32554, 32555, 32556, 32557, 32558, 32559]"
    yq -i ".cuphydriver_config.cells[].eAxC_id_prach = $PRACH_EAXC_IDS" $CUPHY_YAML

    yq -i '.cuphydriver_config.cells[].T1a_max_cp_ul_ns = 535000' $CUPHY_YAML
    yq -i '.cuphydriver_config.cells[].Ta4_min_ns_srs = 75729' $CUPHY_YAML
    yq -i '.cuphydriver_config.cells[].Ta4_max_ns_srs = 1200014' $CUPHY_YAML
    yq -i 'with(.cuphydriver_config.cells[].Tcp_adv_dl_ns; . = 324000 | . line_comment = "Override to 324000(324us) for M-MIMO 64T64R tests per ORAN IOT Profile-1 Entry1 requirements")' $CUPHY_YAML

#   sed -i "s/dlc_bfw_enable_divide_per_cell: [0-9]\+/dlc_bfw_enable_divide_per_cell: 1/g" $CUPHY_YAML
#   sed -i "s/ulc_bfw_enable_divide_per_cell: [0-9]\+/ulc_bfw_enable_divide_per_cell: 1/g" $CUPHY_YAML
#   sed -i "s/T1a_min_cp_ul_ns: [0-9]\+/T1a_min_cp_ul_ns: 285000/g" $CUPHY_YAML
#   sed -i "s/T1a_min_cp_dl_ns: [0-9]\+/T1a_min_cp_dl_ns: 419000/g" $CUPHY_YAML
#   sed -i "s/T1a_max_cp_dl_ns: [0-9]\+/T1a_max_cp_dl_ns: 669000/g" $CUPHY_YAML
else
    yq -i '.cuphydriver_config.bfw_c_plane_chaining_mode = 0' $CUPHY_YAML
fi

#--------------------------------------------------------------
# l2a MIMO configuration
#--------------------------------------------------------------
if [[ "$MUMIMO" == "ON" ]]; then
    echo "Setting enable_precoding to 0"
    yq -i '.enable_precoding = 0' $L2A_YAML
    echo "Setting l2a_allowed_latency to 200us"
    yq -i '.l2a_allowed_latency = 200000' $L2A_YAML
fi
#--------------------------------------------------------------
# RU emulator configuration
#--------------------------------------------------------------
yq -i '.ru_emulator.enable_beam_forming = 1' $RU_YAML
yq -i '.ru_emulator.aerial_fh_split_rx_tx_mempool = 1' $RU_YAML
yq -i '.ru_emulator.oam_cell_ctrl_cmd = 1' $RU_YAML

# Enable/Disable DLC testbench
if [ "$DLC_TB_ENABLED" -eq 1 ]; then
    yq -i '.ru_emulator.dlc_tb = 1' $RU_YAML
elif [ "$DLC_TB_ENABLED" -eq 0 ]; then
    yq -i '.ru_emulator.dlc_tb = 0' $RU_YAML
else
    echo "Error: --dlc-tb|-t must be either 0 or 1"
    exit 1
fi

if [[ "$MUMIMO" == "ON" ]]; then
    echo "Setting aerial_fh_mtu size to 8192"
    yq -i '.ru_emulator.aerial_fh_mtu = 8192' $RU_YAML
    yq -i '.ru_emulator.enable_mmimo = 1' $RU_YAML
    yq -i '.ru_emulator.fix_beta_dl = 0' $RU_YAML

    yq -i '.ru_emulator.aerial_fh_txq_size = 512' $RU_YAML
    yq -i '.ru_emulator.aerial_fh_rxq_size = 512' $RU_YAML

    EAXC_IDS="[32512, 32513, 32514, 32515, 32516, 32517, 32518, 32519, 32520, 32521, 32522, 32523, 32524, 32525, 32526, 32527]"
    if [[ "$ENABLE_32DL" -eq 1 ]]; then
       EAXC_IDS_PDSCH="[32512, 32513, 32514, 32515, 32516, 32517, 32518, 32519, 32520, 32521, 32522, 32523, 32524, 32525, 32526, 32527, 32528, 32529, 32530, 32531, 32532, 32533, 32534, 32535, 32536, 32537, 32538, 32539, 32540, 32541, 32542, 32543]"
    else
        EAXC_IDS_PDSCH="[32512, 32513, 32514, 32515, 32516, 32517, 32518, 32519, 32520, 32521, 32522, 32523, 32524, 32525, 32526, 32527]"
    fi
    yq -i ".ru_emulator.cell_configs[].eAxC_UL = $EAXC_IDS" $RU_YAML
    yq -i ".ru_emulator.cell_configs[].eAxC_DL = $EAXC_IDS_PDSCH" $RU_YAML

    PRACH_EAXC_IDS="[32544, 32545, 32546, 32547, 32548, 32549, 32550, 32551, 32552, 32553, 32554, 32555, 32556, 32557, 32558, 32559]"
    yq -i ".ru_emulator.cell_configs[].eAxC_prach_list = $PRACH_EAXC_IDS" $RU_YAML

    SRS_EAXC_IDS="[32576, 32577, 32578, 32579, 32580, 32581, 32582, 32583, 32584, 32585, 32586, 32587, 32588, 32589, 32590, 32591, 32592, 32593, 32594, 32595, 32596, 32597, 32598, 32599, 32600, 32601, 32602, 32603, 32604, 32605, 32606, 32607, 32608, 32609, 32610, 32611, 32612, 32613, 32614, 32615, 32616, 32617, 32618, 32619, 32620, 32621, 32622, 32623, 32624, 32625, 32626, 32627, 32628, 32629, 32630, 32631, 32632, 32633, 32634, 32635, 32636, 32637, 32638, 32639]"
    yq -i ".ru_emulator.cell_configs[].eAxC_srs_list = $SRS_EAXC_IDS" $RU_YAML
    yq -i '.ru_emulator.oran_timing_info.ul_u_plane_tx_offset_srs = 521' $RU_YAML #Providing a 400us lead time (see Ta4_max_ns_srs above) for RUE to transmit all the SRS U-Plane packets of a symbol before the end of the Tx window (6C peak workload)
    yq -i '.ru_emulator.oran_timing_info.dl_c_plane_timing_delay = 669' $RU_YAML
    yq -i '.ru_emulator.oran_timing_info.dl_c_plane_window_size = 250' $RU_YAML
    yq -i '.ru_emulator.oran_timing_info.ul_c_plane_timing_delay = 535' $RU_YAML
    yq -i '.ru_emulator.oran_timing_info.ul_c_plane_window_size = 250' $RU_YAML
    yq -i '.ru_emulator.split_srs_txq = 1' $RU_YAML
    yq -i '.ru_emulator.enable_srs_eaxcid_pacing = 1' $RU_YAML
    
    # SRS pacing parameters - pattern-specific configurations
    if [[ "$pattern_name" == "79a" || "$pattern_name" == "79b" ]]; then
        # Pattern 79a/79b: 4 SRS symbols in slot *3, 2 in *4, 2 in *5, 16 eAxC IDs per window
        yq -i '.ru_emulator.srs_pacing_s3_srs_symbols = 4' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_s4_srs_symbols = 2' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_s5_srs_symbols = 2' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_eaxcids_per_tx_window = 16' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_eaxcids_per_symbol = 64' $RU_YAML
    elif [[ "$pattern_name" == "81a" || "$pattern_name" == "81c" ]]; then
        # Pattern 81a/81c: 4 SRS symbols in slot *3, 8 eAxC IDs per window
        yq -i '.ru_emulator.srs_pacing_s3_srs_symbols = 4' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_s4_srs_symbols = 0' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_s5_srs_symbols = 0' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_eaxcids_per_tx_window = 8' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_eaxcids_per_symbol = 64' $RU_YAML
    elif [[ "$pattern_name" == "81b" || "$pattern_name" == "81d" ]]; then
        # Pattern 81b/81d: 2 SRS symbols in slot *3, 1 in *4, 1 in *5, 8 eAxC IDs per window
        yq -i '.ru_emulator.srs_pacing_s3_srs_symbols = 2' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_s4_srs_symbols = 1' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_s5_srs_symbols = 1' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_eaxcids_per_tx_window = 8' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_eaxcids_per_symbol = 64' $RU_YAML
    else
        # Default configuration for other MIMO patterns
        yq -i '.ru_emulator.srs_pacing_s3_srs_symbols = 2' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_s4_srs_symbols = 0' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_s5_srs_symbols = 0' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_eaxcids_per_tx_window = 4' $RU_YAML
        yq -i '.ru_emulator.srs_pacing_eaxcids_per_symbol = 64' $RU_YAML
    fi

    if [[ "$pattern_name" == "89" ]]; then
        yq -i '.ru_emulator.min_ul_cores_per_cell_mmimo = 2' $RU_YAML
    fi
fi

echo "Setting RU emulator network interfaces"

# Set first port values
yq -i ".ru_emulator.nics[0].nic_interface = \"$RU_PCIE_0\"" $RU_YAML
yq -i ".ru_emulator.peers[0].peerethaddr = \"$DU_MAC_ADDRESS_0\"" $RU_YAML

# Set all cells to use port 0 for single-port configuration
if [ "$NUM_PORTS" -eq 1 ]; then
    echo "Configuring single-port RU emulator"
    
    # Contract nics array to single entry if it has more than 1
    NIC_COUNT=$(yq '.ru_emulator.nics | length' $RU_YAML)
    if [ "$NIC_COUNT" -gt 1 ]; then
        echo "Contracting RU emulator nics array from $NIC_COUNT to 1 entry"
        yq -i '.ru_emulator.nics = [.ru_emulator.nics[0]]' $RU_YAML
    fi
    
    # Contract peers array to single entry if it has more than 1
    PEER_COUNT=$(yq '.ru_emulator.peers | length' $RU_YAML)
    if [ "$PEER_COUNT" -gt 1 ]; then
        echo "Contracting RU emulator peers array from $PEER_COUNT to 1 entry"
        yq -i '.ru_emulator.peers = [.ru_emulator.peers[0]]' $RU_YAML
    fi
    
    # Set all cells to use port 0
    echo "Setting all RU emulator cells to use port 0"
    CELL_COUNT=$(yq '.ru_emulator.cell_configs | length' $RU_YAML)
    for ((i=0; i<CELL_COUNT; i++)); do
        yq -i ".ru_emulator.cell_configs[$i].peer = 0" $RU_YAML
        yq -i ".ru_emulator.cell_configs[$i].nic = 0" $RU_YAML
    done
fi

# Configure second port if NUM_PORTS is 2
if [ "$NUM_PORTS" -eq 2 ]; then
    echo "Configuring dual-port RU emulator"
    
    # Check if second NIC entry exists, if not create it based on the first one
    NIC_COUNT=$(yq '.ru_emulator.nics | length' $RU_YAML)
    if [ "$NIC_COUNT" -lt 2 ]; then
        echo "Adding second NIC entry to RU emulator config"
        yq -i '.ru_emulator.nics[1] = .ru_emulator.nics[0]' $RU_YAML
    fi
    
    # Check if second peer entry exists, if not create it based on the first one
    PEER_COUNT=$(yq '.ru_emulator.peers | length' $RU_YAML)
    if [ "$PEER_COUNT" -lt 2 ]; then
        echo "Adding second peer entry to RU emulator config"
        yq -i '.ru_emulator.peers[1] = .ru_emulator.peers[0]' $RU_YAML
    fi
    
    # Set second port values
    yq -i ".ru_emulator.nics[1].nic_interface = \"$RU_PCIE_1\"" $RU_YAML
    yq -i ".ru_emulator.peers[1].peerethaddr = \"$DU_MAC_ADDRESS_1\"" $RU_YAML
    
    # Distribute cells across ports: even array index to port 0, odd array index to port 1
    echo "Distributing RU emulator cells across ports (even array index → port 0, odd array index → port 1)"
    CELL_COUNT=$(yq '.ru_emulator.cell_configs | length' $RU_YAML)
    for ((i=0; i<CELL_COUNT; i++)); do
        if [ $((i % 2)) -eq 0 ]; then
            # Even array index (0,2,4,6...) → port 0
            yq -i ".ru_emulator.cell_configs[$i].peer = 0" $RU_YAML
            yq -i ".ru_emulator.cell_configs[$i].nic = 0" $RU_YAML
        else
            # Odd array index (1,3,5,7...) → port 1
            yq -i ".ru_emulator.cell_configs[$i].peer = 1" $RU_YAML
            yq -i ".ru_emulator.cell_configs[$i].nic = 1" $RU_YAML
        fi
    done
fi

#--------------------------------------------------------------
# cuPHY controller network interface configuration
#--------------------------------------------------------------
echo "Setting cuPHY controller network interface values"

# Set first port NIC interface
yq -i ".cuphydriver_config.nics[0].nic = \"$DU_PCIE_0\"" $CUPHY_YAML

# Configure single-port cuPHY controller
if [ "$NUM_PORTS" -eq 1 ]; then
    echo "Configuring single-port cuPHY controller"
    
    # Contract nics array to single entry if it has more than 1
    NIC_COUNT=$(yq '.cuphydriver_config.nics | length' $CUPHY_YAML)
    if [ "$NIC_COUNT" -gt 1 ]; then
        echo "Contracting cuPHY controller nics array from $NIC_COUNT to 1 entry"
        yq -i '.cuphydriver_config.nics = [.cuphydriver_config.nics[0]]' $CUPHY_YAML
    fi
    
    # Set all cells to use port 0
    echo "Setting all cuPHY controller cells to use port 0"
    CELL_COUNT=$(yq '.cuphydriver_config.cells | length' $CUPHY_YAML)
    for ((i=0; i<CELL_COUNT; i++)); do
        yq -i ".cuphydriver_config.cells[$i].nic = \"$DU_PCIE_0\"" $CUPHY_YAML
        yq -i ".cuphydriver_config.cells[$i].src_mac_addr = \"$DU_MAC_ADDRESS_0\"" $CUPHY_YAML
    done
fi

# Configure dual-port cuPHY controller
if [ "$NUM_PORTS" -eq 2 ]; then
    echo "Configuring dual-port cuPHY controller"
    
    # Check if second NIC entry exists, if not create it based on the first one
    NIC_COUNT=$(yq '.cuphydriver_config.nics | length' $CUPHY_YAML)
    if [ "$NIC_COUNT" -lt 2 ]; then
        echo "Adding second NIC entry to cuPHY controller config"
        yq -i '.cuphydriver_config.nics[1] = .cuphydriver_config.nics[0]' $CUPHY_YAML
    fi
    
    # Set the second port NIC interface
    yq -i ".cuphydriver_config.nics[1].nic = \"$DU_PCIE_1\"" $CUPHY_YAML
    
    # Distribute cells across ports: even array index to port 0, odd array index to port 1
    echo "Distributing cuPHY controller cells across ports (even array index → port 0, odd array index → port 1)"
    CELL_COUNT=$(yq '.cuphydriver_config.cells | length' $CUPHY_YAML)
    for ((i=0; i<CELL_COUNT; i++)); do
        if [ $((i % 2)) -eq 0 ]; then
            # Even array index (0,2,4,6...) → first port (DU_PCIE_0)
            yq -i ".cuphydriver_config.cells[$i].nic = \"$DU_PCIE_0\"" $CUPHY_YAML
            yq -i ".cuphydriver_config.cells[$i].src_mac_addr = \"$DU_MAC_ADDRESS_0\"" $CUPHY_YAML
        else
            # Odd array index (1,3,5,7...) → second port (DU_PCIE_1)
            yq -i ".cuphydriver_config.cells[$i].nic = \"$DU_PCIE_1\"" $CUPHY_YAML
            yq -i ".cuphydriver_config.cells[$i].src_mac_addr = \"$DU_MAC_ADDRESS_1\"" $CUPHY_YAML
        fi
    done
fi

#--------------------------------------------------------------
# Setting time window to accept packets on DU
#--------------------------------------------------------------
# transfer window can be anywhere between T0+50us to T0+331us, for perf tests reduce this time window
if [[ $CUPHY_HOST_TYPE == "_CG1" || $CUPHY_HOST_TYPE == "_GL4" ]];then
    # 51 us transfer window on GH, T0+331-51->ul_u_plane_tx_offset = 280
    echo "Setting ul_u_plane_tx_offset to 280"
    yq -i '.ru_emulator.oran_timing_info.ul_u_plane_tx_offset = 280' $RU_YAML
else
    # 100 us transfer window on ROYB, T0+331-100->ul_u_plane_tx_offset = 231
    echo "Setting ul_u_plane_tx_offset to 231"
    yq -i '.ru_emulator.oran_timing_info.ul_u_plane_tx_offset = 231' $RU_YAML
fi


#--------------------------------------------------------------
# Logging configuration
#--------------------------------------------------------------
yq -i '.cuphydriver_config.ul_order_timeout_log_interval_ns = 0' $CUPHY_YAML
yq -i '.cuphydriver_config.ul_order_timeout_gpu_log_enable = 1' $CUPHY_YAML
yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.UL_PACKET_SUMMARY") .shm_level = 5' $NVLOG_YAML
yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.SRS_PACKET_SUMMARY") .shm_level = 5' $NVLOG_YAML

# Enable detailed tracing and processing time logs unless reduced-logging mode is enabled
if [ "$REDUCED_LOGGING" != true ]; then
    yq -i '.cuphydriver_config.enable_cpu_task_tracing = 1' $CUPHY_YAML
    yq -i '.cuphydriver_config.enable_compression_tracing = 1' $CUPHY_YAML
    yq -i '.cuphydriver_config.enable_prepare_tracing = 1' $CUPHY_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "L2A.PROCESSING_TIMES") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "MAC.PROCESSING_TIMES") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "L2A.TICK_TIMES") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.MAP_DL") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.MAP_UL") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.FUNC_DL") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.FUNC_UL") .shm_level = 5' $NVLOG_YAML
else
    echo "Reduced logging mode enabled - skipping detailed tracing and processing time logs"
fi

# Enable CUPTI tracing if requested
if [ "$CUPTI_TRACING" = true ]; then
    echo "Enabling CUPTI tracing"
    yq -i '.cuphydriver_config.cupti_enable_tracing = 1' $CUPHY_YAML
fi

#Logging/cuphy settings for latency summary
#ToDo may be add an option to disable extended logging?
if [ "$LOG_NIC_TIMINGS" = true ]; then

    echo "Enabled extended logging for latency summary"
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.SYMBOL_TIMINGS") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.SYMBOL_TIMINGS_SRS") .shm_level = 5' $NVLOG_YAML

    yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.PACKET_TIMINGS") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "DRV.PACKET_TIMINGS_SRS") .shm_level = 5' $NVLOG_YAML

    yq -i '.nvlog.nvlog_tags[] |= select(.* == "RU.SYMBOL_TIMINGS") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "FH.TX_TIMINGS") .shm_level = 5' $NVLOG_YAML
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "RU.TX_TIMINGS_SUM") .shm_level = 5' $NVLOG_YAML

    yq -i '.cuphydriver_config.ul_rx_pkt_tracing_level = 1' $CUPHY_YAML
    yq -i '.cuphydriver_config.ul_rx_pkt_tracing_level_srs = 1' $CUPHY_YAML
    
fi

#Logging settings for RU C-plane worker tracing
if [ "$RU_WORKER_TRACING" = true ]; then

    echo "Enabled RU C-plane worker tracing logs"
    yq -i '.nvlog.nvlog_tags[] |= select(.* == "RU.CP_WORKER_TRACING") .shm_level = 5' $NVLOG_YAML

    yq -i '.ru_emulator.enable_cplane_worker_tracing = 1' $RU_YAML
    
fi

#--------------------------------------------------------------
# early-HARQ related changes
#--------------------------------------------------------------
if [ "$EARLY_HARQ_ENABLED" -eq 1 ]; then
    yq -i '.indicationPerSlot.uciIndPerSlot = 2' $TESTMAC_YAML
elif [ "$EARLY_HARQ_ENABLED" -eq 0 ]; then
    # For non-early-HARQ
    yq -i '.indicationPerSlot.uciIndPerSlot = 0' $TESTMAC_YAML
else
    echo "Error: --ehq|-q must be either 0 or 1"
    exit 1
fi

#--------------------------------------------------------------
# BFP and compression method related changes
#--------------------------------------------------------------
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

        yq -i ".cuphydriver_config.bfw_beta_prescaler = 16384" $CUPHY_YAML
    fi
else
    echo "Error: Compression method must be either 1 (BFP) or 4 (mod compression)"
    exit 1
fi

if [[ $CONTROLLER_MODE == "F08_GL4" ]];then
    yq -i '.cuphydriver_config.gpu_init_comms_via_cpu = 1' $CUPHY_YAML
    echo "Setting gpu_init_comms_via_cpu to 1"
fi

#==============================================================
# Multi-L2 Configuration
#==============================================================
if [[ "$MULTI_L2" == true ]]; then
    # Create second testMAC config file
    TESTMAC1_YAML=${CONFIG_DIR}/cuPHY-CP/testMAC/testMAC/test_mac_config_1.yaml
    cp "$TESTMAC_YAML" "$TESTMAC1_YAML"

    # Configure nvipc_config_file in L2A_YAML to use nvipc_multi_instances.yaml
    MULTI_NVIPC_YAML=${CONFIG_DIR}/cuPHY-CP/cuphycontroller/config/nvipc_multi_instances.yaml
    if [[ ! -f "$MULTI_NVIPC_YAML" ]]; then
        echo "Error: $MULTI_NVIPC_YAML not found"
        exit 1
    fi

    yq -i '.nvipc_config_file = "nvipc_multi_instances.yaml"' "$L2A_YAML"

    # Calculate cell masks and lists for splitting cells between two instances
    half_cell_num=$(((NUM_CELLS+1)/2))
    cell_mask0=0
    cell_mask1=0
    cell_list0=""
    cell_list1=""

    # Build cell_mask0 and cell_list0 for first half of cells
    for cell_id in $(seq 0 $(($half_cell_num-1))); do
        bit=$((2 ** $cell_id))
        cell_mask0=$(($cell_mask0 | $bit))
        if [ "$cell_list0" != "" ]; then
            cell_list0="${cell_list0}, "
        fi
        cell_list0="${cell_list0}$cell_id"
        # echo "cell_id=${cell_id} half_cell_num=$half_cell_num cell_list0=$cell_list0"
    done

    # Build cell_mask1 and cell_list1 for second half of cells
    for cell_id in $(seq $half_cell_num $(($NUM_CELLS-1))); do
        bit=$((2 ** $cell_id))
        cell_mask1=$(($cell_mask1 | $bit))
        if [ "$cell_list1" != "" ]; then
            cell_list1="${cell_list1}, "
        fi
        cell_list1="${cell_list1}$cell_id"
    done

    # Format as hex
    printf -v ML2_CELL_MASK0 "0x%02X" $cell_mask0
    printf -v ML2_CELL_MASK1 "0x%02X" $cell_mask1
    ML2_CELL_LIST0="[$cell_list0]"
    ML2_CELL_LIST1="[$cell_list1]"

    echo "Multi-L2 cell configuration:"
    echo "  First instance:  cell_mask=$ML2_CELL_MASK0 cell_list=$ML2_CELL_LIST0"
    echo "  Second instance: cell_mask=$ML2_CELL_MASK1 cell_list=$ML2_CELL_LIST1"

    # Update nvipc_multi_instances.yaml with cell lists
    yq -i ".transport[0].phy_cells = $ML2_CELL_LIST0" "$MULTI_NVIPC_YAML" 2>/dev/null || echo "Warning: Failed to update transport 0 phy_cells"
    yq -i ".transport[1].phy_cells = $ML2_CELL_LIST1" "$MULTI_NVIPC_YAML" 2>/dev/null || echo "Warning: Failed to update transport 1 phy_cells"

    # Configure second testMAC instance
    yq -i '.transport.shm_config.prefix = "nvipc1"' "$TESTMAC1_YAML"
    yq -i '.log_name = "testmac1.log"' "$TESTMAC1_YAML"
    yq -i '.oam_server_addr = "0.0.0.0:50053"' "$TESTMAC1_YAML"

    cd $cuBB_SDK/cubb_scripts/autoconfig/l1_core_config
    PHYSICAL_CORES_YAML=${CONFIG_DIR}/testBenches/phase4_test_scripts/physical_cores_du.yaml

    # Auto-assign CPU cores for second testMAC instance
    python3 auto_override_yaml_cores.py "$PHYSICAL_CORES_YAML" -1 "$TESTMAC1_YAML"
    python3 auto_override_yaml_cores.py "$PHYSICAL_CORES_YAML" -m "$MULTI_NVIPC_YAML"

    second_half_cell_num=$((NUM_CELLS-half_cell_num))
    echo "Configured Multi-L2 test with ${half_cell_num} + ${second_half_cell_num} = ${NUM_CELLS} cells"
else
    ML2_CELL_MASK0=""
    ML2_CELL_MASK1=""
    ML2_CELL_LIST0=""
    ML2_CELL_LIST1=""
    TESTMAC1_YAML=""
    yq -i '.nvipc_config_file = "null"' "$L2A_YAML"
    echo "Configured Single-L2 test with total ${NUM_CELLS} cells"
fi

if [ "$WORK_CANCEL_MODE" -eq 2 ]; then
    WC_MODE="enabled using device graphs"
elif [ "$WORK_CANCEL_MODE" -eq 1 ]; then
    WC_MODE="enabled using conditional nodes"
else
    WC_MODE="disabled"
fi

if [ "$EARLY_HARQ_ENABLED" -eq 1 ]; then
    EHQ_STATUS=enabled
else
    EHQ_STATUS=disabled
fi

if [ "$DEVICE_GRAPH_LAUNCH_ENABLED" -eq 1 ]; then
    DGL_STATUS=enabled
else
    DGL_STATUS=disabled
fi

if [ "$USE_GREEN_CONTEXT" -eq 1 ]; then
    GC_STATUS=enabled
else
    GC_STATUS=disabled
fi

#
# Final GPU-based SM cap to handle RTX 4500-class GPUs (SM limit ~82)
# This override is applied at the end to ensure it takes effect regardless of earlier branches.
#
if [[ "$CONTROLLER_MODE" != "F08_GL4" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null)
        if echo "$GPU_NAME" | grep -qiE 'RTX[^,]*4500'; then
            echo "Detected GPU '$GPU_NAME' → capping mps_sm_pusch and mps_sm_pdsch to 80"
	    if [[ "$MUMIMO" == "ON" ]]; then
                yq -i '.cuphydriver_config.mps_sm_pusch = 40' "$CUPHY_YAML"
	    else
                yq -i '.cuphydriver_config.mps_sm_pusch = 80' "$CUPHY_YAML"
	    fi
            yq -i '.cuphydriver_config.mps_sm_pdsch = 80' "$CUPHY_YAML"
        fi
    fi
fi

PATTERN=$pattern_name
for var in ${TEST_VARS}; do
    printf "%-20s : %-20s\n" "$var" "${!var}"
done

if [[ "$FORCE" != true || "$TEST_CONFIG_DONE" != 1 ]]; then
    # Running the script for the first time - need to add TEST_VARS and TEST_CONFIG_DONE.
    # Otherwise, VARS already has these variables when sourcing the TEST_CONFIG_FILE.
    VARS="$VARS $TEST_VARS TEST_CONFIG_DONE"
fi

# update test_config_summary.sh
TEST_CONFIG_DONE=1
> "$TEST_CONFIG_FILE"  # Clear the file before writing
for var in ${VARS}; do
    echo "$var=\"${!var}\"" >> "$TEST_CONFIG_FILE"
done
chmod +x "$TEST_CONFIG_FILE"
