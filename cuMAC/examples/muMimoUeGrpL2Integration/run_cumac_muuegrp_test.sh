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

#--------------------------------------------------------------------
# This script is to be run on DU side for cuMAC muUeGrp_L2_Integration testing
#--------------------------------------------------------------------

script_name=run_cumac_muuegrp_test.sh

TESTCASE=""
TV_INDEX=""

# Parse test configurations
while getopts ":c:C:v:V:" opt; do
  case $opt in
    c|C)
      TESTCASE=$OPTARG
      ;;
    v|V)
      TV_INDEX=$OPTARG
      ;;
    \?)
      echo "Unknown parameter: -$OPTARG" >&2
      ;;
  esac
done

# required parameters to run this script
if [[ -z $TESTCASE || -z $TV_INDEX ]]; then
   echo "testcase and tv_index are required parameters for muUeGrp_L2_Integration tests."
   echo "Usage: $script_name -c muUeGrp_L2_Integration -v <TV_INDEX>"
   exit 1
fi

if [[ "${TESTCASE}" != "muUeGrp_L2_Integration" ]]; then
   echo "Error: Unsupported testcase '${TESTCASE}'. Only 'muUeGrp_L2_Integration' is supported."
   exit 1
fi

echo "TESTCASE:$TESTCASE"
echo "TV_INDEX:$TV_INDEX"

if [ -z "$cuBB_SDK" ]; then
    cuBB_SDK=/opt/nvidia/cuBB
fi
if [ -z "$LOG_PATH" ]; then
    LOG_PATH=/home/aerial/nfs/Log/Sanity/cuMAC/latest
fi
mkdir -p "${LOG_PATH:?LOG_PATH must be set}" && rm -rf "${LOG_PATH:?}"/*
chmod 755 "${LOG_PATH}"
cd "${LOG_PATH}" || exit 1

test_time=$(date -u "+%Y%m%d_%H%M%S")
muUeGrp_config_file="${cuBB_SDK}/cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/config.yaml"


function usage() {
    echo "Usage: $script_name -c muUeGrp_L2_Integration -v <TV_INDEX>"
    echo "    TESTCASE:    muUeGrp_L2_Integration"
    echo "    TV_INDEX:    Test vector index from CSV (e.g., 0001-0480)"
    echo "                 The CSV contains all 480 parameter combinations"
    echo "                 and is auto-generated on first use"
    echo ""
    echo "Example1: ./$script_name -c muUeGrp_L2_Integration -v 0001"
    echo "Example2: ./$script_name -c muUeGrp_L2_Integration -v 0385"
}


function muUeGrp_params_comb() {
    local tv_index=$1
    local config_file=$2
    local csv_file="${cuBB_SDK}/cuMAC/scripts/muuegrp_tv_parameters.csv"

    echo "Loading test parameters from CSV for index: ${tv_index}"

    # Generate CSV if it doesn't exist
    if [[ ! -f ${csv_file} ]]; then
        echo "CSV file not found. Generating muUeGrp parameter combinations..."
        echo "Generating 480 combinations: 4 blocks x 12 primary x 10 secondary"

        local csv_dir=$(dirname "${csv_file}")
        mkdir -p "${csv_dir}"

        # 4 blocks: ENABLE_L1_L2_MEM_SHARING x TDD_PATTERN
        # Format: mem_sharing,tdd_pattern
        local blocks=(
            "false,SDDDS"
            "false,SSSSS"
            "true,SDDDS"
            "true,SSSSS"
        )

        # Primary dimension arrays
        local num_cell=(1 6)
        local num_bs_ant_port=(32 64)
        local num_srs_ue_per_cell=(32 128 256)

        # Write CSV header
        echo "Index,ENABLE_L1_L2_MEM_SHARING,TDD_PATTERN,NUM_CELL,NUM_BS_ANT_PORT,NUM_SRS_UE_PER_CELL,NUM_TIME_SLOTS,NUM_SUBBAND,NUM_PRG_SAMP_PER_SUBBAND,NUM_SRS_UE_PER_SLOT,MAX_NUM_UE_SCHEDULED_PER_CELL_TTI,MAX_NUM_UE_FOR_GRP_PER_CELL" > "${csv_file}"

        # 10 secondary parameter combinations
        # Format: time_slots,subband,prg_samp,ue_per_slot,ue_scheduled_tti,ue_for_grp
        local secondary_combos=(
            "20,1,1,8,16,32"
            "50,4,2,32,64,64"
            "50,1,1,8,64,64"
            "20,4,2,32,16,32"
            "20,4,1,8,16,64"
            "50,1,2,32,16,32"
            "20,1,2,8,16,64"
            "50,4,1,32,64,64"
            "20,4,2,8,64,64"
            "50,1,1,32,16,32"
        )

        # Generate combinations: 4 blocks x 12 primary x 10 secondary = 480 total
        local index=1
        for block in "${blocks[@]}"; do
            IFS=',' read -r mem_sharing tdd_pattern <<< "$block"
            for cell in "${num_cell[@]}"; do
                for ant in "${num_bs_ant_port[@]}"; do
                    for ue in "${num_srs_ue_per_cell[@]}"; do
                        for combo in "${secondary_combos[@]}"; do
                            IFS=',' read -r ts sb prg slot tti grp <<< "$combo"
                            printf "%04d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n" \
                                $index "$mem_sharing" "$tdd_pattern" \
                                $cell $ant $ue $ts $sb $prg $slot $tti $grp >> "${csv_file}"
                            ((index++))
                        done
                    done
                done
            done
        done

        echo "Generated $((index-1)) parameter combinations in ${csv_file}"
    fi

    # Read the specific test vector from CSV
    local params_line
    params_line=$(awk -F',' -v idx="${tv_index}" '$1 == idx {print $2","$3","$4","$5","$6","$7","$8","$9","$10","$11","$12}' "${csv_file}" | tr -d '\r')

    if [ -z "${params_line}" ]; then
        echo "Error: Test vector index ${tv_index} not found in CSV"
        return 1
    fi

    # Parse the parameters
    IFS=',' read -r ENABLE_L1_L2_MEM_SHARING TDD_PATTERN NUM_CELL NUM_BS_ANT_PORT NUM_SRS_UE_PER_CELL NUM_TIME_SLOTS NUM_SUBBAND NUM_PRG_SAMP_PER_SUBBAND NUM_SRS_UE_PER_SLOT MAX_NUM_UE_SCHEDULED_PER_CELL_TTI MAX_NUM_UE_FOR_GRP_PER_CELL <<< "${params_line}"

    # Display loaded parameters
    echo "Loaded parameters from CSV for test index ${tv_index}:"
    echo "  ENABLE_L1_L2_MEM_SHARING=${ENABLE_L1_L2_MEM_SHARING}"
    echo "  TDD_PATTERN=${TDD_PATTERN}"
    echo "  NUM_CELL=${NUM_CELL}"
    echo "  NUM_BS_ANT_PORT=${NUM_BS_ANT_PORT}"
    echo "  NUM_SRS_UE_PER_CELL=${NUM_SRS_UE_PER_CELL}"
    echo "  NUM_TIME_SLOTS=${NUM_TIME_SLOTS}"
    echo "  NUM_SUBBAND=${NUM_SUBBAND}"
    echo "  NUM_PRG_SAMP_PER_SUBBAND=${NUM_PRG_SAMP_PER_SUBBAND}"
    echo "  NUM_SRS_UE_PER_SLOT=${NUM_SRS_UE_PER_SLOT}"
    echo "  MAX_NUM_UE_SCHEDULED_PER_CELL_TTI=${MAX_NUM_UE_SCHEDULED_PER_CELL_TTI}"
    echo "  MAX_NUM_UE_FOR_GRP_PER_CELL=${MAX_NUM_UE_FOR_GRP_PER_CELL}"

    # Export the parameters so they're available globally
    export ENABLE_L1_L2_MEM_SHARING TDD_PATTERN NUM_TIME_SLOTS NUM_CELL NUM_BS_ANT_PORT NUM_SRS_UE_PER_CELL NUM_SUBBAND NUM_PRG_SAMP_PER_SUBBAND NUM_SRS_UE_PER_SLOT MAX_NUM_UE_SCHEDULED_PER_CELL_TTI MAX_NUM_UE_FOR_GRP_PER_CELL

    # Update configuration file if provided
    if [[ -n "${config_file}" ]]; then
        echo "Updating configuration file: ${config_file}"

        # Normalize line endings to LF so yq can parse the file
        sed -i 's/\r//' "${config_file}"

        # Update boolean ENABLE_L1_L2_MEM_SHARING
        if [[ "${ENABLE_L1_L2_MEM_SHARING}" == "true" ]]; then
            yq -i ".ENABLE_L1_L2_MEM_SHARING = true" "${config_file}"
        else
            yq -i ".ENABLE_L1_L2_MEM_SHARING = false" "${config_file}"
        fi
        echo "  ENABLE_L1_L2_MEM_SHARING: ${ENABLE_L1_L2_MEM_SHARING}"

        # Update string TDD_PATTERN
        yq -i ".TDD_PATTERN = \"${TDD_PATTERN}\"" "${config_file}"
        echo "  TDD_PATTERN: ${TDD_PATTERN}"

        # Define numeric parameter mapping: variable_name:yaml_key
        declare -a params=(
            "NUM_CELL:NUM_CELL"
            "NUM_BS_ANT_PORT:NUM_BS_ANT_PORT"
            "NUM_SRS_UE_PER_CELL:NUM_SRS_UE_PER_CELL"
            "NUM_TIME_SLOTS:NUM_TIME_SLOTS"
            "NUM_SUBBAND:NUM_SUBBAND"
            "NUM_PRG_SAMP_PER_SUBBAND:NUM_PRG_SAMP_PER_SUBBAND"
            "NUM_SRS_UE_PER_SLOT:NUM_SRS_UE_PER_SLOT"
            "MAX_NUM_UE_SCHEDULED_PER_CELL_TTI:MAX_NUM_UE_SCHEDULED_PER_CELL_TTI"
            "MAX_NUM_UE_FOR_GRP_PER_CELL:MAX_NUM_UE_FOR_GRP_PER_CELL"
        )

        for param in "${params[@]}"; do
            var_name="${param%%:*}"
            yaml_key="${param##*:}"
            var_value="${!var_name}"

            if [[ -n "${var_value}" ]]; then
                echo "  ${yaml_key}: ${var_value}"
                yq -i ".${yaml_key} = ${var_value}" "${config_file}"
            fi
        done
    fi

    return 0
}

function cumac_muuegrp_l2_integration_test(){
     BUILD="build.$(uname -m)"
     cmd1="cd ${cuBB_SDK} && sudo ./${BUILD}/cuMAC/examples/muMimoUeGrpL2Integration/l1_muUeGrp_test"
     cmd2="cd ${cuBB_SDK} && sudo ./${BUILD}/cuMAC/examples/muMimoUeGrpL2Integration/cumac_muUeGrp_test"
     cmd3="cd ${cuBB_SDK} && sudo ./${BUILD}/cuMAC/examples/muMimoUeGrpL2Integration/l2_muUeGrp_test"

     echo "Start cuMAC muMimoUeGrpL2Integration tests ..."

     # Display updated config (already updated by muUeGrp_params_comb)
     echo "Current configuration:"
     cat ${muUeGrp_config_file}

     # Start L1 first - it is PRIMARY for L1-cuMAC semaphore and memory pools
     echo "Start l1_muUeGrp_test  ..."
     screen -L -t l1_muuegrp_test -dmS l1_muuegrp_test bash -c "$cmd1"
     # Brief delay so L1 can create semaphores and memory pools before cuMAC attaches
     sleep 2

     # Start cuMAC (SECONDARY) after L1 has created shared resources
     echo "Start cumac_muUeGrp_test  ..."
     screen -L -t cumac_muuegrp_test -dmS cumac_muuegrp_test bash -c "$cmd2"
     echo "Start l2_muUeGrp_test  ..."
     screen -L -t l2_muuegrp_test -dmS l2_muuegrp_test bash -c "$cmd3"
     sleep 5
     ${cuBB_SDK}/cuMAC/scripts/getcore.sh > ${LOG_PATH}/gnb_core.log
     sleep ${NUM_TIME_SLOTS}

     #wait_test_completed "${LOG_PATH}" "cumac_muuegrp_test" "cuMAC RECV: time slot $((${NUM_TIME_SLOTS}-1)), received messages of ${NUM_CELL} cells"
     sleep 10
     screen -X -S l1_muuegrp_test quit
     screen -X -S cumac_muuegrp_test quit
     screen -X -S l2_muuegrp_test quit
     sudo -E killall -9 -q l1_muUeGrp_test || true
     sudo -E killall -9 -q cumac_muUeGrp_test || true
     sudo -E killall -9 -q l2_muUeGrp_test || true
}

function dump_config(){
     echo "Dump configuration ..."
     cp ${muUeGrp_config_file} ${LOG_PATH}
}

function copy_log(){
     echo "Copy logs to ${log_folder} ..."
     if [[ -z "${LOG_PATH}" || "${LOG_PATH}" == "/" ]]; then
         echo "Error: copy_log: LOG_PATH must be set and must not be '/' (LOG_PATH='${LOG_PATH:-}')." >&2
         return 1
     fi
     local parent_dir
     parent_dir=$(dirname -- "${LOG_PATH}") || {
         echo "Error: copy_log: dirname failed for LOG_PATH='${LOG_PATH}'." >&2
         return 1
     }
     if ! cd -- "${parent_dir}"; then
         echo "Error: copy_log: cannot cd to parent_dir='${parent_dir}' (LOG_PATH='${LOG_PATH}')." >&2
         return 1
     fi
     cp -r "${LOG_PATH}" "${log_folder}"

     # Check for all 7 thread success messages
     l1_log_content=$(cat ${LOG_PATH}/screenlog_l1_muuegrp_test.log 2>/dev/null)
     cumac_log_content=$(cat ${LOG_PATH}/screenlog_cumac_muuegrp_test.log 2>/dev/null)
     l2_log_content=$(cat ${LOG_PATH}/screenlog_l2_muuegrp_test.log 2>/dev/null)
     all_log_content="${l1_log_content}${cumac_log_content}${l2_log_content}"

     l2_main_pass=$(echo "$all_log_content" | grep "L2-MAIN: test completed successfully")
     l2_recv_pass=$(echo "$all_log_content" | grep "L2-cuMAC RECV: test completed successfully")
     cumac_main_pass=$(echo "$all_log_content" | grep "cuMAC-MAIN: test completed successfully")
     if [[ "${ENABLE_L1_L2_MEM_SHARING}" == "false" ]]; then
         # 26-1:Workaround for 6018336 temporarily skip the UE pairing solution CPU verification for No MemShare mode
         cumac_cpu_verify_pass="Skip the UE pairing solution CPU verification for No MemShare mode"
     else
         cumac_cpu_verify_pass=$(echo "$all_log_content" | grep "cuMAC-MAIN: UE pairing solution CPU verification - test completed successfully")
     fi
     cumac_recv_pass=$(echo "$all_log_content" | grep "cuMAC-L2 RECV: test completed successfully")
     l1_main_pass=$(echo "$all_log_content" | grep "L1-MAIN: test completed successfully")
     l1_recv_pass=$(echo "$all_log_content" | grep "L1-L2 RECV: test completed successfully")

     local pass_flag=0
     if [[ -n "${l2_main_pass}" && -n "${l2_recv_pass}" && -n "${cumac_main_pass}" && -n "${cumac_cpu_verify_pass}" && -n "${cumac_recv_pass}" && -n "${l1_main_pass}" && -n "${l1_recv_pass}" ]]; then
          new_log_folder="${log_folder}_PASS"
          echo "TEST PASSED: All 7 threads completed successfully."
          pass_flag=1
     else
          new_log_folder="${log_folder}_FAIL"
          echo "TEST FAILED: One or more threads did not complete successfully."
          [[ -z "${l2_main_pass}" ]]         && echo "  Missing: L2-MAIN: test completed successfully"
          [[ -z "${l2_recv_pass}" ]]          && echo "  Missing: L2-cuMAC RECV: test completed successfully"
          [[ -z "${cumac_main_pass}" ]]       && echo "  Missing: cuMAC-MAIN: test completed successfully"
          [[ -z "${cumac_cpu_verify_pass}" ]] && echo "  Missing: cuMAC-MAIN: UE pairing solution CPU verification - test completed successfully"
          [[ -z "${cumac_recv_pass}" ]]       && echo "  Missing: cuMAC-L2 RECV: test completed successfully"
          [[ -z "${l1_main_pass}" ]]          && echo "  Missing: L1-MAIN: test completed successfully"
          [[ -z "${l1_recv_pass}" ]]          && echo "  Missing: L1-L2 RECV: test completed successfully"
     fi
     mv ${log_folder} ${new_log_folder}
     if [[ -n "${HTML_SERVER}" && -n "${HTML_SERVER_PASSWD}" ]]; then
        echo "Copy the log folder to html server for single flow passing"
        sshpass -p "${HTML_SERVER_PASSWD}" scp -o StrictHostKeyChecking=no -r "${new_log_folder}" "aerial@${HTML_SERVER}:/var/www/html/Automation/cumac/nfs_log"
     fi
     return $((1 - pass_flag))
}

function wait_test_completed()
{
    local log_path=$1
    local module=$2 # cumac_muuegrp_test, l2_muuegrp_test
    local key_words=$3
    local log_file="$log_path/screenlog_${module}.log"

    echo "wait_test_completed: $log_path $module $key_words, log_file: ${log_file}"

    local time_out=180
    local start_time=$(date +%s)
    local counter=0

    while [[ $(($(date +%s) - $start_time)) -lt $time_out ]]; do
        if [[ -f "$log_file" ]] && grep -qE "$key_words" "$log_file"; then
            echo "$(date): ${module} test completed successfully in ${counter} seconds"
            # Show a few matching lines for debugging
            grep -E "$key_words" "$log_file" | tail -n 1
            return 0
        else
            # Print status message every 20 seconds
            if [[ $((counter % 20)) -eq 0 ]]; then
                echo "Waiting for pattern '$key_words' in $log_file for ${counter} seconds ..."
            fi
            counter=$((counter + 1))
            sleep 1
        fi
    done

    echo "Timeout: Test completed pattern '$key_words' not found in $log_file after ${time_out}s"
    return 1
}

############################################################################
# main: run cumac muUeGrp_L2_Integration testing (skipped when sourced)
############################################################################
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ -z "${TV_INDEX}" ]]; then
        echo "Error: -v <TV_INDEX> is required for muUeGrp_L2_Integration tests"
        echo "Please specify a test vector index (0001-0480)"
        usage
        exit 1
    fi

    # Snapshot pre-TV config before muUeGrp_params_comb mutates it in-place (yq -i / sed -i)
    if [[ -f "${muUeGrp_config_file}" ]]; then
        cp -- "${muUeGrp_config_file}" "${muUeGrp_config_file}_ori"
    fi

    # Load parameters from CSV and update config
    if ! muUeGrp_params_comb "${TV_INDEX}" "${muUeGrp_config_file}"; then
        echo "Failed to load parameters from CSV for index ${TV_INDEX}"
        exit 1
    fi

    # Set log folder name
    mem_tag="NoMemShare"
    [[ "${ENABLE_L1_L2_MEM_SHARING}" == "true" ]] && mem_tag="MemShare"
    export log_folder=${test_time}_cuMAC_muUeGrp_${mem_tag}_${TDD_PATTERN}_${NUM_CELL}Cell_${NUM_BS_ANT_PORT}Ant_${NUM_SRS_UE_PER_CELL}UE_${NUM_SUBBAND}Sb_${NUM_PRG_SAMP_PER_SUBBAND}Prg_${NUM_SRS_UE_PER_SLOT}SrsPerSlot_${MAX_NUM_UE_SCHEDULED_PER_CELL_TTI}UePerTTI_${MAX_NUM_UE_FOR_GRP_PER_CELL}UeGrpPerCell_${NUM_TIME_SLOTS}Ts_Test
    cumac_muuegrp_l2_integration_test
    dump_config
    copy_log
    copy_log_status=$?
    if [[ -f "${muUeGrp_config_file}_ori" ]]; then
        mv -- "${muUeGrp_config_file}_ori" "${muUeGrp_config_file}"
    fi
    exit "${copy_log_status}"
fi
