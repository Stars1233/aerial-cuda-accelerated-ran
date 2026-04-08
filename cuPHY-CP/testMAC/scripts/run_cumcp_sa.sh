#!/bin/bash

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

script_name=${0##*/}
function usage {
    echo "Usage: $script_name <duration> <test_group> <test_case> [-a|--alloc_type=<value>]"
    echo "    duration:    The duration to checking status before exit. Unit: second"
    echo "    test_group:  F08"
    echo "    test_case:   1C_60, 2C_60, 3C_60, ..."
    echo "    -a, --alloc_type: cuMAC allocType type parameter (default: 1)"
    echo ""
    echo "Examples:"
    echo "    $script_name 20 F08 8C_60 -a 1"
    echo "    $script_name 20 F08 8C_60 --alloc_type=1"
    echo "    $script_name 20 F08 8C_60 --alloc_type 0"
}

# Initialize default values
alloc_type=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --alloc_type=*)
            alloc_type="${1#*=}"
            shift
            ;;
        -a|--alloc_type)
            if [[ -n "$2" && "$2" != -* ]]; then
                alloc_type="$2"
                shift 2
            else
                echo "Error: -a/--alloc_type requires a value"
                usage
                exit 1
            fi
            ;;
        -*)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
        *)
            # Positional arguments
            if [[ -z "$duration" ]]; then
                duration="$1"
            elif [[ -z "$test_group" ]]; then
                test_group="$1"
            elif [[ -z "$test_case" ]]; then
                test_case="$1"
            else
                echo "Error: Too many positional arguments"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Check required positional parameters
if [[ -z "$duration" || -z "$test_group" || -z "$test_case" ]]; then
    usage
    exit 1
fi

echo "Parameters: duration=$duration, test_group=$test_group, test_case=$test_case, alloc_type=$alloc_type"

script_start=$(date +%s)

if [ "$cuBB_SDK" = "" ]; then
    echo "Please set cuBB_SDK first"
    exit 1
fi

LOCAL_DIR=$(dirname $(readlink -f "$0"))

echo "cuBB_SDK=$cuBB_SDK"

if [ "${BUILD}" != "" ]; then
    export BUILD="${BUILD}"
else
    export BUILD="build.$(arch)"
fi

if [ "$test_group" != "F08" ]; then
    echo "Error: test_group must be F08"
    usage
    exit 1
fi

task_bitmask="0xF"
if [ "$CUMAC_TASK" != "" ]; then
    task_bitmask="$CUMAC_TASK"
fi

# Parse cell_num from test_case like "8C_60"
cell_num=$(echo "$test_case" | sed 's/[^0-9].*//')
if [ "$cell_num" = "" ]; then
    echo "Error: failed to parse cell_num from input parameter"
    exit 1
fi

if [ "$LOG_PREFIX" = "" ]; then
    log_prefix=$(date -u "+%Y%m%d_%H%M%S")
else
    log_prefix=${LOG_PREFIX}
fi

log_folder=${log_prefix}_${test_group}_${test_case}_TYPE${alloc_type}_CUMSA

log_folder=${log_folder}_${duration}s

echo "Test case: $duration seconds ${test_group} ${test_case}"

if [ `whoami` = "root" ];then
    USE_SUDO=""
else
    USE_SUDO="sudo -E"
fi

if [ "$LOG_PATH" = "" ]; then
    export LOG_PATH=$cuBB_SDK/logs/latest
fi
echo "LOG_PATH=$LOG_PATH"

# Make sure the parent directory of LOG_PATH is writable
log_parent=$(dirname -- "$LOG_PATH")
${USE_SUDO} chmod 777 "$log_parent" || true
echo "ls -la $log_parent"
ls -la "$log_parent"

# Create LOG_PATH if it doesn't exist
if [ ! -d "$LOG_PATH" ]; then
    if ! mkdir -p "$LOG_PATH"; then
        echo "Error: cannot create LOG_PATH: $LOG_PATH"
        exit 1
    fi
fi

# chmod to fix NFS storage permission issue
chmod 777 "$LOG_PATH"
if [ ! -w "$LOG_PATH" ]; then
    echo "Error: LOG_PATH is not writable: $LOG_PATH"
    exit 1
fi

# Clean old logs if exist
rm -rf $LOG_PATH/*

cd "$LOG_PATH" || { echo "Error: cannot cd to LOG_PATH: $LOG_PATH"; exit 1; }

# Show the LOG_PATH directory contents
echo "ls -la $LOG_PATH"
ls -la "$LOG_PATH"

function kill_all {
    signal=$1
    if [ "$signal" = "" ]; then
        signal=SIGKILL
    fi

    echo "$(date +%T) start killing ... signal=$signal"
    start_time=$(date +%s)
    ${USE_SUDO} killall -9 -q cumac_cp mac_main phy_main l2_adapter_cuphycontroller_scf
    interval=$(($(date +%s) - $start_time))
    echo "$(date +%T) killed all in ${interval} seconds"
}

# Kill previous running processes if exist
kill_all SIGKILL

function sed_set_value {
    name="$1"
    value="$2"
    file="$3"
    # sed_cmd="sed -i 's/${name}[ ]*:.*/${name}: ${value}/g' ${file}"
    sed_cmd="sed -i 's/${name}[ ]*:.*/${name}: ${value}/g' ${file}"
    echo "$sed_cmd"
    eval "$sed_cmd"
}

# Check wehther TV exist
$LOCAL_DIR/cumac_cp_tv.sh -c ${cell_num} -a ${alloc_type}
return_code=$?
if [ ${return_code} -ne 0 ]; then
    echo "Generate TV failed"
    exit 1
fi

export mac_cfg_yaml="cuPHY-CP/testMAC/testMAC/test_mac_config.yaml"
export test_cumac_yaml="cuPHY-CP/testMAC/testMAC/test_cumac_config.yaml"
export cumac_cp_yaml="cuMAC-CP/config/cumac_cp.yaml"
export nvlog_cfg_yaml="cuPHY/nvlog/config/nvlog_config.yaml"

# test_mac_config.yaml
sed_set_value "test_cumac_config_file" "test_cumac_config.yaml" "${cuBB_SDK}/${mac_cfg_yaml}"

# test_cumac_config.yaml
sed_set_value "cumac_cp_standalone" "1" "${cuBB_SDK}/${test_cumac_yaml}"
sed_set_value "task_bitmask" "${task_bitmask}" "${cuBB_SDK}/${test_cumac_yaml}"
sed_set_value "cumac_cell_num" "${cell_num}" "${cuBB_SDK}/${test_cumac_yaml}"

# test_cumac_config.yaml worker_cores
if [ "$cell_num" -gt 8 ]; then
    # For 9 ~ 16 cells, requires 1 core per cell
    test_cumac_worker_cores="[26, 27, 28, 29, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]"
else
    test_cumac_worker_cores="[26, 27, 28, 29]"
fi
sed_set_value "worker_cores" "${test_cumac_worker_cores}" "${cuBB_SDK}/${test_cumac_yaml}"

# cumac_cp.yaml
sed_set_value "cell_num" "${cell_num}" "${cuBB_SDK}/${cumac_cp_yaml}"

# cumac_cp.yaml debug_option
if [ "$CUMCP_DEBUG" != "" ]; then
    sed_set_value "debug_option" "${CUMCP_DEBUG}" "${cuBB_SDK}/${cumac_cp_yaml}"
fi

# test_cumac_config.yaml debug_option
if [ "$CUMAC_DEBUG" != "" ]; then
    sed_set_value "debug_option" "${CUMAC_DEBUG}" "${cuBB_SDK}/${test_cumac_yaml}"
fi

LOCAL_CUMAC_CP_CMD="$cuBB_SDK/$BUILD/cuMAC-CP/cumac_cp"
LOCAL_MAC_CMD="$cuBB_SDK/$BUILD/cuPHY-CP/testMAC/testMAC/test_mac ${test_group} ${test_case}"

if [ "$TIMEOUT_BASE" = "" ]; then
    TIMEOUT_BASE=300
fi
N_SEC=$(($TIMEOUT_BASE + $duration))
LOCAL_CUMAC_CP_CMD="cd $LOG_PATH && $USE_SUDO timeout -s 9 $N_SEC $LOCAL_CUMAC_CP_CMD"

LOCAL_MAC_CMD="export cuBB_SDK=${cuBB_SDK} && cd $LOG_PATH && $USE_SUDO ${LOCAL_MAC_CMD}"
LOCAL_CMD_PHY_LOG="cp /tmp/cumac_cp.log /tmp/testmac.log ${cuBB_SDK}/${mac_cfg_yaml} ${cuBB_SDK}/${test_cumac_yaml} ${cuBB_SDK}/${cumac_cp_yaml} ${cuBB_SDK}/${nvlog_cfg_yaml} $LOG_PATH"

# Disable test_mac FAPI validation for L2SA test
LOCAL_MAC_CMD="${LOCAL_MAC_CMD}"

is_screen_running() {
    title=$1
    check=$(screen -ls | grep "${title}")
    if [ "$check" != "" ]; then
        echo 1
    else
        echo 0
    fi
}

print_cores () {
  pnames=$1
  for pname in $pnames
  do
    pid=$(pidof ${pname})
    if [ "${pid}" != "" ]; then
      echo "===== NAME: ${pname} PID: ${pid} ====="
      ps H -o 'pid tid comm policy psr priority %cpu %mem vsz rss' ${pid}
      echo ""
    fi
  done
}

thrput_started=0
poll_thrput_start() {
    poll_ret=0
    if [ $thrput_started -eq 0 ]; then
        result=$(grep -E "Cell +0 \|" $LOG_PATH/screenlog_mac.log)
        poll_ret=$?
        if [ $poll_ret -eq 0 ]; then
            thrput_started=1
            echo "MAC throughput started, run ps to get core usage ..."
            print_cores "cumac_cp test_mac cuphycontroller_scf l2_adapter_cuphycontroller_scf" > $LOG_PATH/core.log 2>&1
        fi
    fi
    return $poll_ret
}

moniter_all_running() {
    titles=$1
    pids=$2
    poll_cmd=$3
    timeout=$4
    ret=0

    if [ "${timeout}" = "" ]; then
        timeout=100000000
    fi

    echo "monitor: watching screens [$titles] and process pids [$pids] in $timeout seconds"

    let counter=0
    while :
    do
        sleep 1

        for title in $titles; do
            # screen -ls | grep "\.${title}"
            if [ $(is_screen_running $title) -eq 0 ]; then
                echo "monitor: $title had exited"
                ret=1
                break
            fi
        done

        for pid in $pids; do
            running=$(ps -o pid= -p $pid)
            if [ "$running" = "" ]; then
                # echo "monitor: process pid=$pid had exited"
                ret=2
                break
            fi
        done

        if [ "$poll_cmd" != "" ]; then
            eval "$poll_cmd"
        fi

        if [ $ret -ne 0 ]; then
            break
        fi

        let counter=counter+1
        if [ $counter -gt $timeout ]; then
            echo "monitor finished by timeout"
            return 0
         fi

    done
    return $ret
}

run_in_screen() {
    title=$1
    cmd=$2
    debug=$3
    if [ "$debug" != "" ]; then
        echo "screen[$title]: $cmd"
    fi
    screen -L -t $title -dmS $title bash -c "$cmd";
}

# Show the LOG_PATH directory contents
echo "ls -la $LOG_PATH"
ls -la "$LOG_PATH"
echo "PWD=$(pwd)"

run_in_screen cum "$LOCAL_CUMAC_CP_CMD" 1
sleep 2
run_in_screen mac "$LOCAL_MAC_CMD" 1

# Check running screens
screen -ls

if [ "$CHECK_RESULT_SCRIPT_PATH" = "" ]; then
    CHECK_RESULT_SCRIPT_PATH=$LOCAL_DIR
fi

$CHECK_RESULT_SCRIPT_PATH/check_result_cumcp.py ${duration} & checker_pid=$!
echo "check_result_cumcp.py running in background: pid=$checker_pid"

# Handler Ctrl + C to kill the background check_result_cumcp.py
clean_up() {
    echo "Single received, kill $checker_pid and exit"
    kill -9 $checker_pid
    exit 1
}
trap clean_up SIGINT

moniter_all_running "cum mac" "$checker_pid" "poll_thrput_start" ${N_SEC}
ret=$?
if [ $ret -eq 1 ]; then
    echo "monitor: unexpected exit - kill check_result_cumcp.py pid=$checker_pid"
    sleep 3
    kill $checker_pid
    echo "Test FAILED"
fi

wait $checker_pid
test_result=$?

# Kill after test
kill_all SIGINT

bash -c "$LOCAL_CMD_PHY_LOG" > $LOG_PATH/nvlog_collect.log 2>&1

# $USE_SUDO chown $(whoami):$(whoami) ./*

script_end=$(date +%s)
echo "Total test time: $((script_end - script_start))s"

if [ "${QA_TEST}" != "" ]; then
    exit $test_result
fi

echo "copy logs ..."
cd ${LOG_PATH}/..
cp -r ${LOG_PATH} ${log_folder}
error=$(grep " ERR \[" ${LOG_PATH}/*.log)
if [ $test_result -ne 0 ]; then
    mv ${log_folder} ${log_folder}_FAIL
elif [[ -n $error ]]; then
    mv ${log_folder} ${log_folder}_ERR
    test_result=1
else
    mv ${log_folder} ${log_folder}_PASS
fi

exit $test_result
