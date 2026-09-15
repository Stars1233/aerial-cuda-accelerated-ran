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
    echo "Usage: $script_name <duration> <test_group> <test_case> [-a|--alloc_type=<value>] [--gpu_share=<n>]"
    echo "    duration:    The duration to checking status before exit. Unit: second"
    echo "    test_group:  F08"
    echo "    test_case:   1C_60, 2C_60, 3C_60, ..."
    echo "    -a, --alloc_type: cuMAC allocType type parameter (default: 1)"
    echo "    -r, --renew: Force regeneration even if test vectors already exist"
    echo "    -t, --task <hex>: cuMAC task bitmask, exported as CUMAC_TASK (default: 0xF)"
    echo "    --gpu_share <n>: GPU-share mode: 0=disabled (default), 1=enabled, 2=enabled with cuBB dumped H5 input"
    echo "    --gdb_script <path>: Wrapper script (e.g. gdb_bt.sh) prepended to cumac_cp/test_mac launches."
    echo "                         Pass an empty string to disable wrapping. Default: \$cuBB_SDK/cicd-scripts/gdb_bt.sh if present, else empty."
    echo ""
    echo "Examples:"
    echo "    $script_name 20 F08 8C_60 -a 1"
    echo "    $script_name 20 F08 8C_60 --alloc_type=1"
    echo "    $script_name 20 F08 8C_60 --alloc_type 0 --task 0x0F"
    echo "    $script_name 20 F08 2C_66c --gpu_share 1 --task 0x20 --renew"
    echo "    $script_name 20 F08 2C_66c --gpu_share 2 --task 0x21 --renew"
    echo "    $script_name 20 F08 2C_66c --gdb_script /path/to/my_gdb_wrapper.sh"
    echo "    $script_name 20 F08 2C_66c --gdb_script ''"
}

LOG_INFO() {
    LOG_TIME="$(date -u '+%T.%6N')"
    info="$1"
    echo "${LOG_TIME} ${info}"
}

LOG_CMD() {
    LOG_TIME="$(date -u '+%T.%6N')"
    cmd="$1"
    echo "${LOG_TIME} [${cmd}]"
    eval "$cmd"
}

# Initialize default values
alloc_type=1
gpu_share=0
force_renew=0
task=""

GDB_SCRIPT=$(which gdb_bt.sh)

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
        -r|--renew)
            force_renew=1
            shift
            ;;
        --task=*)
            task="${1#*=}"
            shift
            ;;
        -t|--task)
            if [[ -n "$2" && "$2" != -* ]]; then
                task="$2"
                shift 2
            else
                echo "Error: -t/--task requires a value"
                usage
                exit 1
            fi
            ;;
        --gpu_share=*)
            gpu_share="${1#*=}"
            shift
            ;;
        --gpu_share)
            if [[ -n "$2" && "$2" != -* ]]; then
                gpu_share="$2"
                shift 2
            else
                echo "Error: --gpu_share requires a value (0, 1, or 2)"
                usage
                exit 1
            fi
            ;;
        --gdb_script=*)
            GDB_SCRIPT="${1#*=}"
            shift
            ;;
        --gdb_script)
            GDB_SCRIPT="$2"
            shift 2
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

echo "Parameters: duration=$duration, test_group=$test_group, test_case=$test_case, alloc_type=$alloc_type, gpu_share=$gpu_share, force_renew=$force_renew, task=$task, gdb_script='$GDB_SCRIPT'"

script_start=$(date +%s)

if [ "$cuBB_SDK" = "" ]; then
    echo "Please set cuBB_SDK first"
    exit 1
fi

LOCAL_DIR=$(dirname $(readlink -f "$0"))

echo "cuBB_SDK=$cuBB_SDK"

if [ "$test_group" != "F08" ]; then
    echo "Error: test_group must be F08"
    usage
    exit 1
fi

# Verify GDB_SCRIPT path
if [ ! -f ${GDB_SCRIPT} ]; then
    LOG_INFO "GDB_SCRIPT ${GDB_SCRIPT} not found, set to empty"
    GDB_SCRIPT=""
fi

# Resolve task_bitmask: default 0xF, overridden by CUMAC_TASK env, with the
# -t/--task command-line value taking highest priority.
task_bitmask="0xF"
if [ -n "$CUMAC_TASK" ]; then
    task_bitmask="$CUMAC_TASK"
fi
if [ -n "$task" ]; then
    task_bitmask="$task"
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

log_folder=${log_prefix}_${test_group}_${test_case}_TYPE${alloc_type}_CUMSA_TASK_${task_bitmask}

# Append gpu_share value if MU UE Grouping is enabled
if (( (task_bitmask & 0x20) != 0 )); then
    log_folder=${log_folder}_GPU_SHARE${gpu_share}
fi

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

main_log=$LOG_PATH/main_cumcp_sa.log

# Show the LOG_PATH directory contents
LOG_CMD "ls -la $LOG_PATH" | tee -a $main_log
LOG_CMD "cd $cuBB_SDK && ls -lah build*" | tee -a $main_log

if [ "${BUILD}" = "" ]; then
    export BUILD="build.$(arch)"
fi

# If a build tree already exists, use it. Otherwise do a fresh build
if [ -d "${cuBB_SDK}/${BUILD}" ]; then
    export BUILD_DIR="${BUILD}"
    LOG_INFO "Using existing build tree: ${BUILD}" | tee -a $main_log
elif [ -d "${cuBB_SDK}/build.$(arch)" ]; then
    export BUILD="build.$(arch)"
    export BUILD_DIR="${BUILD}"
    LOG_INFO "Using existing build tree: ${BUILD}" | tee -a $main_log
elif [ -d "${cuBB_SDK}/build" ]; then
    export BUILD="build"
    export BUILD_DIR="${BUILD}"
    LOG_INFO "Using existing build tree: ${BUILD}" | tee -a $main_log
else
    export BUILD="build.$(arch)"
    export BUILD_DIR="${BUILD}"
    LOG_INFO "No build tree found, doing a fresh build at build.$(arch)" | tee -a $main_log
    LOG_INFO "======================================" | tee -a $main_log
    LOCAL_CMD="cd ${cuBB_SDK} && testBenches/phase4_test_scripts/build_aerial_sdk.sh > ${LOG_PATH}/build_all.log 2>&1"
    LOG_INFO "${LOCAL_CMD}" | tee -a $main_log
    build_start=$(date +%s)
    eval "${LOCAL_CMD}" | tee -a "$main_log"
    return_code=${PIPESTATUS[0]}
    build_end=$(date +%s)
    LOG_INFO "[${LOCAL_CMD}] time cost: $((build_end - build_start))s" | tee -a $main_log
    if [ ${return_code} -ne 0 ]; then
        LOG_INFO "[${LOCAL_CMD}] failed" | tee -a $main_log
        exit 1
    fi
    LOG_INFO "======================================" | tee -a $main_log
fi

cd "$LOG_PATH" || { echo "Error: cannot cd to LOG_PATH: $LOG_PATH"; exit 1; }

function kill_all {
    signal=$1
    if [ "$signal" = "" ]; then
        signal=SIGKILL
    fi

    echo "$(date +%T) start killing ... signal=$signal" | tee -a $main_log
    start_time=$(date +%s)
    ${USE_SUDO} killall -9 -q l2_adapter_cuphycontroller_scf phy_main
    ${USE_SUDO} killall -9 -q mac_main
    ${USE_SUDO} killall -9 -q cumac_cp
    interval=$(($(date +%s) - $start_time))
    echo "$(date +%T) killed all in ${interval} seconds" | tee -a $main_log
}

# Kill previous running processes if exist
kill_all SIGKILL

function sed_set_value {
    name="$1"
    value="$2"
    file="$3"
    # sed_cmd="sed -i 's/${name}[ ]*:.*/${name}: ${value}/g' ${file}"
    sed_cmd="sed -i 's/${name}[ ]*:.*/${name}: ${value}/g' ${file}"
    echo "$sed_cmd" | tee -a $main_log
    eval "$sed_cmd"
}

# Check wehther TV exist
tv_args=("-c" "${cell_num}" "-a" "${alloc_type}" "--gpu_share" "${gpu_share}")
tv_args+=("-t" "${task_bitmask}")
if [ "$force_renew" = "1" ]; then
    tv_args+=("-r")
fi
# Forward --gdb_script to cumac_cp_tv.sh only when wrapping is enabled
# here, so an unset / cleared GDB_SCRIPT keeps the TV-gen launches
# unwrapped (default behaviour) instead of forcing an empty wrapper.
if [ -n "${GDB_SCRIPT}" ]; then
    tv_args+=("--gdb_script" "${GDB_SCRIPT}")
fi
"$LOCAL_DIR/cumac_cp_tv.sh" "${tv_args[@]}" | tee -a "$main_log"
return_code=${PIPESTATUS[0]}
if [ ${return_code} -ne 0 ]; then
    echo "Generate TV failed" | tee -a $main_log
    exit 1
fi

export mac_cfg_yaml="cuPHY-CP/testMAC/testMAC/test_mac_config.yaml"
export test_cumac_yaml="cuPHY-CP/testMAC/testMAC/test_cumac_config.yaml"
export cumac_cp_yaml="cuMAC-CP/config/cumac_cp.yaml"
export nvlog_cfg_yaml="cuPHY/nvlog/config/nvlog_config.yaml"
export ue_grp_tv_yaml="cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/config.yaml"

# test_mac_config.yaml
sed_set_value "test_cumac_config_file" "test_cumac_config.yaml" "${cuBB_SDK}/${mac_cfg_yaml}"

# test_cumac_config.yaml
sed_set_value "cumac_cp_standalone" "1" "${cuBB_SDK}/${test_cumac_yaml}"
sed_set_value "task_bitmask" "${task_bitmask}" "${cuBB_SDK}/${test_cumac_yaml}"
sed_set_value "cumac_cell_num" "${cell_num}" "${cuBB_SDK}/${test_cumac_yaml}"
sed_set_value "srs_slot_lag" "0" "${cuBB_SDK}/${test_cumac_yaml}"
sed_set_value "NUM_CELL" "${cell_num}" "${cuBB_SDK}/${ue_grp_tv_yaml}"

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
sed_set_value "enable_tv_test" "1" "${cuBB_SDK}/${cumac_cp_yaml}"
sed_set_value "enable_cubb" "0" "${cuBB_SDK}/${cumac_cp_yaml}"
sed_set_value "srs_slot_lag" "0" "${cuBB_SDK}/${cumac_cp_yaml}"
sed_set_value "task_bitmask" "${task_bitmask}" "${cuBB_SDK}/${cumac_cp_yaml}"
if [ "${gpu_share}" != "0" ]; then
    sed_set_value "enable_gpu_share" "1" "${cuBB_SDK}/${cumac_cp_yaml}"
else
    sed_set_value "enable_gpu_share" "0" "${cuBB_SDK}/${cumac_cp_yaml}"
fi

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
LOCAL_CUMAC_CP_CMD="cd $LOG_PATH && $USE_SUDO timeout -s 9 $N_SEC $GDB_SCRIPT $LOCAL_CUMAC_CP_CMD"

LOCAL_MAC_CMD="export cuBB_SDK=${cuBB_SDK} && cd $LOG_PATH && $USE_SUDO $GDB_SCRIPT ${LOCAL_MAC_CMD}"

COPY_LOG_FILES="/tmp/cumac_cp.log /tmp/testmac.log ${cuBB_SDK}/${nvlog_cfg_yaml}"
COPY_LOG_FILES="${COPY_LOG_FILES} ${cuBB_SDK}/${mac_cfg_yaml} ${cuBB_SDK}/${test_cumac_yaml}"
COPY_LOG_FILES="${COPY_LOG_FILES} ${cuBB_SDK}/${cumac_cp_yaml} ${cuBB_SDK}/${ue_grp_tv_yaml}"
LOCAL_CMD_PHY_LOG="cp ${COPY_LOG_FILES} $LOG_PATH"

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
        echo "screen[$title]: $cmd" | tee -a $main_log
    fi
    screen -L -t $title -dmS $title bash -c "$cmd; ret=\$?; echo ret=\$ret";
}

# Show the LOG_PATH directory contents
LOG_CMD "ls -la $LOG_PATH" | tee -a $main_log
LOG_INFO "PWD=$(pwd)" | tee -a $main_log

run_in_screen cum "$LOCAL_CUMAC_CP_CMD" 1
sleep 2
run_in_screen mac "$LOCAL_MAC_CMD" 1

# Check running screens
LOG_CMD "screen -ls" | tee -a $main_log

if [ "$CHECK_RESULT_SCRIPT_PATH" = "" ]; then
    CHECK_RESULT_SCRIPT_PATH=$LOCAL_DIR
fi

"$CHECK_RESULT_SCRIPT_PATH/check_result_cumcp.py" ${duration} --cumcp-sa --mac-log "$LOG_PATH/screenlog_mac.log" --cumcp-log "$LOG_PATH/screenlog_cum.log" > >(tee -a "$main_log") 2>&1 &
checker_pid=$!
echo "check_result_cumcp.py running in background: pid=$checker_pid" | tee -a $main_log

# Handler Ctrl + C to kill the background check_result_cumcp.py
clean_up() {
    echo "Single received, kill $checker_pid and exit" | tee -a $main_log
    kill -9 $checker_pid
    exit 1
}
trap clean_up SIGINT

moniter_all_running "cum mac" "$checker_pid" "poll_thrput_start" ${N_SEC}
ret=$?
if [ $ret -eq 1 ]; then
    echo "monitor: unexpected exit - kill check_result_cumcp.py pid=$checker_pid" | tee -a $main_log
    sleep 3
    kill $checker_pid
    echo "Test FAILED" | tee -a $main_log
fi

wait $checker_pid
test_result=$?

# Kill after test
kill_all SIGINT

LOG_CMD "$LOCAL_CMD_PHY_LOG > $LOG_PATH/nvlog_collect.log 2>&1" | tee -a $main_log

# $USE_SUDO chown $(whoami):$(whoami) ./*

script_end=$(date +%s)
echo "Total test time: $((script_end - script_start))s check_result=$test_result" | tee -a $main_log

if [ "${QA_TEST}" != "" ]; then
    exit $test_result
fi

LOG_INFO "copy logs ..." | tee -a $main_log
LOG_CMD "cd ${LOG_PATH}/.."
LOG_CMD "cp -r ${LOG_PATH} ${log_folder}"
error=$(grep -E "[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6} ERR" ${LOG_PATH}/screenlog_*.log)
if [ $test_result -ne 0 ]; then
    LOG_CMD "mv ${log_folder} ${log_folder}_FAIL"
elif [[ -n $error ]]; then
    LOG_CMD "mv ${log_folder} ${log_folder}_ERR"
    test_result=1
else
    LOG_CMD "mv ${log_folder} ${log_folder}_PASS"
fi

exit $test_result
