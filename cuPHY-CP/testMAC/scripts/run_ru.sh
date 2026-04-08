#!/bin/bash

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

script_name=run_ru.sh
function usage {
    echo "Usage: $script_name <duration> <test_group> <test_case> --channels <channels>"
    echo "    duration:    The duration to checking status before exit. Unit: second"
    echo "    test_group:  F08, F13, nrSim"
    echo "    test_case:   For F08/F13:  1C, 2C, 3C, 4C"
    echo "                 For nrSim:    Case number, like 1001, 2001, ..."
    echo "Example: ./$script_name 20 nrSim 6001 --channels PUCCH"
}

if [ "$3" = "" ]; then
    usage
    exit 1
fi

if [ "$cuBB_SDK" = "" ]; then
    echo "Please set cuBB_SDK first"
    exit 1
fi

echo "cuBB_SDK=$cuBB_SDK"
echo "CUBB_HOME=$CUBB_HOME"

BUILD=build
if [[ -d $cuBB_SDK/build-$(arch) ]]; then
    BUILD=build-$(arch)
fi

cleanup() {
    start_time=$(date +%s)
    echo "$(date +%T:%N) $0 start cleanup ..."

    # Kill test_mac too - we use test_mac to get the thresholds and
    # that could have locked up and still be running.
    sudo pkill -e -SIGKILL test_mac

    echo "$(date +%T:%N) $0 kill all sub processes"
    pkill -e -P $$

    wait
    interval=$(($(date +%s) - $start_time))
    echo "$(date +%T:%N) %0 cleaned up all in ${interval} seconds"
}
trap cleanup EXIT

duration=$1
test_group=$2
test_case=$3

if [ "$test_group" == "nrSim" ]; then
    # This breaks for anything passed in with leading zeros, since printf considers
    # the value to not be base 10. Remove leading zeros first.
    # This only happens if someone is passing in something like 0101 instead of 101
    test_case=$(echo $test_case | sed 's/^0*//')

    # Then format the test_case and put back the leading zeros again
    printf -v test_case "%04d" $test_case
    if [[ "${test_case}" == "1"* ]]; then
        channels="PBCH";
    elif [[ "${test_case}" == "2"* ]]; then
        channels="PDCCH_DL";
    elif [[ "${test_case}" == "3"* ]]; then
        channels="PDSCH";
    elif [[ "${test_case}" == "4"* ]]; then
        channels="CSI_RS";
    elif [[ "${test_case}" == "5"* ]]; then
        channels="PRACH";
    elif [[ "${test_case}" == "6"* ]]; then
        channels="PUCCH";
    elif [[ "${test_case}" == "7"* ]]; then
        channels="PUSCH";
    fi
fi

if [ "$5" != "" ]; then
    channels=$5
fi

if [ ! -z "${channels}" ]; then
    channel_option=" --channels $channels"
else
    channel_option=""
fi

echo "Test case: $duration seconds ${test_group} ${test_case} $channels"

if [ `whoami` = "root" ];then
    USE_SUDO=""
else
    USE_SUDO="sudo -E LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi



if [ "$LOG_PATH" == "" ]; then
    export LOG_PATH=$cuBB_SDK/logs/ru
fi
echo "LOG_PATH=$LOG_PATH"
if [ ! -d $LOG_PATH ]; then
    mkdir -p $LOG_PATH
fi

cd $LOG_PATH


LOCAL_RU_CMD="stdbuf -o0 -e0 $cuBB_SDK/$BUILD/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator ${test_group} ${test_case}${channel_option}"
LOCAL_MAC_CMD="stdbuf -o0 -e0 $cuBB_SDK/$BUILD/cuPHY-CP/testMAC/testMAC/test_mac ${test_group} ${test_case}${channel_option}"

if [ -f $cuBB_SDK/cicd-scripts/gdb_bt.sh ]; then
    LOCAL_RU_CMD="$cuBB_SDK/cicd-scripts/gdb_bt.sh ${LOCAL_RU_CMD}"
    LOCAL_MAC_CMD="$cuBB_SDK/cicd-scripts/gdb_bt.sh ${LOCAL_MAC_CMD}"
fi
LOCAL_RU_CMD="$USE_SUDO ${LOCAL_RU_CMD}"
LOCAL_MAC_CMD="$USE_SUDO ${LOCAL_MAC_CMD}"


# Clean old logs
rm -f $LOG_PATH/screenlog_ru.log
rm -f $LOG_PATH/mac_init.log
rm -f $LOG_PATH/check_aggregate_result.log
rm -f $LOG_PATH/check_result.log


( $LOCAL_RU_CMD 2>&1 | tee $LOG_PATH/screenlog_ru.log > /dev/null )&
RU_PID=$!

ulimit -c unlimited
# Run "test_mac ... --thrput" to print the expected throughput data
bash -c "$LOCAL_MAC_CMD --thrput" | tee $LOG_PATH/mac_init.log 

# Check running screens
#screen -ls

$cuBB_SDK/cuPHY-CP/testMAC/scripts/check_result.py ${duration} "$LOG_PATH/mac_init.log" 
ret=$?


sudo pkill -e -SIGINT ru_emulator
echo "Waiting for $RU_PID"
wait $RU_PID

$cuBB_SDK/cuPHY-CP/testMAC/scripts/check_aggregate_result.py 
ret2=$?

if [[ "$ret" == 0 && "$ret2" == 0 ]]; then
    exit 0
else
    echo check_result.py exited with $ret
    echo check_aggregate_result.py exited with $ret2
    exit 1
fi
