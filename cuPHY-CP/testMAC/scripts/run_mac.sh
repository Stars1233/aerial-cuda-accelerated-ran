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

script_name=run_mac.sh
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

BUILD=build
if [[ -d $cuBB_SDK/build-$(arch) ]]; then
    BUILD=build-$(arch)
fi


cleanup() {
    start_time=$(date +%s)
    echo "$(date +%T:%N) $0 start cleanup ..."

    # For numactl debug on 23-2 manifest
    ps H -o 'pid tid comm policy psr priority %cpu %mem vsz rss' $(pidof test_mac) > mac.core.log
    ps H -o 'pid tid comm policy psr priority %cpu %mem vsz rss' $(pidof cuphycontroller_scf) > phy.core.log

    echo "*********** DEBUG FOR numactl debug on 23-2 manifest"
    echo "------mac.core.log"
    cat mac.core.log
    echo "------mac.core.log"
    echo ""
    echo ""
    echo "------phy.core.log"
    cat phy.core.log
    echo "------phy.core.log"
    echo "*********** DEBUG FOR numactl debug on 23-2 manifest"

    # The signal_handler function in: cuPHY-CP/testMAC/testMAC/test_mac.cpp
    # require us to send a signal twice for testmac to stop
    sudo pkill -e -SIGKILL test_mac
    sudo pkill -e -SIGKILL test_mac

    PYTHONPATH=$cuBB_SDK/$BUILD/cuPHY-CP/cuphyoam/ python3 $cuBB_SDK/cuPHY-CP/cuphyoam/examples/aerial_terminate_cuphycontroller.py

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
    export LOG_PATH=$cuBB_SDK/logs/mac
fi
echo "LOG_PATH=$LOG_PATH"
if [ ! -d $LOG_PATH ]; then
    mkdir -p $LOG_PATH
fi


cd $LOG_PATH

# Temporary fix for 23-2 isolated CPU core problem - taskset to core 19, which is dedicated to low priority test-mac threads
# Otherwise, there is a chance the startup threads use a core in the isolate set, which could be heavily utilized by high
# priority threads and then starve test_mac, so it hangs
LOCAL_MAC_CMD="stdbuf -o0 -e0 taskset -c 19 $cuBB_SDK/$BUILD/cuPHY-CP/testMAC/testMAC/test_mac ${test_group} ${test_case}${channel_option}"

# R750 needs to run on numa node 1. This is always true if the R750 is configured and installed the same
# way with the same PCIE port for the A100X. Use numactl to do this and check output of lscpu command.
#
# Example
#   ~$ lscpu | grep "NUMA node(s)"
#   NUMA node(s):        2
NUMA_NODES=$(lscpu | grep 'NUMA node(s)' | awk '{print $3}')
if [ "$NUMA_NODES" == "2" ]; then
    echo "NUMA node count is equal to 2 - set mac to run on node 1 using numactl"
    LOCAL_MAC_CMD="numactl -N 1 -m 1 ${LOCAL_MAC_CMD}"
fi

if [ -f $cuBB_SDK/cicd-scripts/gdb_bt.sh ]; then
    # Include a taskset -c 19 for the gdb_bt.sh call as well. See comment about the taskset on core 19 above for previous LOCAL_MAC_CMD
    LOCAL_MAC_CMD="taskset -c 19 $cuBB_SDK/cicd-scripts/gdb_bt.sh ${LOCAL_MAC_CMD}"
fi

LOCAL_MAC_CMD="$USE_SUDO ${LOCAL_MAC_CMD}"

# Clean old logs
rm -f $LOG_PATH/screenlog_mac.log
rm -f $LOG_PATH/check_result.log  

( $LOCAL_MAC_CMD 2>&1 | tee $LOG_PATH/screenlog_mac.log > /dev/null )&

$cuBB_SDK/cuPHY-CP/testMAC/scripts/check_result.py ${duration} "$LOG_PATH/screenlog_mac.log" "no_ru.log"
ret=$?

exit $ret
