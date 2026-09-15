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
#This script is to be run on DU side for cuMAC-CP testing
#--------------------------------------------------------------------

# Identify SCRIPT_DIR
script_name=${0##*/}
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}
CONFIG_DIR=${CONFIG_DIR:-$cuBB_SDK}
CELL_NUM=""
ALLOC_TYPE=""
GPU_SHARE=""
TEST_ARCH=""
TEST_MODULE=""
GDB_SCRIPT=""

if [ "$(whoami)" != "root" ]; then
    USE_SUDO="sudo -E"
else
    USE_SUDO=""
fi

TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
[ -f $TEST_CONFIG_FILE ] && source $TEST_CONFIG_FILE

if [ "$NUM_CELLS" != "" ]; then
    CELL_NUM=$NUM_CELLS
fi

function show_usage() {
    echo "Usage: $script_name [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message and exit"
    echo "  -b, --build_dir      Build directory (default: build.$(uname -m))"
    echo "  -c, --cell_num   Number of cells (default: 8)"
    echo "  -a, --alloc_type Allocation type (default: 1)"
    echo "  -g, --gpu_share  GPU share count (default: 0, disables enable_gpu_share)"
    echo "  -t, --test_arch  Test architecture (default: GPU)"
    echo "  -m, --test_module  Test module name or bitmask (default: all)"
    echo "                     Names: all, multiCellUeSelection"
    echo "                     Bitmask: e.g. 0xF (all), 0x1 (multiCellUeSelection)"
    echo "  --gdb_script <script>  Wrapper script (e.g. gdb_bt.sh) prepended to the cumac_cp launch."
    echo "                         Default: empty (no wrapper)."
    echo ""
    echo "Examples:"
    echo "  $script_name -b build.perf.aarch64 -c 8 -a 1 -g 2 -t GPU -m all"
    echo "  $script_name -c 8 -a 1 -t GPU -m multiCellUeSelection"
    echo "  $script_name -c 8 -a 1 -t GPU -m 0xF"
    echo "  $script_name -c 8 -a 0 -t CPU -m 0x1"
    echo "  $script_name -c 8 --gdb_script \$cuBB_SDK/cicd-scripts/gdb_bt.sh"
    echo "  $script_name -h"
}

# Parse command line options (short and GNU long; getopts does not support --long)
while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--build_dir)
      [[ -n ${2+x} ]] || { echo "Option $1 requires an argument" >&2; show_usage; exit 1; }
      BUILD_DIR="$2"
      shift 2
      ;;
    -c|--cell_num)
      [[ -n ${2+x} ]] || { echo "Option $1 requires an argument" >&2; show_usage; exit 1; }
      CELL_NUM="$2"
      shift 2
      ;;
    -a|--alloc_type)
      [[ -n ${2+x} ]] || { echo "Option $1 requires an argument" >&2; show_usage; exit 1; }
      ALLOC_TYPE="$2"
      shift 2
      ;;
    -g|--gpu_share)
      [[ -n ${2+x} ]] || { echo "Option $1 requires an argument" >&2; show_usage; exit 1; }
      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: --gpu_share must be a non-negative integer" >&2; exit 1
      fi
      GPU_SHARE="$2"
      shift 2
      ;;
    -t|--test_arch)
      [[ -n ${2+x} ]] || { echo "Option $1 requires an argument" >&2; show_usage; exit 1; }
      TEST_ARCH="$2"
      shift 2
      ;;
    -m|--test_module)
      [[ -n ${2+x} ]] || { echo "Option $1 requires an argument" >&2; show_usage; exit 1; }
      TEST_MODULE="$2"
      shift 2
      ;;
    --gdb_script=*)
      GDB_SCRIPT="${1#*=}"
      shift
      ;;
    --gdb_script)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: Missing value for --gdb_script option" >&2
        show_usage
        exit 1
      fi
      GDB_SCRIPT="$2"
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      show_usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="build.$(uname -m)"
fi
CELL_NUM=${CELL_NUM:-8}
ALLOC_TYPE=${ALLOC_TYPE:-1}
GPU_SHARE=${GPU_SHARE:-0}
TEST_ARCH=${TEST_ARCH:-GPU}
TEST_MODULE=${TEST_MODULE:-all}
CUMAC_TASK=${CUMAC_TASK:-0xFFFF}
echo "BUILD_DIR: $BUILD_DIR, CELL_NUM: $CELL_NUM, ALLOC_TYPE: $ALLOC_TYPE, GPU_SHARE: $GPU_SHARE, TEST_ARCH: $TEST_ARCH, TEST_MODULE: $TEST_MODULE, GDB_SCRIPT: '$GDB_SCRIPT'"

# Validate required options
if [[ ! -d $cuBB_SDK/"$BUILD_DIR" ]]; then
    echo "Error: Unable to access $cuBB_SDK/$BUILD_DIR"
    exit 1
fi


function run_cumac_cp(){
   # $GDB_SCRIPT is unquoted on purpose: when --gdb_script is not passed it
   # is empty and the cumac_cp binary runs directly; when it is set (e.g.
   # to $cuBB_SDK/cicd-scripts/gdb_bt.sh), the wrapper script + binary are
   # invoked together, matching the pattern used by run2_cuPHYcontroller.sh.
   export CUDA_DEVICE_MAX_CONNECTIONS=8 && export CUDA_MPS_PIPE_DIRECTORY=/var && export CUDA_MPS_LOG_DIRECTORY=/var && ${USE_SUDO} $GDB_SCRIPT $cuBB_SDK/${BUILD_DIR}/cuMAC-CP/cumac_cp
}

function generate_tv(){
   export BUILD="${BUILD_DIR}"
   ${cuBB_SDK}/cuPHY-CP/testMAC/scripts/cumac_cp_tv.sh -c "${CELL_NUM}" -a "${ALLOC_TYPE}" --gpu_share "${GPU_SHARE}" -t "${CUMAC_TASK}"
}

function cumac_cp_config(){

	  test_mac_yaml="${cuBB_SDK}/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml"
	  test_cumac_yaml="${cuBB_SDK}/cuPHY-CP/testMAC/testMAC/test_cumac_config.yaml"
	  cumac_cp_yaml="${cuBB_SDK}/cuMAC-CP/config/cumac_cp.yaml"

    echo "Enable cuMAC feature in testmac instance ..."
    sed -i "s/test_cumac_config_file:.*/test_cumac_config_file: test_cumac_config.yaml/g" ${test_mac_yaml}
    sed -i "s/^cell_num:.*/cell_num: ${CELL_NUM}/g" ${cumac_cp_yaml}
    sed -i "s/^cumac_cell_num:.*/cumac_cell_num: ${CELL_NUM}/g" ${test_cumac_yaml}

	  echo "Update the core usage for cuMAC instance"
    sed -i "s/worker_cores:.*/worker_cores: [26, 27, 28, 22]/g" ${test_cumac_yaml}
    local _gpu_share_flag=0
    [ "${GPU_SHARE:-0}" -gt 0 ] && _gpu_share_flag=1
    sed -i "s/enable_gpu_share:.*/enable_gpu_share: ${_gpu_share_flag}/g" ${cumac_cp_yaml}

    if [ "${TEST_ARCH}" == "CPU" ];then
        sed -i "s/run_in_cpu:.*/run_in_cpu: 1/g" ${cumac_cp_yaml}
    elif [ "${TEST_ARCH}" == "GPU" ];then
        sed -i "s/run_in_cpu:.*/run_in_cpu: 0/g" ${cumac_cp_yaml}
    else
        echo "Error: Unknown TEST_ARCH '${TEST_ARCH}'. Must be CPU or GPU." >&2
        exit 1
    fi
    # Accept either a hex bitmask (e.g. 0xF) or a module name
    if [[ "${TEST_MODULE}" =~ ^0[xX][0-9a-fA-F]+$ ]]; then
        CUMAC_TASK="${TEST_MODULE}"
    elif [ "${TEST_MODULE}" == "multiCellUeSelection" ]; then
        CUMAC_TASK="0x1"
    elif [ "${TEST_MODULE}" == "all" ]; then
        CUMAC_TASK="0xf"
    else
        echo "Error: Unknown test module '${TEST_MODULE}'. Use a bitmask (e.g. 0xF) or a name (all, multiCellUeSelection)." >&2
        exit 1
    fi
    sed -i "s/task_bitmask:.*/task_bitmask: ${CUMAC_TASK}/g" ${test_cumac_yaml}
    sed -i "s/task_bitmask:.*/task_bitmask: ${CUMAC_TASK}/g" ${cumac_cp_yaml}
}

cumac_cp_config
generate_tv
run_cumac_cp
