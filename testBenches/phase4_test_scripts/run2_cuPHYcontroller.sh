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
#This script is to be run on DU side for cuPHY controller
#--------------------------------------------------------------------

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}

CONFIG_DIR=$cuBB_SDK

NSYS_ENABLED=false
MEMTRACE_ENABLED=false
CUDA_TRACER_ENABLED=false
CUDA_TRACER_OUTPUT="/tmp/cuda_api_tracer.log"
BUILD_DIR=build.$(uname -m)
ADDITIONAL_OPTIONS=""

show_usage() {
  echo "Usage: $0 [options]"
  echo
  echo "run the cuPHY controller on the DU side, with options to enable nsys profiling or memory tracing."
  echo
  echo "Options:"
  echo "  --build_dir <build-directory>    Specify the build directory to use."
  echo "  --config_dir <path>              Specify the path to the directory containing config files. (default: "\$cuBB_SDK")"
  echo "                                   the testBenches scripts will modify configuration files and write output files to this location"
  echo "  --memtrace                       Enable dynamic memory tracing."
  echo "                                   Note: If --nsys|-n is used, --memtrace will be ignored."
  echo "  -n, --nsys <additional options>  Enable nsys profiling with optional additional options."
  echo "  --nsys_exec <path to nsys>       Specify the path to a custom Nsight Systems executable."
  echo "  --nsys_trace <nsys -t options>   Specify Nsight Systems' -t options, e.g., cuda,osrt."
  echo "  --gdb_script <script>            Specify the gdb script to use."
  echo "  --timeout <seconds>              Kill cuphy after seconds"
  echo "  --cuda_tracer [output_path]      Enable CUDA API call counting (LD_PRELOAD). Optional path for log (default: /tmp/cuda_api_tracer.log)."
  echo "  -h, --help                       Show this help message."
  echo
  echo "Examples:"
  echo "  $0 --memtrace --build_dir build_rel"
  echo "    Runs the cuPHY controller in $cuBB_SDK/build_rel path with AERIAL_MEMTRACE=1 to enable memory tracing."
  echo
  echo "  $0 --nsys -y 60 -o /tmp/report_name"
  echo "    Runs the cuPHY controller with nsys profiling enabled, and specifies additional options."
}

NSYS_EXEC=nsys # will be overwritten if the user specifies the --nsys_exec <path to nsys> option
NSYS_TRACE_TYPE="cuda" # will be overwritten if the user specifies the --nsys_trace <nsys -t options> option
GDB_SCRIPT=""
TIMEOUT=0

#parse input
while [[ $# -gt 0 ]]; do
    case $1 in
        --build_dir=*)
          BUILD_DIR="${1#*=}"
          shift
          ;;
        --build_dir)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for --build_dir option"
            show_usage
            exit 1
          fi
          BUILD_DIR="$2"
          shift 2
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
        -n|--nsys)
            NSYS_ENABLED=true
            shift
            ;;
        --nsys_exec)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for --nsys_exec option"
            show_usage
            exit 1
          fi
          NSYS_EXEC="$2"
          echo "Will use NSYS_EXEC=$NSYS_EXEC"
          shift 2
          ;;
        --nsys_trace)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for --nsys_trace option"
            show_usage
            exit 1
          fi
          NSYS_TRACE_TYPE="$2"
          echo "Will use NSYS_TRACE_TYPE=$NSYS_TRACE_TYPE"
          shift 2
          ;;
        --memtrace)
            MEMTRACE_ENABLED=true
            shift
            ;;
        --gdb_script=*)
          GDB_SCRIPT="${1#*=}"
          shift
          ;;
        --gdb_script)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for --gdb_script option"
            show_usage
            exit 1
          fi
          GDB_SCRIPT="$2"
          shift 2
          ;;
        --timeout=*)
          TIMEOUT="${1#*=}"
          shift
          ;;
        --timeout)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for --timeout option"
            show_usage
            exit 1
          fi
          TIMEOUT="$2"
          shift 2
          ;;
        --cuda_tracer)
          CUDA_TRACER_ENABLED=true
          if [[ -n "$2" && "$2" != -* ]]; then
            CUDA_TRACER_OUTPUT="$2"
            shift 2
          else
            shift
          fi
          ;;
        --cuda_tracer=*)
          CUDA_TRACER_ENABLED=true
          CUDA_TRACER_OUTPUT="${1#*=}"
          shift
          ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            # Collect any other options to pass to nsys
            ADDITIONAL_OPTIONS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -d $cuBB_SDK/"$BUILD_DIR" ]]; then
  echo "Error: Unable to access $cuBB_SDK/"$BUILD_DIR""
  exit 1
fi


if [ "$NSYS_ENABLED" = true ] && [ "$MEMTRACE_ENABLED" = true ]; then
  echo "Warning: --memtrace option is ignored"
fi

# Filter out elements that contain only spaces
for i in "${!ADDITIONAL_OPTIONS[@]}"; do
  if [[ "${ADDITIONAL_OPTIONS[i]// }" == "" ]]; then
    unset 'ADDITIONAL_OPTIONS[i]'
  fi
done

TEST_CONFIG_FILE=$CONFIG_DIR/testBenches/phase4_test_scripts/test_config_summary.sh
if [[ ! -f $TEST_CONFIG_FILE ]]; then
    echo "$TEST_CONFIG_FILE is missing. Please run setup1_DU.sh and setup2_RU.sh first"
    exit 1
fi
source $TEST_CONFIG_FILE
if [[ ! -v TEST_CONFIG_DONE ]]; then
    echo "Error: Please run test_config.sh before executing the run scripts."
    exit 1
fi

#-------------------------------------------------------------------------------------------------------
#verify if setup1_DU.sh has been run before running cuPHY-controller
# Check first interface (always required)
ACTUAL_DU_MAC_ADDRESS_0=$(cat /sys/class/net/"${DU_ETH_INTERFACE_0}"/address)
if [ "$ACTUAL_DU_MAC_ADDRESS_0" != "$DU_MAC_ADDRESS_0" ]; then
    echo "Error: MAC addresses do not match for interface 0. Expected $ACTUAL_DU_MAC_ADDRESS_0, but reading $DU_MAC_ADDRESS_0 from logs. Please ensure to run setup1_DU.sh and setup2_RU.sh before running run2_cuPHYcontroller.sh"
    exit 1
fi

# Check second interface if running in 2-port mode
if [ "${NUM_PORTS:-1}" -eq 2 ]; then
    ACTUAL_DU_MAC_ADDRESS_1=$(cat /sys/class/net/"${DU_ETH_INTERFACE_1}"/address)
    if [ "$ACTUAL_DU_MAC_ADDRESS_1" != "$DU_MAC_ADDRESS_1" ]; then
        echo "Error: MAC addresses do not match for interface 1. Expected $ACTUAL_DU_MAC_ADDRESS_1, but reading $DU_MAC_ADDRESS_1 from logs. Please ensure to run setup1_DU.sh and setup2_RU.sh before running run2_cuPHYcontroller.sh"
        exit 1
    fi
fi

#----------------------------------------------------------------------------------
if [ "$NSYS_ENABLED" = true ]; then
  # number of time slot from test_config_summary.sh
  #issue a warning for large number of time slots when nsys profiling enabled
  if (( TEST_SLOTS > 3000 )); then
    echo "Warning: using a large number of time slots, '$TEST_SLOTS', for nsys profiling"
  fi
fi
#----------------------------------------------------------------------------------
stop_mps() {
    # Stop existing MPS
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
    sudo -E echo quit | sudo -E nvidia-cuda-mps-control || true
}

restart_mps() {
    # Stop existing MPS
    stop_mps

    # Start MPS
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
    sudo -E nvidia-cuda-mps-control -d
    sudo -E echo start_server -uid 0 | sudo -E nvidia-cuda-mps-control
}
#----------------------------------------------------------------------------------
GREEN_CONTEXTS_MODE=$(grep "^USE_GREEN_CONTEXT" "$TEST_CONFIG_FILE" | sed 's/.*="//;s/";*$//')
if [ "$GREEN_CONTEXTS_MODE" = "1" ]; then
    echo "stopping MPS for green contexts"
    stop_mps
else
    echo "restarting MPS"
    restart_mps
fi

if [ $TIMEOUT -gt 0 ]; then
    WITH_TIMEOUT="timeout --kill-after=10 ${TIMEOUT}"
else
    WITH_TIMEOUT=""
fi

# CUDA API tracer: use env to pass LD_PRELOAD (sudo often strips it even with -E)
if [ "$CUDA_TRACER_ENABLED" = true ]; then
    # Unset inherited tracer env so nvidia-smi, nvcc, etc. do not load the tracer and create stray log files
    unset LD_PRELOAD CUDA_API_TRACER_OUTPUT 2>/dev/null || true
    CUDA_TRACER_SO="${cuBB_SDK}/testBenches/phase4_test_scripts/syscall_tracer/libcuda_api_tracer.so"
    if [[ ! -f "$CUDA_TRACER_SO" ]]; then
        echo "Error: CUDA tracer library not found at $CUDA_TRACER_SO. Run ./build_cuda_api_tracer.sh first."
        exit 1
    fi
    CUDA_TRACER_PREFIX=(env "LD_PRELOAD=$CUDA_TRACER_SO" "CUDA_API_TRACER_OUTPUT=$CUDA_TRACER_OUTPUT")
    echo "CUDA API tracer enabled: LD_PRELOAD=$CUDA_TRACER_SO, output=$CUDA_TRACER_OUTPUT"
else
    CUDA_TRACER_PREFIX=()
fi

if [ "$NSYS_ENABLED" = true ] && [ "$CUDA_TRACER_ENABLED" = true ]; then
    echo "Warning: CUDA API tracer is ignored when nsys profiling is enabled (LD_PRELOAD would apply to nsys, not the target)."
    CUDA_TRACER_PREFIX=()
fi

# Log some system information, such as nvidia-smi output.
echo "nvidia-smi | grep \"Driver Version\""
nvidia-smi | grep "Driver Version" || echo "Warning: Could not retrieve driver version from nvidia-smi"
echo "cat /proc/driver/nvidia/version"
cat /proc/driver/nvidia/version || echo "Warning: Could not read /proc/driver/nvidia/version"
echo "nvcc --version"
nvcc --version || echo "Warning: Could not retrieve nvcc version"


if [ "$NSYS_ENABLED" = true ]; then
    echo "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 $NSYS_EXEC profile -t $NSYS_TRACE_TYPE -s none --cpuctxsw=none --run-as=root "${ADDITIONAL_OPTIONS[@]}" "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" $CONTROLLER_MODE
    sudo -E "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 $NSYS_EXEC profile -t $NSYS_TRACE_TYPE -s none --cpuctxsw=none --run-as=root "${ADDITIONAL_OPTIONS[@]}" "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" $CONTROLLER_MODE
    RET=$?
    # If you want to collect osrt events, you may want to increase he osrt threshold to 10us (e.g., via --osrt-threshold 10000)

else
    if [ "$MEMTRACE_ENABLED" = false ]; then
        echo "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 $WITH_TIMEOUT $GDB_SCRIPT $cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf $CONTROLLER_MODE
        sudo -E "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 $WITH_TIMEOUT $GDB_SCRIPT "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" $CONTROLLER_MODE
        RET=$?
    else
        echo "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 AERIAL_MEMTRACE=1 $WITH_TIMEOUT $GDB_SCRIPT $cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf $CONTROLLER_MODE
        sudo -E "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 AERIAL_MEMTRACE=1 $WITH_TIMEOUT $GDB_SCRIPT "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" $CONTROLLER_MODE
        RET=$?
    fi
fi
exit $RET
