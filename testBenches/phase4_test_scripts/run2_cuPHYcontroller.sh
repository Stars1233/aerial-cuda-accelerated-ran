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
PERF_TRACE_ENABLED=false
PERF_TRACE_OPTS=""
PERF_TRACE_READY_TIMEOUT=120
PERF_TRACE_DIR=""
BPFTRACE_ENABLED=false
BPFTRACE_OPTS=""
BPFTRACE_DIR=""
BPF_OUTPUT_FILE=""

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
  echo "  --perf_trace                     Launch perf_trace_workers.sh automatically once L1 is ready."
  echo "  --perf_trace_opts \"<opts>\"        Options forwarded to perf_trace_workers.sh (e.g. \"-c 60 -e syscalls:sys_enter_futex\")."
  echo "  --perf_trace_dir <dir>           Directory for perf output files (required with --perf_trace). Created if missing."
  echo "  --perf_trace_timeout <seconds>   Timeout waiting for L1 readiness before giving up (default: 120)."
  echo "  --bpftrace                       Launch bpftrace_trace_workers.sh (in-kernel all-syscall count) once L1 is ready."
  echo "  --bpftrace_opts \"<opts>\"          Options forwarded to bpftrace_trace_workers.sh (e.g. \"--raw\" or \"-d 5\")."
  echo "  --bpftrace_dir <dir>             Directory for bpftrace output files (required with --bpftrace). Created if missing."
  echo "                                   (--perf_trace_timeout also governs the bpftrace readiness wait.)"
  echo "  -h, --help                       Show this help message."
  echo
  echo "Examples:"
  echo "  $0 --memtrace --build_dir build_rel"
  echo "    Runs the cuPHY controller in $cuBB_SDK/build_rel path with AERIAL_MEMTRACE=1 to enable memory tracing."
  echo
  echo "  $0 --nsys -y 60 -o /tmp/report_name"
  echo "    Runs the cuPHY controller with nsys profiling enabled, and specifies additional options."
  echo
  echo "  $0 --perf_trace"
  echo "    Waits for L1 readiness, then auto-launches perf_trace_workers.sh on DL/UL worker threads."
  echo
  echo "  $0 --perf_trace --perf_trace_opts \"-c 60 -o /tmp/run1.data\" --perf_trace_timeout 180"
  echo "    Same as above, with custom perf_trace_workers.sh options and a 180s readiness timeout."
  echo
  echo "  $0 --perf_trace --perf_trace_dir /tmp/perf --bpftrace --bpftrace_dir /tmp/bpf"
  echo "    Runs perf (futex, CPU 58) and bpftrace (all-syscalls, CPU 59) together on orthogonal cores."
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
        --perf_trace)
          PERF_TRACE_ENABLED=true
          shift
          ;;
        --perf_trace_opts=*)
          PERF_TRACE_OPTS="${1#*=}"
          shift
          ;;
        --perf_trace_opts)
          if [[ -z "$2" ]]; then
            echo "Error: Missing value for --perf_trace_opts option"
            show_usage
            exit 1
          fi
          PERF_TRACE_OPTS="$2"
          shift 2
          ;;
        --perf_trace_timeout=*)
          PERF_TRACE_READY_TIMEOUT="${1#*=}"
          shift
          ;;
        --perf_trace_timeout)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for --perf_trace_timeout option"
            show_usage
            exit 1
          fi
          PERF_TRACE_READY_TIMEOUT="$2"
          shift 2
          ;;
        --perf_trace_dir=*)
          PERF_TRACE_DIR="${1#*=}"
          shift
          ;;
        --perf_trace_dir)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for --perf_trace_dir option"
            show_usage
            exit 1
          fi
          PERF_TRACE_DIR="$2"
          shift 2
          ;;
        --bpftrace)
          BPFTRACE_ENABLED=true
          shift
          ;;
        --bpftrace_opts=*)
          BPFTRACE_OPTS="${1#*=}"
          shift
          ;;
        --bpftrace_opts)
          if [[ -z "$2" ]]; then
            echo "Error: Missing value for --bpftrace_opts option"
            show_usage
            exit 1
          fi
          BPFTRACE_OPTS="$2"
          shift 2
          ;;
        --bpftrace_dir=*)
          BPFTRACE_DIR="${1#*=}"
          shift
          ;;
        --bpftrace_dir)
          if [[ -z "$2" || "$2" == -* ]]; then
            echo "Error: Missing value for --bpftrace_dir option"
            show_usage
            exit 1
          fi
          BPFTRACE_DIR="$2"
          shift 2
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
for ((p=0; p<${NUM_PORTS:-1}; p++)); do
    iface_var="DU_ETH_INTERFACE_${p}"
    mac_var="DU_MAC_ADDRESS_${p}"
    iface="${!iface_var}"
    expected_mac="${!mac_var}"
    if [[ -z "$iface" || -z "$expected_mac" ]]; then
        echo "Error: DU_ETH_INTERFACE_$p or DU_MAC_ADDRESS_$p not set. Please ensure setup1_DU.sh and setup2_RU.sh completed successfully."
        exit 1
    fi
    actual_mac=$(cat /sys/class/net/"${iface}"/address)
    if [ "$actual_mac" != "$expected_mac" ]; then
        echo "Error: MAC addresses do not match for interface $p. Expected $expected_mac (from config), but interface reports $actual_mac. Please ensure to run setup1_DU.sh and setup2_RU.sh before running run2_cuPHYcontroller.sh"
        exit 1
    fi
done

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
    CUDA_TRACER_BUILD="${cuBB_SDK}/testBenches/phase4_test_scripts/syscall_tracer/build_cuda_api_tracer.sh"
    if [[ ! -x "$CUDA_TRACER_BUILD" ]]; then
        echo "Error: build_cuda_api_tracer.sh not found or not executable at $CUDA_TRACER_BUILD"
        exit 1
    fi
    echo "Ensuring CUDA tracer library is up to date via $CUDA_TRACER_BUILD ..."
    "$CUDA_TRACER_BUILD" || {
        echo "Error: build_cuda_api_tracer.sh failed."
        exit 1
    }
    if [[ ! -f "$CUDA_TRACER_SO" ]]; then
        echo "Error: CUDA tracer library not found at $CUDA_TRACER_SO after build attempt."
        exit 1
    fi
    # Pre-create the tracer output file as the current user (NFS root_squash workaround:
    # mirrors what we do for perf record so the log stays owned by us when the binary
    # runs under sudo). Remove any stale log first -- a previous sudo'd run before this
    # pre-create block existed may have left a root-owned file we cannot truncate.
    mkdir -p "$(dirname "$CUDA_TRACER_OUTPUT")"
    rm -f "$CUDA_TRACER_OUTPUT" 2>/dev/null || sudo rm -f "$CUDA_TRACER_OUTPUT"
    touch "$CUDA_TRACER_OUTPUT"
    chmod 644 "$CUDA_TRACER_OUTPUT"
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
echo "nvidia-smi | grep \"KMD Version\"" # "KMD Version" is the old "Driver Version"; "CUDA UMD Version" is the old "CUDA Version"
nvidia-smi | grep "KMD Version" || echo "Warning: Could not retrieve KMD version from nvidia-smi"
echo "cat /proc/driver/nvidia/version"
cat /proc/driver/nvidia/version || echo "Warning: Could not read /proc/driver/nvidia/version"
echo "nvcc --version"
nvcc --version || echo "Warning: Could not retrieve nvcc version"
echo "uname -a"
uname -a || echo "Warning: Could not get uname -a info"

#----------------------------------------------------------------------------------
# Perf trace integration: monitor cuphycontroller output for L1 readiness,
# then automatically launch perf_trace_workers.sh in the background.
#----------------------------------------------------------------------------------
CUPHY_LOG=""
PERF_MONITOR_PID=""
BPFTRACE_MONITOR_PID=""

# Stop a readiness/trace monitor subshell gracefully (SIGINT so perf/bpftrace
# flush their data), then force-kill after a grace period of grace_ticks*0.5s.
_stop_trace_monitor() {
    local pid="$1" name="$2" grace_ticks="$3" _i
    { [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; } || return 0
    echo ""
    echo "Stopping $name (PID: $pid)..."
    kill -INT "$pid" 2>/dev/null || true
    for (( _i = 0; _i < grace_ticks; _i++ )); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

trace_cleanup() {
    # perf record -t <tids> self-exits when the traced threads die (cuphycontroller
    # end), so gracefully stopping the monitor subshell is enough.
    _stop_trace_monitor "$PERF_MONITOR_PID" "perf trace monitor" 10

    # bpftrace does NOT self-exit when cuphycontroller dies -- it attaches to
    # global tracepoints -- and its buffered -o output only flushes on a clean
    # SIGINT/SIGTERM. Signalling the monitor SUBSHELL does not reach the bpftrace
    # process, so send SIGINT directly to the bpftrace instance for this run (runs
    # as root via sudo) to make it dump its map, then let the wrapper finish
    # flushing + summarizing before the subshell is reaped.
    if [[ -n "$BPFTRACE_MONITOR_PID" ]] && kill -0 "$BPFTRACE_MONITOR_PID" 2>/dev/null; then
        echo ""
        echo "Stopping bpftrace (SIGINT to flush its map)..."
        if [[ -n "$BPF_OUTPUT_FILE" ]]; then
            sudo pkill -INT -f "bpftrace.*-o[[:space:]]*${BPF_OUTPUT_FILE}" 2>/dev/null || true
        fi
        for _i in $(seq 1 40); do
            kill -0 "$BPFTRACE_MONITOR_PID" 2>/dev/null || break
            sleep 0.5
        done
    fi
    _stop_trace_monitor "$BPFTRACE_MONITOR_PID" "bpftrace monitor" 10

    rm -f "${CUPHY_LOG}"
}

run_cuphy() {
    if [[ "$PERF_TRACE_ENABLED" = true || "$BPFTRACE_ENABLED" = true ]]; then
        script -q -e -f -c "$(printf '%q ' "$@")" "$CUPHY_LOG"
    else
        "$@"
    fi
}

# Shared readiness log + cleanup trap for the perf and/or bpftrace monitors.
if [[ "$PERF_TRACE_ENABLED" = true || "$BPFTRACE_ENABLED" = true ]]; then
    CUPHY_LOG=$(mktemp /tmp/cuphy_output.XXXXXX.log)
    trap trace_cleanup EXIT
fi

if [[ "$PERF_TRACE_ENABLED" = true ]]; then
    # `perf` on PATH is often just the linux-tools-common dispatcher wrapper, which
    # fails at runtime ("perf not found for kernel <ver>") when the per-kernel
    # package is missing. So test that perf actually WORKS (emits a real version
    # string), not merely that the wrapper exists on PATH.
    perf_functional() { perf --version 2>&1 | grep -q '^perf version'; }
    if ! perf_functional; then
        echo "perf not functional for kernel $(uname -r) -- installing linux-tools packages..."
        sudo apt-get update
        # Prefer the exact per-kernel package; fall back to tracking meta-packages.
        # apt stderr is intentionally NOT suppressed so failures (e.g. "Unable to
        # locate package ...") are visible in the run log.
        sudo apt-get install -y "linux-tools-$(uname -r)" \
            || sudo apt-get install -y linux-tools-nvidia-64k linux-tools-generic \
            || sudo apt-get install -y linux-tools-generic \
            || true

        # Self-heal for kernels whose exact linux-tools package is unavailable in the
        # configured repos (e.g. the nvidia-64k flavor): the wrapper dispatches to
        # /usr/lib/linux-tools/$(uname -r)/perf, but the install above may only have
        # provided a perf binary under a different kernel version's directory. A perf
        # binary from a nearby version records tracepoints fine, so symlink an
        # available one into the path the wrapper expects.
        if ! perf_functional; then
            PERF_WRAPPER_PATH="/usr/lib/linux-tools/$(uname -r)/perf"
            ALT_PERF=""
            for cand in /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf; do
                [[ -x "$cand" ]] || continue
                [[ "$cand" == "$PERF_WRAPPER_PATH" ]] && continue
                ALT_PERF="$cand"
                break
            done
            if [[ -n "$ALT_PERF" ]]; then
                echo "Exact per-kernel perf unavailable; linking existing perf binary:"
                echo "  $ALT_PERF -> $PERF_WRAPPER_PATH"
                sudo mkdir -p "$(dirname "$PERF_WRAPPER_PATH")"
                sudo ln -sf "$ALT_PERF" "$PERF_WRAPPER_PATH"
            fi
        fi

        if ! perf_functional; then
            echo "Error: perf still not functional after install/link attempts."
            echo "  linux-tools-$(uname -r) is not in the configured apt repositories, and no"
            echo "  alternative perf binary was found under /usr/lib/linux-tools*/ to link."
            echo "  Provide a perf binary at: /usr/lib/linux-tools/$(uname -r)/perf"
            exit 1
        fi
        echo "perf ready: $(perf --version 2>/dev/null)"
    else
        echo "perf already available: $(perf --version 2>/dev/null || echo 'unknown version')"
    fi

    PERF_TRACE_SCRIPT="${SCRIPT_DIR}/syscall_tracer/perf_trace_workers.sh"
    if [[ ! -x "$PERF_TRACE_SCRIPT" ]]; then
        echo "Error: perf_trace_workers.sh not found or not executable at $PERF_TRACE_SCRIPT"
        exit 1
    fi

    if [[ -z "$PERF_TRACE_DIR" ]]; then
        echo "Error: --perf_trace requires --perf_trace_dir <dir> to specify the output directory."
        exit 1
    fi
    mkdir -p "$PERF_TRACE_DIR"
    chmod 777 "$PERF_TRACE_DIR"
    echo "Perf trace output directory: $PERF_TRACE_DIR"

    OLD_FILES=$(find "$PERF_TRACE_DIR" -maxdepth 1 -type f \( -name "*.data" -o -name "*.data.old" -o -name "*.txt" -o -name "*.csv" -o -name "*.log" \) 2>/dev/null)
    if [[ -n "$OLD_FILES" ]]; then
        echo "Purging stale perf output files from $PERF_TRACE_DIR ..."
        echo "$OLD_FILES" | while IFS= read -r f; do echo "  rm $f"; rm -f "$f"; done
    fi

    # Determine the perf output file path. If the user already specified -o / --output
    # in PERF_TRACE_OPTS, extract it; otherwise inject our default.
    if [[ "$PERF_TRACE_OPTS" =~ (-o|--output)=([^[:space:]]+) ]]; then
        PERF_OUTPUT_FILE="${BASH_REMATCH[2]}"
    elif [[ "$PERF_TRACE_OPTS" =~ (-o|--output)[[:space:]]+([^[:space:]]+) ]]; then
        PERF_OUTPUT_FILE="${BASH_REMATCH[2]}"
    else
        PERF_OUTPUT_FILE="$PERF_TRACE_DIR/perf_syscalls.data"
        PERF_TRACE_OPTS="$PERF_TRACE_OPTS -o $PERF_OUTPUT_FILE"
    fi

    # NFS root_squash workaround: pre-create the output file as the current user
    # with world-writable permissions. When perf record (running as squashed nobody)
    # opens the existing file with O_TRUNC, it writes into it but preserves the
    # original ownership -- so the file stays owned by the current user.
    mkdir -p "$(dirname "$PERF_OUTPUT_FILE")"
    touch "$PERF_OUTPUT_FILE"
    chmod 666 "$PERF_OUTPUT_FILE"

    (
        waited=0
        max_polls=$(( PERF_TRACE_READY_TIMEOUT * 2 ))
        while (( waited < max_polls )); do
            if grep -qF "cuPHYController initialized, L1 is ready!" "$CUPHY_LOG" 2>/dev/null; then
                echo ""
                echo "========================================================="
                echo "  L1 ready detected -- launching perf_trace_workers.sh"
                echo "========================================================="
                echo ""
                # shellcheck disable=SC2086
                "$PERF_TRACE_SCRIPT" $PERF_TRACE_OPTS
                exit 0
            fi
            sleep 0.5
            (( ++waited )) || true
        done
        echo "Warning: timed out (${PERF_TRACE_READY_TIMEOUT}s) waiting for L1 readiness; perf trace not started." >&2
    ) &
    PERF_MONITOR_PID=$!
    echo "Perf trace monitor started (PID: $PERF_MONITOR_PID, timeout: ${PERF_TRACE_READY_TIMEOUT}s)"
fi

#----------------------------------------------------------------------------------
# bpftrace integration: same L1-readiness monitor pattern as perf, but launches
# bpftrace_trace_workers.sh (in-kernel all-syscall count). Runs on CPU 59 by
# default (orthogonal to perf's CPU 58) so both tracers can run together.
#----------------------------------------------------------------------------------
if [[ "$BPFTRACE_ENABLED" = true ]]; then
    if ! command -v bpftrace &>/dev/null; then
        echo "bpftrace not found -- installing..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq bpftrace
        if ! command -v bpftrace &>/dev/null; then
            echo "Error: failed to install bpftrace. Install manually with:"
            echo "  sudo apt-get update && sudo apt-get install -y bpftrace"
            exit 1
        fi
        echo "bpftrace installed successfully: $(bpftrace --version 2>/dev/null | head -1)"
    else
        echo "bpftrace already available: $(bpftrace --version 2>/dev/null | head -1)"
    fi

    BPFTRACE_SCRIPT="${SCRIPT_DIR}/syscall_tracer/bpftrace_trace_workers.sh"
    if [[ ! -x "$BPFTRACE_SCRIPT" ]]; then
        echo "Error: bpftrace_trace_workers.sh not found or not executable at $BPFTRACE_SCRIPT"
        exit 1
    fi

    if [[ -z "$BPFTRACE_DIR" ]]; then
        echo "Error: --bpftrace requires --bpftrace_dir <dir> to specify the output directory."
        exit 1
    fi
    mkdir -p "$BPFTRACE_DIR"
    chmod 777 "$BPFTRACE_DIR"
    echo "bpftrace output directory: $BPFTRACE_DIR"

    OLD_BPF_FILES=$(find "$BPFTRACE_DIR" -maxdepth 1 -type f \( -name "*.txt" -o -name "*.log" \) 2>/dev/null)
    if [[ -n "$OLD_BPF_FILES" ]]; then
        echo "Purging stale bpftrace output files from $BPFTRACE_DIR ..."
        echo "$OLD_BPF_FILES" | while IFS= read -r f; do echo "  rm $f"; rm -f "$f"; done
    fi

    # Determine the bpftrace raw-dump path: honor a user-supplied -o/--output in
    # BPFTRACE_OPTS, otherwise inject our default under BPFTRACE_DIR.
    if [[ "$BPFTRACE_OPTS" =~ (-o|--output)=([^[:space:]]+) ]]; then
        BPF_OUTPUT_FILE="${BASH_REMATCH[2]}"
    elif [[ "$BPFTRACE_OPTS" =~ (-o|--output)[[:space:]]+([^[:space:]]+) ]]; then
        BPF_OUTPUT_FILE="${BASH_REMATCH[2]}"
    else
        BPF_OUTPUT_FILE="$BPFTRACE_DIR/bpftrace_syscalls.txt"
        BPFTRACE_OPTS="$BPFTRACE_OPTS -o $BPF_OUTPUT_FILE"
    fi
    # Inject a default summary path if the user didn't specify one.
    if [[ ! "$BPFTRACE_OPTS" =~ (-s|--summary)[=[:space:]] ]]; then
        BPFTRACE_OPTS="$BPFTRACE_OPTS -s $BPFTRACE_DIR/bpftrace_syscall_summary.txt"
    fi

    # NFS root_squash workaround (mirrors perf): pre-create the dump file as the
    # current user so bpftrace (running as squashed nobody) keeps it owned by us
    # when it opens the existing file with O_TRUNC.
    mkdir -p "$(dirname "$BPF_OUTPUT_FILE")"
    touch "$BPF_OUTPUT_FILE"
    chmod 666 "$BPF_OUTPUT_FILE"

    (
        waited=0
        max_polls=$(( PERF_TRACE_READY_TIMEOUT * 2 ))
        while (( waited < max_polls )); do
            if grep -qF "cuPHYController initialized, L1 is ready!" "$CUPHY_LOG" 2>/dev/null; then
                echo ""
                echo "========================================================="
                echo "  L1 ready detected -- launching bpftrace_trace_workers.sh"
                echo "========================================================="
                echo ""
                # shellcheck disable=SC2086
                "$BPFTRACE_SCRIPT" $BPFTRACE_OPTS
                exit 0
            fi
            sleep 0.5
            (( ++waited )) || true
        done
        echo "Warning: timed out (${PERF_TRACE_READY_TIMEOUT}s) waiting for L1 readiness; bpftrace not started." >&2
    ) &
    BPFTRACE_MONITOR_PID=$!
    echo "bpftrace monitor started (PID: $BPFTRACE_MONITOR_PID, timeout: ${PERF_TRACE_READY_TIMEOUT}s)"
fi

if [ "$NSYS_ENABLED" = true ]; then
    echo "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 $NSYS_EXEC profile -t $NSYS_TRACE_TYPE -s none --cpuctxsw=none --run-as=root "${ADDITIONAL_OPTIONS[@]}" "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" $CONTROLLER_MODE
    run_cuphy sudo -E "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 $NSYS_EXEC profile -t $NSYS_TRACE_TYPE -s none --cpuctxsw=none --run-as=root "${ADDITIONAL_OPTIONS[@]}" "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" $CONTROLLER_MODE
    RET=$?
    # If you want to collect osrt events, you may want to increase he osrt threshold to 10us (e.g., via --osrt-threshold 10000)

else
    if [ "$MEMTRACE_ENABLED" = false ]; then
        echo "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 $WITH_TIMEOUT $GDB_SCRIPT $cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf $CONTROLLER_MODE
        run_cuphy sudo -E "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 $WITH_TIMEOUT $GDB_SCRIPT "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" $CONTROLLER_MODE
        RET=$?
    else
        echo "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 AERIAL_MEMTRACE=1 $WITH_TIMEOUT $GDB_SCRIPT $cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf $CONTROLLER_MODE
        run_cuphy sudo -E "${CUDA_TRACER_PREFIX[@]}" LD_BIND_NOW=1 AERIAL_MEMTRACE=1 $WITH_TIMEOUT $GDB_SCRIPT "$cuBB_SDK/$BUILD_DIR/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf" $CONTROLLER_MODE
        RET=$?
    fi
fi
exit $RET
