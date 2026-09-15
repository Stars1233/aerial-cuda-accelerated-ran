#!/usr/bin/env bash

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

# Automated perf tracing wrapper for cuBB DL/UL worker threads.
#
# This script:
#   1. Discovers DL/UL PhyDriver worker TIDs via list_cuphy_worker_tids.sh
#   2. Switches each worker thread to SCHED_OTHER (removes RT priority)
#   3. Launches perf record pinned to a chosen CPU, filtering the specified
#      syscall tracepoint(s) on those TIDs
#
# Usage:
#   sudo ./perf_trace_workers.sh [OPTIONS]
#
# Options:
#   -c, --cpu CPU_ID          CPU to pin perf to via taskset (default: 58)
#   -e, --event EVENT         Tracepoint event filter (default: arch-dependent; see source)
#   -o, --output PATH         Output perf data file (default: /tmp/perf_syscalls.data)
#   -m, --mmap-size SIZE      Per-CPU ring buffer size (default: 4M)
#   -p, --process NAME        Process name to match (default: cuphycontroller_scf)
#       --skip-sched           Skip the SCHED_OTHER policy change
#   -h, --help                Show this help and exit
#
# Examples:
#   sudo ./perf_trace_workers.sh
#   sudo ./perf_trace_workers.sh -c 60 -e syscalls:sys_enter_futex,syscalls:sys_exit_futex -o /tmp/run1.data
#   sudo ./perf_trace_workers.sh --skip-sched -m 8M

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CPU_ID=58
# Arch-aware default event list. perf record accepts a comma-separated list and
# writes all events into one data file; downstream parsers filter by event name.
#
# Page fault tracepoints differ by arch:
#   x86_64:   `exceptions:page_fault_user` and `exceptions:page_fault_kernel`
#             exist as real kernel tracepoints (clean user/kernel split).
#   aarch64:  those tracepoints are x86-only. The ARM64 kernel does not expose
#             an equivalent split, and the `:u` / `:k` privilege modifiers on
#             the `page-faults` software event are normally ignored by the
#             kernel, so we record a single combined `page-faults` event. The
#             summary script handles either schema.
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)
        DEFAULT_PAGE_FAULT_EVENTS="exceptions:page_fault_user,exceptions:page_fault_kernel"
        ;;
    aarch64|arm64)
        DEFAULT_PAGE_FAULT_EVENTS="page-faults"
        ;;
    *)
        echo "Warning: unknown arch '$ARCH'; falling back to generic page-faults (override with -e if needed)." >&2
        DEFAULT_PAGE_FAULT_EVENTS="page-faults"
        ;;
esac
EVENT="syscalls:sys_enter_futex,${DEFAULT_PAGE_FAULT_EVENTS}"
OUTPUT="/tmp/perf_syscalls.data"
MMAP_SIZE="4M"
PROCESS_NAME="cuphycontroller_scf"
SKIP_SCHED=0

usage() {
    sed -n '/^# Usage:/,/^[^#]/{ /^#/s/^# \?//p }' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cpu)        CPU_ID="$2";        shift 2 ;;
        -e|--event)      EVENT="$2";         shift 2 ;;
        -o|--output)     OUTPUT="$2";        shift 2 ;;
        -m|--mmap-size)  MMAP_SIZE="$2";     shift 2 ;;
        -p|--process)    PROCESS_NAME="$2";  shift 2 ;;
        --skip-sched)    SKIP_SCHED=1;       shift   ;;
        -h|--help)       usage 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage 1
            ;;
    esac
done

# --- Step 1: Discover worker TIDs ------------------------------------------

echo "=== Step 1: Discovering DL/UL worker TIDs ==="
"${SCRIPT_DIR}/list_cuphy_worker_tids.sh" "$PROCESS_NAME" -x
echo ""

ENV_FILE="${SCRIPT_DIR}/.cuphy_worker_tids.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: env file not created at $ENV_FILE" >&2
    echo "list_cuphy_worker_tids.sh may have failed to find worker threads." >&2
    exit 1
fi

source "$ENV_FILE"

if [[ ${#CUPHY_ALL_WORKER_TIDS[@]} -eq 0 ]]; then
    echo "Error: no DL/UL worker TIDs found." >&2
    exit 1
fi

echo "Worker TIDs (${#CUPHY_ALL_WORKER_TIDS[@]}): ${CUPHY_ALL_WORKER_TIDS[*]}"
echo ""

# --- Step 2: Switch scheduler policy to SCHED_OTHER ------------------------

if [[ "$SKIP_SCHED" -eq 0 ]]; then
    echo "=== Step 2: Setting SCHED_OTHER (priority 0) on worker TIDs ==="
    for tid in "${CUPHY_ALL_WORKER_TIDS[@]}"; do
        echo "  chrt -o -p 0 $tid"
        sudo chrt -o -p 0 "$tid"
    done
    echo "Done."
    echo ""
else
    echo "=== Step 2: Skipped (--skip-sched) ==="
    echo ""
fi

# --- Step 3: Launch perf record ---------------------------------------------

TID_LIST=$(IFS=,; echo "${CUPHY_ALL_WORKER_TIDS[*]}")

echo "=== Step 3: Starting perf record ==="
echo "  CPU:          $CPU_ID"
echo "  Event(s):     $EVENT"
echo "  TIDs:         $TID_LIST"
echo "  Ring buffer:  $MMAP_SIZE"
echo "  Output:       $OUTPUT"
echo ""

mkdir -p "$(dirname "$OUTPUT")"

echo "Recording... press Ctrl-C to stop."
echo ""

sudo taskset -c "$CPU_ID" \
    perf record --no-inherit \
    -e "$EVENT" \
    -t "$TID_LIST" \
    -m "$MMAP_SIZE" \
    -g --call-graph fp \
    -o "$OUTPUT"

echo ""
echo "Perf data written to: $OUTPUT"
echo "Inspect with:  sudo perf script -i $OUTPUT"
