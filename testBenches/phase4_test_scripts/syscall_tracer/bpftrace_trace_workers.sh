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

# Automated bpftrace wrapper for counting ALL syscall types issued by cuBB
# DL/UL worker threads.
#
# Unlike perf record/stat, bpftrace aggregates counts in-kernel (per-CPU,
# lock-free) with no per-event export, so it is light enough to run against the
# real-time workers without causing timing errors.
#
# This script:
#   1. Discovers DL/UL PhyDriver worker TIDs via list_cuphy_worker_tids.sh
#   2. Resolves those TIDs to thread names (comm), because bpftrace must filter
#      by comm, NOT tid: eBPF sees host/init-namespace TIDs while the TID list is
#      resolved in the container PID namespace, so the numbers never match.
#   3. Ensures debugfs is mounted (bpftrace needs /sys/kernel/debug/tracing)
#   4. Runs bpftrace, counting every syscall (syscalls:sys_enter_* wildcard,
#      named form) for the matched worker threads
#   5. Optionally post-processes the capture into a per-syscall/per-worker table
#      and emits a PASS/FAIL verdict (FAIL only if the capture is empty)
#
# Usage:
#   sudo ./bpftrace_trace_workers.sh [OPTIONS]
#
# Options:
#   -o, --output PATH         bpftrace map dump file (default: /tmp/bpftrace_syscalls.txt)
#   -s, --summary PATH        Summary table file (default: alongside the dump)
#   -d, --duration SECS       Auto-stop after SECS seconds (default: run until Ctrl-C)
#   -c, --cpu CPU_ID          CPU to pin the bpftrace process to via taskset (default: 59;
#                             kept orthogonal to the perf wrapper's CPU 58 so the two
#                             tracers can run together without sharing a core)
#   -p, --process NAME        Process name to match (default: cuphycontroller_scf)
#       --raw                 Use the lighter raw_syscalls:sys_enter single probe
#                             (numeric syscall ids) instead of the named wildcard
#       --no-summary          Skip post-processing; only write the raw dump
#   -h, --help                Show this help and exit
#
# Examples:
#   sudo ./bpftrace_trace_workers.sh
#   sudo ./bpftrace_trace_workers.sh -d 5 -o /tmp/run1/bpftrace_syscalls.txt
#   sudo ./bpftrace_trace_workers.sh --raw --no-summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT="/tmp/bpftrace_syscalls.txt"
SUMMARY=""
DURATION=""
# Default CPU 59 keeps bpftrace off perf's CPU 58 (perf_trace_workers.sh), so the
# two tracers run on orthogonal cores when used together.
CPU_ID=59
PROCESS_NAME="cuphycontroller_scf"
USE_RAW=0
RUN_SUMMARY=1

usage() {
    sed -n '/^# Usage:/,/^[^#]/{ /^#/s/^# \?//p }' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)    OUTPUT="$2";        shift 2 ;;
        -s|--summary)   SUMMARY="$2";       shift 2 ;;
        -d|--duration)  DURATION="$2";      shift 2 ;;
        -c|--cpu)       CPU_ID="$2";        shift 2 ;;
        -p|--process)   PROCESS_NAME="$2";  shift 2 ;;
        --raw)          USE_RAW=1;          shift   ;;
        --no-summary)   RUN_SUMMARY=0;      shift   ;;
        -h|--help)      usage 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage 1
            ;;
    esac
done

command -v bpftrace >/dev/null 2>&1 || {
    echo "Error: bpftrace not found. Install with: sudo apt-get install -y bpftrace" >&2
    exit 1
}

# --- Step 1: Discover worker TIDs ------------------------------------------

echo "=== Step 1: Discovering DL/UL worker TIDs ==="
"${SCRIPT_DIR}/list_cuphy_worker_tids.sh" "$PROCESS_NAME" -x
echo ""

ENV_FILE="${SCRIPT_DIR}/.cuphy_worker_tids.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: env file not created at $ENV_FILE" >&2
    exit 1
fi
source "$ENV_FILE"

if [[ ${#CUPHY_ALL_WORKER_TIDS[@]} -eq 0 ]]; then
    echo "Error: no DL/UL worker TIDs found." >&2
    exit 1
fi

# --- Step 2: Resolve TIDs -> comm (thread names) ---------------------------
# bpftrace filters by comm, not tid (PID-namespace: eBPF reports host TIDs, the
# list is container TIDs). comm is namespace-independent.

echo "=== Step 2: Resolving worker thread names (comm) ==="
declare -A SEEN_COMM=()
WORKER_COMMS=()
for tid in "${CUPHY_ALL_WORKER_TIDS[@]}"; do
    comm=$(cat "/proc/$tid/comm" 2>/dev/null || true)
    [[ -z "$comm" ]] && continue
    if [[ -z "${SEEN_COMM[$comm]:-}" ]]; then
        SEEN_COMM[$comm]=1
        WORKER_COMMS+=("$comm")
        echo "  TID $tid -> $comm"
    fi
done

if [[ ${#WORKER_COMMS[@]} -eq 0 ]]; then
    echo "Error: could not resolve any worker thread names from TIDs." >&2
    echo "Are the TIDs in $ENV_FILE stale? Re-run against the live process." >&2
    exit 1
fi
echo ""

# --- Step 3: Ensure debugfs is mounted (bpftrace requirement) --------------

# bpftrace needs tracefs under /sys/kernel/debug/tracing. Probe readability AS
# ROOT (sudo test): /sys/kernel/debug is mode 0700 root-only, so a `-r` test run
# as the current unprivileged user reports it missing even when it is mounted --
# and bpftrace itself runs as root, so root readability is what actually matters.
if ! sudo test -r /sys/kernel/debug/tracing/available_events 2>/dev/null; then
    echo "=== Step 3: Mounting debugfs (bpftrace needs /sys/kernel/debug/tracing) ==="
    sudo mount -t debugfs none /sys/kernel/debug 2>/dev/null || true
    if ! sudo test -r /sys/kernel/debug/tracing/available_events 2>/dev/null; then
        # Some kernels require tracefs to be mounted explicitly under debugfs.
        sudo mkdir -p /sys/kernel/debug/tracing 2>/dev/null || true
        sudo mount -t tracefs nodev /sys/kernel/debug/tracing 2>/dev/null || true
    fi
    if ! sudo test -r /sys/kernel/debug/tracing/available_events 2>/dev/null; then
        echo "Error: /sys/kernel/debug/tracing still unavailable after mount." >&2
        exit 1
    fi
    echo "Mounted."
    echo ""
fi

# --- Step 4: Build predicate and bpftrace program --------------------------

CPRED=""
for comm in "${WORKER_COMMS[@]}"; do
    CPRED+="comm == \"$comm\" || "
done
CPRED="${CPRED% || }"

if [[ "$USE_RAW" -eq 1 ]]; then
    # Single probe, numeric syscall id. args->id (arrow) matches the container's
    # bpftrace build; the summary parser accepts the numeric label as-is.
    PROG="tracepoint:raw_syscalls:sys_enter /$CPRED/ { @syscalls[comm, args->id] = count(); }"
else
    # Named wildcard: one probe per syscall, human-readable names via 'probe'.
    PROG="tracepoint:syscalls:sys_enter_* /$CPRED/ { @syscalls[comm, probe] = count(); }"
fi
if [[ -n "$DURATION" ]]; then
    PROG+=" interval:s:${DURATION} { exit(); }"
fi

# --- Step 5: Run bpftrace ---------------------------------------------------

echo "=== Step 5: Starting bpftrace ==="
echo "  Workers:      ${WORKER_COMMS[*]}"
echo "  Mode:         $([[ "$USE_RAW" -eq 1 ]] && echo 'raw_syscalls (numeric id)' || echo 'syscalls:sys_enter_* (named)')"
echo "  Stop:         $([[ -n "$DURATION" ]] && echo "auto after ${DURATION}s" || echo 'Ctrl-C')"
echo "  CPU:          $CPU_ID"
echo "  Output:       $OUTPUT"
echo ""
if [[ -z "$DURATION" ]]; then
    echo "Recording... start traffic, then press Ctrl-C to stop."
    echo ""
fi

mkdir -p "$(dirname "$OUTPUT")"
# bpftrace sends the map to -o, but attach/verifier errors go to stderr. Capture
# stderr to a sidecar log so failures survive an unattended harness run, and do
# NOT let a bpftrace non-zero exit abort before the summary step below (the
# summary correctly reports an empty capture as FAIL).
BPFTRACE_STDERR="${OUTPUT}.stderr.log"
set +e
sudo taskset -c "$CPU_ID" bpftrace -o "$OUTPUT" -e "$PROG" 2>"$BPFTRACE_STDERR"
BPFTRACE_RC=$?
set -e
if [[ -s "$BPFTRACE_STDERR" ]]; then
    echo "----- bpftrace stderr ($BPFTRACE_STDERR) -----" >&2
    cat "$BPFTRACE_STDERR" >&2
    echo "-----------------------------------------------" >&2
fi
if [[ $BPFTRACE_RC -ne 0 ]]; then
    echo "Warning: bpftrace exited with code $BPFTRACE_RC (see $BPFTRACE_STDERR)" >&2
fi

echo ""
echo "bpftrace map dump written to: $OUTPUT"

# --- Step 6: Post-process into summary + verdict ---------------------------

if [[ "$RUN_SUMMARY" -eq 1 ]]; then
    [[ -z "$SUMMARY" ]] && SUMMARY="$(dirname "$OUTPUT")/bpftrace_syscall_summary.txt"
    echo ""
    echo "=== Step 6: Summarizing ==="
    set +e
    python3 "${SCRIPT_DIR}/bpftrace_syscall_summary.py" -i "$OUTPUT" -o "$SUMMARY"
    rc=$?
    set -e
    echo ""
    echo "Summary: $SUMMARY  (verdict exit code: $rc)"
    exit "$rc"
fi

exit "$BPFTRACE_RC"
