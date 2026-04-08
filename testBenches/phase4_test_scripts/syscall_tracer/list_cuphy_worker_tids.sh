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

# List PID and TIDs of DL/UL worker threads for a running cuphycontroller_scf process.
# Run after cuphycontroller_scf has been started.
#
# Usage:
#   ./list_cuphy_worker_tids.sh [process_name]
#
# Default process name: cuphycontroller_scf
# Output: human-readable table and, if -x, export PID and TID list for use by other tools.
#

set -e

PROCESS_NAME="cuphycontroller_scf"
EXPORT_VARS=0
for arg in "$@"; do
  case "$arg" in
    -x|--export) EXPORT_VARS=1 ;;
    -h|--help)
      echo "Usage: $0 [process_name] [ -x | --export ]"
      echo "  process_name  Match this name (default: cuphycontroller_scf)"
      echo "  -x, --export  Write .cuphy_worker_tids.env with CUPHY_PID and TID arrays"
      exit 0
      ;;
    -*)
      ;;
    *)
      PROCESS_NAME="$arg"
      ;;
  esac
done

# Resolve script dir so we can source/write env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.cuphy_worker_tids.env"

# Find the actual process: pgrep -f may return a wrapper (e.g. sudo) when the
# binary is run as "sudo ./cuphycontroller_scf". We need the process whose
# main thread comm is the binary (truncated to 15 chars: "cuphycontroller_s")
# or a child of the wrapper that has the worker threads.
resolve_cuphy_pid() {
  local pids
  pids=()
  while read -r p; do
    [ -n "$p" ] && [ -d "/proc/$p" ] && pids+=("$p")
  done < <(pgrep -f "$PROCESS_NAME" 2>/dev/null || true)

  for p in "${pids[@]}"; do
    local comm=""
    [ -r "/proc/$p/comm" ] && comm=$(cat "/proc/$p/comm" 2>/dev/null | tr -d '\n')
    # Main process: comm is executable name (up to 15 chars)
    if [[ "$comm" == cuphycontroller* ]]; then
      echo "$p"
      return 0
    fi
    # Wrapper (e.g. sudo): use its child that exec'd the binary
    if [[ "$comm" == "sudo" ]] && [ -r "/proc/$p/task/$p/children" ]; then
      local children
      children=$(cat "/proc/$p/task/$p/children" 2>/dev/null) || true
      for c in $children; do
        [ -z "$c" ] || [ ! -d "/proc/$c" ] && continue
        local ccomm=""
        [ -r "/proc/$c/comm" ] && ccomm=$(cat "/proc/$c/comm" 2>/dev/null | tr -d '\n')
        if [[ "$ccomm" == cuphycontroller* ]]; then
          echo "$c"
          return 0
        fi
      done
    fi
  done

  # Fallback: pick the PID that has DlPhyDriver/UlPhyDriver threads (the real process).
  # Count worker threads for this PID and for its children (in case wrapper is what matched).
  count_worker_threads() {
    local p=$1
    local c=0
    [ ! -d "/proc/$p/task" ] && echo 0 && return
    for task in /proc/"$p"/task/*; do
      [ -d "$task" ] || continue
      local tcomm=""
      [ -r "$task/comm" ] && tcomm=$(cat "$task/comm" 2>/dev/null | tr -d '\n')
      case "$tcomm" in
        DlPhyDriver*|UlPhyDriver*) c=$((c + 1)) ;;
      esac
    done
    echo "$c"
  }
  for p in "${pids[@]}"; do
    local n
    n=$(count_worker_threads "$p")
    if [ "$n" -gt 0 ]; then
      echo "$p"
      return 0
    fi
    # No workers in this PID; if it has children (e.g. sudo -> cuphycontroller_scf), check them
    [ -r "/proc/$p/task/$p/children" ] || continue
    for c in $(cat "/proc/$p/task/$p/children" 2>/dev/null); do
      [ -z "$c" ] || [ ! -d "/proc/$c" ] && continue
      n=$(count_worker_threads "$c")
      if [ "$n" -gt 0 ]; then
        echo "$c"
        return 0
      fi
    done
  done
  return 1
}

pid=""
pid=$(resolve_cuphy_pid) || true

if [ -z "$pid" ]; then
  echo "Error: no running process matching '$PROCESS_NAME' found." >&2
  echo "Start cuphycontroller_scf first, then run this script." >&2
  exit 1
fi

echo "Process: $PROCESS_NAME"
echo "PID:     $pid"
echo ""

# Collect TIDs by role
dl_tids=()
ul_tids=()
other_tids=()

for task in /proc/"$pid"/task/*; do
  [ -d "$task" ] || continue
  tid=$(basename "$task")
  comm=""
  [ -r "$task/comm" ] && comm=$(cat "$task/comm" 2>/dev/null | tr -d '\n')
  case "$comm" in
    DlPhyDriver*)
      dl_tids+=("$tid:$comm")
      ;;
    UlPhyDriver*)
      ul_tids+=("$tid:$comm")
      ;;
    *)
      other_tids+=("$tid:$comm")
      ;;
  esac
done

echo "--- DL workers ---"
if [ ${#dl_tids[@]} -eq 0 ]; then
  echo "(none)"
else
  for entry in "${dl_tids[@]}"; do
    t="${entry%%:*}"
    c="${entry#*:}"
    echo "  TID $t  $c"
  done
fi

echo ""
echo "--- UL workers ---"
if [ ${#ul_tids[@]} -eq 0 ]; then
  echo "(none)"
else
  for entry in "${ul_tids[@]}"; do
    t="${entry%%:*}"
    c="${entry#*:}"
    echo "  TID $t  $c"
  done
fi

echo ""
echo "--- Other threads (same process) ---"
for entry in "${other_tids[@]}"; do
  t="${entry%%:*}"
  c="${entry#*:}"
  echo "  TID $t  $c"
done

# Combined list for strace (DL and UL only); arrays store tid:comm, export needs TIDs only
all_worker_tids=("${dl_tids[@]%%:*}" "${ul_tids[@]%%:*}")

if [ "$EXPORT_VARS" -eq 1 ]; then
  {
    echo "CUPHY_PID=$pid"
    echo "CUPHY_DL_TIDS=(${dl_tids[*]%%:*})"
    echo "CUPHY_UL_TIDS=(${ul_tids[*]%%:*})"
    echo "CUPHY_ALL_WORKER_TIDS=(${all_worker_tids[*]})"
  } > "$ENV_FILE"
  echo ""
  echo "Exported to $ENV_FILE (source it to use CUPHY_PID, CUPHY_DL_TIDS, CUPHY_UL_TIDS, CUPHY_ALL_WORKER_TIDS)"
fi

echo ""
echo "Use the PID and TIDs above with your preferred tracing or debugging tools."
