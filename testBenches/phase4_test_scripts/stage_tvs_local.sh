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
# Stage the test-vector working set for the current test from the
# (slow, ~1Gbps-capped) NFS testVectors directory onto fast local
# storage (default: /dev/shm tmpfs), so ru_emulator's load_tvs() reads
# from local RAM/NVMe instead of NFS.
#
# Idempotent: a file is (re)copied only if missing or its size/mtime
# differs from the source, so repeat runs of the same test stage
# nothing and return immediately.
#
# Layout produced (matches what run1_RU.sh --tv-base-path expects):
#   <stage-dir>/                 <- TV .h5 files
#   <stage-dir>/multi-cell/      <- launch pattern yaml(s)
#--------------------------------------------------------------------

set -u

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cuBB_SDK=${cuBB_SDK:-$(realpath "$SCRIPT_DIR/../..")}

SRC="$cuBB_SDK/testVectors"
STAGE_DIR="${TV_STAGE_DIR:-/dev/shm/cubb_tv_stage}"
PATTERN=""
NUM_CELLS=""
QUIET=0

show_usage() {
  cat <<EOF
Usage: $0 [options]

Stage the TV working set for the current test onto fast local storage.

Options:
  --pattern <p>         Pattern (e.g. 79). Default: \$PATTERN from test_config_summary.sh
  --num-cells <n>       Number of cells (e.g. 6). Default: \$NUM_CELLS from test_config_summary.sh
  --src <dir>           Source TV directory. Default: \$cuBB_SDK/testVectors
  --stage-dir <dir>     Local staging directory. Default: \${TV_STAGE_DIR:-/dev/shm/cubb_tv_stage}
  --quiet               Only print the resulting base path (for scripting)
  -h, --help            Show this help

On success the final stdout line is:  TV_BASE_PATH=<stage-dir>
Pass that directory to run1_RU.sh via --tv-base-path (run1_RU.sh --stage-local does this automatically).
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --pattern)    PATTERN="$2"; shift 2;;
    --num-cells)  NUM_CELLS="$2"; shift 2;;
    --src)        SRC="$2"; shift 2;;
    --stage-dir)  STAGE_DIR="$2"; shift 2;;
    --quiet)      QUIET=1; shift;;
    -h|--help)    show_usage; exit 0;;
    *) echo "Unknown option: $1" >&2; show_usage; exit 1;;
  esac
done

log() { [[ "$QUIET" -eq 1 ]] || echo "$@"; }

# Pull PATTERN / NUM_CELLS from the test summary if not supplied. Sourcing the
# summary sets both vars unconditionally, so save any CLI-provided values first
# and restore them afterwards — the summary only fills in what the user omitted.
SUMMARY="$cuBB_SDK/testBenches/phase4_test_scripts/test_config_summary.sh"
CLI_PATTERN="$PATTERN"; CLI_NUM_CELLS="$NUM_CELLS"
if [[ ( -z "$PATTERN" || -z "$NUM_CELLS" ) && -f "$SUMMARY" ]]; then
  # shellcheck disable=SC1090
  source "$SUMMARY"
fi
[[ -n "$CLI_PATTERN"   ]] && PATTERN="$CLI_PATTERN"
[[ -n "$CLI_NUM_CELLS" ]] && NUM_CELLS="$CLI_NUM_CELLS"

if [[ -z "$PATTERN" || -z "$NUM_CELLS" ]]; then
  echo "Error: pattern/num-cells not set. Run test_config.sh first, or pass --pattern/--num-cells." >&2
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "Error: source TV dir not found: $SRC" >&2
  exit 1
fi

# Locate the launch pattern that drives this test (F08 first, then nrSim).
LP_DIR="$SRC/multi-cell"
LP_FILE="$LP_DIR/launch_pattern_F08_${NUM_CELLS}C_${PATTERN}.yaml"
if [[ ! -f "$LP_FILE" ]]; then
  ALT="$SRC/launch_pattern_nrSim_${PATTERN}.yaml"
  if [[ -f "$ALT" ]]; then
    LP_FILE="$ALT"; LP_DIR="$SRC"
  else
    echo "Error: launch pattern not found:" >&2
    echo "  $SRC/multi-cell/launch_pattern_F08_${NUM_CELLS}C_${PATTERN}.yaml" >&2
    echo "  $ALT" >&2
    exit 1
  fi
fi

# Extract the .h5 working set referenced by the launch pattern.
mapfile -t TV_FILES < <(grep -Eo '[[:alnum:]_]+\.h5' "$LP_FILE" | sort -u)
if [[ "${#TV_FILES[@]}" -eq 0 ]]; then
  echo "Error: no .h5 files referenced in $LP_FILE" >&2
  exit 1
fi
# cuPhyChEstCoeffs.h5 is always needed (see copy_test_files.sh).
[[ -f "$SRC/cuPhyChEstCoeffs.h5" ]] && TV_FILES+=("cuPhyChEstCoeffs.h5")

log "Staging test: pattern=$PATTERN cells=$NUM_CELLS"
log "  source : $SRC"
log "  stage  : $STAGE_DIR"
log "  pattern file: $(basename "$LP_FILE")  (${#TV_FILES[@]} files referenced)"

# Free-space check. Count only bytes that will actually be written this run (dest
# absent or stale), mirroring the copy loop's idempotency test below. Summing the
# whole working set would spuriously fail an up-to-date re-run: df avail already
# excludes the previously staged files, so need(full set) > avail(capacity-staged)
# whenever the set exceeds the remaining free space, even when nothing is copied.
mkdir -p "$STAGE_DIR/multi-cell" || { echo "Error: cannot create $STAGE_DIR" >&2; exit 1; }
NEED_BYTES=0
for f in "${TV_FILES[@]}"; do
  s="$SRC/$f"; d="$STAGE_DIR/$f"
  [[ -f "$s" ]] || continue
  sz=$(stat -c %s "$s" 2>/dev/null || echo 0)
  if [[ -f "$d" ]]; then
    ds=$(stat -c %s "$d" 2>/dev/null || echo 0)
    sm=$(stat -c %Y "$s" 2>/dev/null || echo 0)
    dm=$(stat -c %Y "$d" 2>/dev/null || echo 0)
    [[ "$sz" == "$ds" && "$dm" -ge "$sm" ]] && continue   # up-to-date: no new bytes
  fi
  NEED_BYTES=$((NEED_BYTES + sz))
done
AVAIL_KB=$(df -Pk "$STAGE_DIR" | awk 'NR==2{print $4}')
NEED_KB=$(((NEED_BYTES + 1023) / 1024))   # round up: never undercount required space
if [[ "$NEED_KB" -gt "$AVAIL_KB" ]]; then
  echo "Error: not enough space in $STAGE_DIR: need $((NEED_KB/1024)) MiB, have $((AVAIL_KB/1024)) MiB." >&2
  echo "       Use --stage-dir to point at a larger local filesystem." >&2
  exit 1
fi
log "  to copy: $((NEED_BYTES/1024/1024)) MiB  (avail in stage: $((AVAIL_KB/1024)) MiB)"

# Copy launch pattern yaml(s). For F08 copy all cell-count variants (tiny) so a
# differing NUM_CELLS still finds its pattern; for nrSim copy the single file.
if [[ "$LP_DIR" == "$SRC/multi-cell" ]]; then
  cp -p "$SRC/multi-cell/"launch_pattern_F08_*_"${PATTERN}".yaml "$STAGE_DIR/multi-cell/" 2>/dev/null
else
  cp -p "$LP_FILE" "$STAGE_DIR/multi-cell/"
fi

# Progress bar. Rendered on STDERR so run1_RU.sh's $(...) capture of our stdout
# (the TV_BASE_PATH line) is unaffected and the bar still shows live. On a TTY it
# redraws in place; otherwise it prints a line at each 10% milestone (log-friendly).
TOTAL_FILES=${#TV_FILES[@]}
_last_decile=-1
show_progress() {
  [[ "$QUIET" -eq 1 ]] && return
  local cur=$1 width=40 done_mib=$(( copied_bytes / 1024 / 1024 ))
  local pct=$(( cur * 100 / TOTAL_FILES ))
  local filled=$(( cur * width / TOTAL_FILES )) bar empty
  printf -v bar   '%*s' "$filled"            ''; bar=${bar// /#}
  printf -v empty '%*s' "$(( width-filled ))" ''; empty=${empty// /-}
  if [[ -t 2 ]]; then
    printf '\r  staging [%s%s] %3d%%  %d/%d files  %d MiB copied ' \
           "$bar" "$empty" "$pct" "$cur" "$TOTAL_FILES" "$done_mib" >&2
  else
    local decile=$(( pct / 10 ))
    if [[ "$decile" -ne "$_last_decile" ]]; then
      _last_decile=$decile
      printf '  staging %3d%%  (%d/%d files, %d MiB copied)\n' \
             "$pct" "$cur" "$TOTAL_FILES" "$done_mib" >&2
    fi
  fi
}

# Idempotent copy: skip when dest exists with identical size and >= mtime.
copied=0; skipped=0; missing=0; copied_bytes=0; idx=0
for f in "${TV_FILES[@]}"; do
  idx=$((idx+1))
  s="$SRC/$f"; d="$STAGE_DIR/$f"
  if [[ ! -f "$s" ]]; then
    log "  WARN missing in source: $f"; missing=$((missing+1)); show_progress "$idx"; continue
  fi
  if [[ -f "$d" ]]; then
    ss=$(stat -c %s "$s"); ds=$(stat -c %s "$d")
    sm=$(stat -c %Y "$s"); dm=$(stat -c %Y "$d")
    if [[ "$ss" == "$ds" && "$dm" -ge "$sm" ]]; then
      skipped=$((skipped+1)); show_progress "$idx"; continue
    fi
  fi
  if cp -p "$s" "$d"; then
    copied=$((copied+1)); copied_bytes=$((copied_bytes + $(stat -c %s "$s")))
  else
    echo "Error: failed to copy $s -> $d (staging aborted to avoid a partial set)" >&2
    exit 1
  fi
  show_progress "$idx"
done
# Terminate the in-place bar line with a newline on a TTY.
[[ "$QUIET" -eq 1 || ! -t 2 ]] || printf '\n' >&2

log "  staged: copied=$copied ($((copied_bytes/1024/1024)) MiB)  skipped(up-to-date)=$skipped  missing=$missing"
[[ "$missing" -gt 0 ]] && log "  NOTE: $missing referenced file(s) missing from source; emulator may still run if unused."

# Final machine-readable line.
echo "TV_BASE_PATH=$STAGE_DIR"
