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

# Three-step orchestrator:
#   Step 1 — Run cuBB test and dump SRS TV (H5 files)
#   Step 2 — Generate cuMAC TV via run_cumcp_sa.sh inside gNB container
#   Step 3 — Run cuBB + cuMAC-CP combined 4-app test

script_name=${0##*/}
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# ── Usage ─────────────────────────────────────────────────────────────────────
function usage {
    echo "Usage: $script_name <test_case_string> <duration> [alloc_type] [gpu_share] [options]"
    echo ""
    echo "  test_case_string  Full cuBB test case string passed to parse_test_config_params.sh."
    echo "                    Format: F08_<N>C_<pattern>[_modifiers...]"
    echo "                    E.g. \"F08_3C_66c_BFP9_STT455000_EH_1P\""
    echo "                    E.g. \"F08_6C_79_MODCOMP_STT480000_EH_1P\""
    echo "  duration          Duration (s).  E.g. 300"
    echo "  alloc_type        (optional) cuMAC alloc type for TV dir naming.  Default: 1"
    echo "                    Only used by cumcp-sa / ue-group / other; ignored for cubb / cubb-srs."
    echo "  gpu_share         (optional) GPU share mode.  Default: 0"
    echo "                    Only used by cumcp-sa / ue-group / other; ignored for cubb / cubb-srs."
    echo ""
    echo "  --test <mode>     Test mode (default: ue-group):"
    echo "    cubb            Normal cuBB Test — no SRS TV dump"
    echo "    cubb-srs        cuBB Test + SRS TV Dump (prerequisite for ue-group)"
    echo "    cumcp-sa        cuMAC-CP Standalone TV Generation only"
    echo "    ue-group        cuBB + cuMAC-CP UE Group:"
    echo "                      gpu_share=2 → cuBB Test + SRS TV Dump → cuMAC-CP SA → cuBB + cuMAC-CP Combined Test"
    echo "                      gpu_share≠2 → cuMAC-CP SA → cuBB + cuMAC-CP Combined Test"
    echo "                      Requires CUMAC_TASK_MASK=0x20 (enable_cubb=1, srs enabled)"
    echo "    other           cuBB + cuMAC-CP Other Tasks:"
    echo "                      cuBB + cuMAC-CP Combined Test only (enable_cubb=0)"
    echo ""
    echo "  -b, --build_dir <dir>"
    echo "                    Build directory under \$CUBB_SDK containing the cumac_cp binary."
    echo "                    Default: auto-detected from RUN2_CUPHYCONTROLLER_PARAMS."
    echo "                    E.g. -b build.perf.x86_64"
    echo "  --cumac_task <mask>"
    echo "                    cuMAC-CP task bitmask (default: \$CUMAC_TASK_MASK or 0xF)."
    echo "                    E.g. --cumac_task 0x20"
    echo ""
    echo "Required env (set in env.sh):"
    echo "  GNB_SERVER   gNB server hostname or IP"
    echo "  RU_SERVER    RU server hostname or IP"
    echo "  CUBB_HOST    NFS path — same on all hosts and inside containers (data: YAML, logs, TV)"
    echo "  HOST_CONFIG  Platform string for parse_test_config_params.sh, e.g. SPRK_R750"
    echo ""
    echo "Optional env:"
    echo "  CUBB_SDK          \$cuBB_SDK inside containers (default: /opt/nvidia/cuBB)"
    echo "  gNB_RU_Share      1 = gNB and RU share the NFS SDK; skip RU-side setup (default: 0)"
    echo "  TV_SRC            Source dir for copy_test_files.sh (QA: /home/aerial/nfs/GPU_test_input)"
    echo "  LOG_BASE          Parent dir for run logs (default: \$CUBB_HOST/logs)"
    echo "  SSH_USER/SSH_PASS SSH credentials (default: \$USER / key auth)"
    echo "  CONTAINER         Docker container name (default: c_aerial_\${SSH_USER})"
    echo "  DUMP_SRS_SLOT_NUM SRS slots to dump in cubb-srs / ue-group (default: 4)"
    echo "  CUMAC_TASK_MASK   Default cuMAC-CP task bitmask (overridden by --cumac_task)"
    echo "  SRS_SLOT_LAG      SRS slot lag applied in ue-group Step 3 (default: 13)"
    echo "  TDD_PATTERN       TDD slot pattern for Step 2 TV generation (SDDDS / SSSSS)"
    echo "  SHM_LEVEL         Global nvlog SHM level"
    echo "  LOG_SIZE          Maximum rotating nvlog file size in bytes"
    echo "  NVIPC_PCAP_ENABLE 1 = enable nvIPC PCAP and collect nvipc.pcap (default: 0)"
    echo "  TRY               1 = keep result checking active through transient failures"
    echo ""
    echo "Examples:"
    echo "  $script_name \"F08_1C_66c_BFP9_STT455000_EH_1P\" 60 --test cubb"
    echo "  $script_name \"F08_1C_66c_BFP9_STT455000_EH_1P\" 60 --test cubb-srs"
    echo "  $script_name \"F08_1C_66c_BFP9_STT455000_EH_1P\" 60 1 2 --test cumcp-sa"
    echo "  $script_name \"F08_1C_66c_BFP9_STT455000_EH_1P\" 60 1 2 --cumac_task 0x20 --test ue-group"
    echo "  $script_name \"F08_1C_66c_BFP9_STT455000_EH_1P\" 60 1 0 --cumac_task 0xF --test other"
}

# Handle help before sourcing env.sh so usage is always available, even when
# required environment variables have not been configured yet.
for _arg in "$@"; do
    case "$_arg" in -h|--help) usage; exit 0 ;; esac
done
unset _arg

print_missing_required_env() {
    local var missing=()
    for var in GNB_SERVER RU_SERVER CUBB_HOST HOST_CONFIG; do
        [ -n "${!var:-}" ] || missing+=("${var}")
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        echo "Missing required environment variables: ${missing[*]}"
        echo "Export them before running, or set them in env.sh:"
        for var in "${missing[@]}"; do
            echo "  export ${var}=<value>"
        done
        echo ""
    fi
}

# ── Source env and config ─────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/env.sh" ]; then
    # shellcheck source=env.sh
    source "$SCRIPT_DIR/env.sh" || {
        echo "ERROR: env.sh failed — configure the required variables below and retry."
        print_missing_required_env
        usage
        exit 1
    }
else
    echo "ERROR: env.sh not found at $SCRIPT_DIR/env.sh"
    usage
    exit 1
fi

if [ -f "$SCRIPT_DIR/config.sh" ]; then
    # shellcheck source=config.sh
    source "$SCRIPT_DIR/config.sh"
fi

# ── Parse positional arguments ────────────────────────────────────────────────
# test_case_string and duration are required.  alloc_type and gpu_share are
# OPTIONAL: they only matter for cumcp-sa / ue-group / other (cuMAC TV dir naming
# + GPU share); cubb / cubb-srs do not use them.  When omitted they default below.
if [ $# -lt 2 ]; then
    usage
    exit 1
fi

test_case_string="$1"; shift
duration="$1"; shift
alloc_type=1
gpu_share=0
# Consume the next positionals as alloc_type / gpu_share only if they are not flags.
[[ $# -gt 0 && "$1" != -* ]] && { alloc_type="$1"; shift; }
[[ $# -gt 0 && "$1" != -* ]] && { gpu_share="$1"; shift; }

RUN_TEST="ue-group"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            [[ -n "${2+x}" ]] || { echo "ERROR: --test requires a mode argument"; usage; exit 1; }
            RUN_TEST="$2"; shift 2 ;;
        --test=*)
            RUN_TEST="${1#*=}"; shift ;;
        -b|--build_dir)
            [[ -n "${2+x}" ]] || { echo "ERROR: $1 requires an argument"; usage; exit 1; }
            CUMAC_BUILD_DIR="$2"; shift 2 ;;
        --build_dir=*)
            CUMAC_BUILD_DIR="${1#*=}"; shift ;;
        --cumac_task)
            [[ -n "${2+x}" ]] || { echo "ERROR: --cumac_task requires a mask argument"; usage; exit 1; }
            CUMAC_TASK_MASK="$2"; shift 2 ;;
        --cumac_task=*)
            CUMAC_TASK_MASK="${1#*=}"; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ ! "$CUMAC_TASK_MASK" =~ ^(0[xX][0-9a-fA-F]+|[0-9]+)$ ]]; then
    echo "ERROR: Invalid --cumac_task value: '$CUMAC_TASK_MASK' (expected decimal or hexadecimal mask)"
    exit 1
fi

# ── Parse test_group, cell_num, and pattern from test_case_string ─────────────
# Format: F08_3C_66c_BFP9_STT455000_EH_1P  →  group=F08  cell_num=3  pattern=66c
test_group=$(echo "$test_case_string" | cut -d_ -f1)
cell_field=$(echo "$test_case_string" | cut -d_ -f2)       # e.g. "3C"
cell_num=$(echo "$cell_field" | grep -oP '^\d+')
pattern=$(echo "$test_case_string" | cut -d_ -f3)          # e.g. "60" or "59c" or "66c"
if [ -z "$cell_num" ]; then
    echo "ERROR: Cannot parse cell_num from test_case_string='$test_case_string'"
    echo "       Expected format: F08_<N>C_<pattern>[_modifiers...]"
    exit 1
fi
# Short test_case used by run_cumcp_sa.sh (Step 2): e.g. "3C_66c"
test_case="${cell_num}C_${pattern}"

# ── Derived paths ─────────────────────────────────────────────────────────────
# phase4 / cumcp_scripts_sdk: paths to SDK scripts INSIDE containers, under
# CUBB_SDK (default /opt/nvidia/cuBB; normally different from the CUBB_HOST NFS
# data path, in both ENG and QA).
# check_result_cumcp.py runs locally — use $SCRIPT_DIR (always correct, independent of CUBB_HOST).
phase4="$CUBB_SDK/testBenches/phase4_test_scripts"
cumcp_scripts_sdk="$CUBB_SDK/cuPHY-CP/testMAC/scripts"
cumac_tv_dir="$CUBB_SDK/testVectors/cumac.${cell_num}c.type${alloc_type}.gpu_share${gpu_share}"
cumac_tv_link="$CUBB_SDK/testVectors/cumac"
# The parser runs inside the container; the top-level script reads the same
# bind-mounted file through the corresponding host path.
TEST_PARAMS_HOST="$CUBB_HOST/test_params.sh"
TEST_PARAMS_SDK="$CUBB_SDK/test_params.sh"

# ── Log path setup ────────────────────────────────────────────────────────────
ts=$(date -u "+%Y%m%d_%H%M%S")
# LOG_BASE: parent directory for all runs.
#   Default: $CUBB_HOST/logs  (dev)
#   QA:      $CUBB_HOST/Log/Sanity/cuMAC_CP  (set before running)
# Each run creates a timestamped subdir inside LOG_BASE, renamed _PASS/_FAIL/_ERR at the end.
_log_parent="${LOG_BASE:-$CUBB_HOST/logs}"
LOG_RUN_DIR="$_log_parent/${ts}_${test_case_string}"
mkdir -p "$LOG_RUN_DIR"
chmod 775 "$LOG_RUN_DIR"

# 'latest' symlink for easy monitoring
ln -sfn "$LOG_RUN_DIR" "$_log_parent/latest"

MAIN_LOG="$LOG_RUN_DIR/main.log"

# ── Logging helpers ───────────────────────────────────────────────────────────
_log() {
    local label="$1"; local msg="$2"
    local ts_now; ts_now=$(date -u '+%H:%M:%S.%6N')
    echo "[${label}][${ts_now}] ${msg}" | tee -a "$MAIN_LOG"
}

log_info()  { _log "INFO" "$1"; }
log_step()  { _log "STEP" "$1"; }
log_cfg()   { _log "CFG " "$1"; }
log_cmd()   { _log "CMD " "$1"; }
log_check() { _log "CHK " "$1"; }
log_err()   { _log "ERR " "$1"; }

# SSH helpers are defined in env.sh (sourced above).
# Route their log output through this script's structured logger.
_ssh_log() { log_cmd "$1"; }

# ── Prerequisite checks ───────────────────────────────────────────────────────
check_srs_tv() {
    log_check "Verifying SRS TV files inside gNB container (/tmp/cubb_srs_buffers_*.h5) ..."
    local count _ssh_cmd
    if [ -n "${GNB_SSH_PASS:-}" ]; then
        count=$(SSHPASS="${GNB_SSH_PASS}" sshpass -e ssh "${_GNB_SSH_USER_AT}${GNB_SERVER}" \
            "docker exec -i ${CONTAINER} bash -c \
            'ls /tmp/cubb_srs_buffers_*.h5 2>/dev/null | wc -l'" 2>/dev/null || echo 0)
    else
        count=$(ssh "${_GNB_SSH_USER_AT}${GNB_SERVER}" \
            "docker exec -i ${CONTAINER} bash -c \
            'ls /tmp/cubb_srs_buffers_*.h5 2>/dev/null | wc -l'" 2>/dev/null || echo 0)
    fi
    count=$(echo "$count" | tr -d '[:space:]')
    if [ "${count:-0}" -eq 0 ]; then
        log_err "SRS H5 files missing in gNB container — re-run Step 1 with DUMP_SRS_SLOT_NUM >= 1"
        return 1
    fi
    log_check "SRS TV: ${count} file(s) found. OK."
    return 0
}

# _resolve_param_on_host: echo a test_params.sh variable's value as resolved ON the
# host that will execute it.  BUILD_AERIAL_PARAMS / RUN*_PARAMS embed
# build.perf.$(uname -m), so they are arch-specific and the gNB/RU archs may differ.
# generate_test_params calls this once per param and stores the result back into a
# global, so later steps can use the values directly without re-sourcing
# test_params.sh over SSH.  Strips the _ssh_log "[SSH ]" line and the -tt PTY CRs.
#   $1 = ssh helper (ssh_gnb | ssh_ru), $2 = variable name
_resolve_param_on_host() {
    local ssh_fn="$1" var="$2"
    "$ssh_fn" "source ${TEST_PARAMS_SDK} && echo \$${var}" 2>/dev/null \
        | grep -v '^\[' | tr -d '\r' | tail -1
}

# detect_cumac_build_dir: cuMAC-CP build dir relative to $CUBB_SDK (via stdout).
# CUMAC_BUILD_DIR override wins; otherwise reuse cuBB's build dir parsed from
# RUN2_CUPHYCONTROLLER_PARAMS — already resolved to the gNB value (where cuMAC-CP
# runs) by generate_test_params, so a plain local parse is correct.
detect_cumac_build_dir() {
    if [ -n "${CUMAC_BUILD_DIR:-}" ]; then
        echo "${CUMAC_BUILD_DIR}"
        return 0
    fi
    local cubb_bdir
    cubb_bdir=$(echo "${RUN2_CUPHYCONTROLLER_PARAMS:-}" \
        | grep -oP -- '--build_dir(=|\s+)\K\S+' | head -1)
    echo "${cubb_bdir}"
}

# build_if_needed: build Aerial SDK on gNB and RU independently — each host is
# built only when ITS OWN (arch-specific) build folder is absent.  The gNB and RU
# build dirs differ on cross-arch setups (e.g. build.perf.aarch64 vs build.x86_64),
# so each host must be probed separately.  Skipping when present avoids redundant
# rebuilds across Step 1 and Step 3.
#   gNB build dir: from RUN2_CUPHYCONTROLLER_PARAMS (via detect_cumac_build_dir)
#   RU  build dir: parsed from RUN1_RU_PARAMS (already RU-resolved)
# BUILD_AERIAL_PARAMS_GNB / BUILD_AERIAL_PARAMS_RU were pre-resolved per host in
# generate_test_params, so they are passed directly.
build_if_needed() {
    local gnb_bdir ru_bdir
    gnb_bdir=$(detect_cumac_build_dir)
    # Fall back to build.$(uname -m) when unresolved: otherwise an empty gnb_bdir
    # makes the existence check "[ -d ${CUBB_SDK}/ ]" always true and silently
    # skips the gNB build.  A wrong-but-nonempty name just forces a (safe) build.
    gnb_bdir="${gnb_bdir:-build.$(uname -m)}"
    ru_bdir=$(echo "${RUN1_RU_PARAMS:-}" | grep -oP -- '--build_dir(=|\s+)\K\S+' | head -1)
    ru_bdir="${ru_bdir:-build.$(uname -m)}"

    local gnb_exists ru_exists
    gnb_exists=$(ssh_gnb "[ -d ${CUBB_SDK}/${gnb_bdir} ] && echo yes || echo no" 2>/dev/null | grep -v '^\[' | tr -d '\r' | tail -1)
    ru_exists=$(ssh_ru  "[ -d ${CUBB_SDK}/${ru_bdir} ] && echo yes || echo no"  2>/dev/null | grep -v '^\[' | tr -d '\r' | tail -1)

    if [[ "${gnb_exists}" = "yes" && "${FORCE_REBUILD_CUBB}" != "1" ]]; then
        log_info "gNB build folder ${CUBB_SDK}/${gnb_bdir} exists — skipping gNB build."
    else
        log_step "Building on gNB [${gnb_bdir}]: build_aerial_sdk.sh ${BUILD_AERIAL_PARAMS_GNB}"
        ssh_gnb "$phase4/build_aerial_sdk.sh ${BUILD_AERIAL_PARAMS_GNB}" \
            || { log_err "build_aerial_sdk.sh (gNB) failed"; return 1; }
    fi

    if [[ "${ru_exists}" = "yes" && "${FORCE_REBUILD_CUBB}" != "1" ]]; then
        log_info "RU build folder ${CUBB_SDK}/${ru_bdir} exists — skipping RU build."
    else
        log_step "Building on RU [${ru_bdir}]: build_aerial_sdk.sh ${BUILD_AERIAL_PARAMS_RU}"
        ssh_ru "$phase4/build_aerial_sdk.sh ${BUILD_AERIAL_PARAMS_RU}" \
            || { log_err "build_aerial_sdk.sh (RU) failed"; return 1; }
    fi
}

check_cumac_tv() {
    log_check "Verifying cuMAC TV symlink at $cumac_tv_link ..."
    local n target
    n=$(ssh_gnb "test -L ${cumac_tv_link} && ls ${cumac_tv_link}/ 2>/dev/null | wc -l || echo 0" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
    if [ "${n:-0}" -eq 0 ]; then
        log_err "Symlink ${cumac_tv_link} does not exist or is empty — re-run Step 2"
        return 1
    fi
    target=$(ssh_gnb "readlink -f ${cumac_tv_link}" 2>/dev/null | grep -v '^\[' | tail -1)
    log_check "cuMAC TV: symlink → $target ($n files). OK."
    return 0
}

# ── Wait-for-log-pattern: poll NFS screenlog until pattern appears ────────────
# Mirrors wait_process() in cubb_phase4_execute env_setup.sh.
wait_for_log_pattern() {
    local logfile="$1"; local pattern="$2"; local timeout_sec="$3"; local label="$4"
    local elapsed=0
    log_check "Waiting for '$label' ready signal ($pattern) in $logfile (timeout ${timeout_sec}s) ..."
    while [ $elapsed -lt "$timeout_sec" ]; do
        if [ -f "$logfile" ] && grep -qE "$pattern" "$logfile" 2>/dev/null; then
            log_check "$label ready (elapsed ${elapsed}s)."
            sleep 1
            return 0
        fi
        if [ -f "$logfile" ] && grep -qE "_exit_code=[0-9]+" "$logfile" 2>/dev/null; then
            log_err "$label exited before reporting ready."
            printf '%s\n' "$label" > "$(dirname "$logfile")/.crash_detected"
            return 1
        fi
        if [ $(( elapsed % 20 )) -eq 0 ] && [ $elapsed -gt 0 ]; then
            log_check "Still waiting for '$label' ... (${elapsed}s elapsed)"
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    log_err "$label did NOT signal ready within ${timeout_sec}s. Aborting."
    return 1
}

# ── Monitor screen sessions for unexpected exits ─────────────────────────────
# Runs in the background alongside check_result_cumcp.py.
# If any app exits (exit-code line appears in its screenlog) before the checker
# finishes, the checker is killed so the test fails fast instead of timing out.
# Mirrors moniter_all_running() in cubb_phase4_execute env_setup.sh.
monitor_screens() {
    local log_dir="$1"; local titles="$2"; local checker_pid="$3"
    while true; do
        sleep 2
        if ! kill -0 "$checker_pid" 2>/dev/null; then
            return 0
        fi
        for title in $titles; do
            local logfile="$log_dir/screenlog_${title}.log"
            if [ -f "$logfile" ] && grep -qE "${title}_exit_code=" "$logfile" 2>/dev/null; then
                log_err "Process '$title' exited unexpectedly — killing checker (pid=$checker_pid)."
                printf '%s\n' "$title" > "${log_dir}/.crash_detected"
                kill "$checker_pid" 2>/dev/null
                return 1
            fi
        done
    done
}

# ── Runtime diagnostics and application log collection ───────────────────────
gnb_procs="cuphycontroller_scf cumac_cp test_mac"
ru_procs="ru_emulator"

_core_snapshot_cmd() {
    local process_names="$1" pname cmd
    cmd="echo \"process_names: ${process_names}\"; echo \"----------------------------------------\"; "
    for pname in ${process_names}; do
        cmd="${cmd}pids=\$(pidof ${pname} 2>/dev/null); "
        cmd="${cmd}if [ -n \"\${pids}\" ]; then "
        cmd="${cmd}echo \"===== NAME: ${pname} PID: \${pids} =====\"; "
        cmd="${cmd}ps H -o pid,tid,comm,policy,psr,priority,%cpu,%mem,vsz,rss \${pids}; echo; fi; "
    done
    echo "${cmd}"
}

collect_core_snapshots() {
    local log_dir="$1"
    log_step "Capturing DU/RU process snapshots and GPU state ..."
    ( _ssh_log() { :; }; SSH_LOG_EXEC=0
      ssh_gnb "$(_core_snapshot_cmd "phc2sys ptp4l ${gnb_procs}")"
    ) > "${log_dir}/core.du.log" 2>&1 || true
    ( _ssh_log() { :; }; SSH_LOG_EXEC=0
      ssh_ru "$(_core_snapshot_cmd "phc2sys ptp4l ${ru_procs}")"
    ) > "${log_dir}/core.ru.log" 2>&1 || true
    ssh_gnb "nvidia-smi" > "${log_dir}/smi.log" 2>&1 || true
}

l2_adapter_config_path() {
    local du_platform="${HOST_CONFIG%%_*}"
    echo "${CUBB_SDK}/cuPHY-CP/cuphycontroller/config/l2_adapter_config_${test_group}_${du_platform}.yaml"
}

configure_nvipc_pcap() {
    case "${NVIPC_PCAP_ENABLE:-0}" in
        0|"")
            return 0
            ;;
        1)
            local l2_yaml
            l2_yaml=$(l2_adapter_config_path)
            log_step "Enabling nvIPC PCAP in ${l2_yaml} ..."
            cfg_set "pcap_enable" "1" "${l2_yaml}" || return 1
            ;;
        *)
            log_err "NVIPC_PCAP_ENABLE must be 0 or 1 (got '${NVIPC_PCAP_ENABLE}')"
            return 1
            ;;
    esac
}

collect_runtime_logs() {
    local log_dir="$1"
    local gnb_bdir l2_yaml ipc_dump pcap_collect
    gnb_bdir=$(detect_cumac_build_dir)
    l2_yaml=$(grep -o "L2Adapter config file:.*\.yaml" "${log_dir}/screenlog_phy.log" 2>/dev/null \
        | awk '{print $NF}' | head -1)
    l2_yaml="${l2_yaml:-$(l2_adapter_config_path)}"
    ipc_dump="${CUBB_SDK}/${gnb_bdir}/cuPHY-CP/gt_common_libs/nvIPC/tests/dump/ipc_dump"
    pcap_collect="${CUBB_SDK}/${gnb_bdir}/cuPHY-CP/gt_common_libs/nvIPC/tests/pcap/pcap_collect"

    log_step "Collecting rotating application logs to ${log_dir} ..."
    {
        ssh_gnb "cp -f /tmp/phy.log* /tmp/testmac.log* /tmp/cumac_cp.log* ${log_dir}/ 2>/dev/null || true"
        ssh_gnb "cp -f /tmp/ul_packet_times.txt ${log_dir}/ 2>/dev/null || true"
        ssh_ru  "cp -f /tmp/ru.log* ${log_dir}/ 2>/dev/null || true"
    } > "${log_dir}/nvlog_collect.log" 2>&1

    log_step "Collecting nvIPC queue dump ..."
    ssh_gnb "cd ${log_dir} && sudo ${ipc_dump} nvipc ${l2_yaml}" \
        > "${log_dir}/ipc_dump.phy.nvipc.log" 2>&1 || true

    if [ "${NVIPC_PCAP_ENABLE:-0}" = "1" ]; then
        log_step "Collecting nvIPC PCAP ..."
        # Use a short relative destination: nvIPC's PCAP writer has a small
        # internal path buffer and truncates long absolute log directory names.
        ssh_gnb "cd ${log_dir} && sudo ${pcap_collect} nvipc ." \
            >> "${log_dir}/nvlog_collect.log" 2>&1 || true
    fi
}

# ── Dump configurations and system info to a step log directory ───────────────
# Call after kill_all so all processes have stopped and config files are stable.
# Copies: cuphycontroller/ru-emulator/testMAC/cuMAC-CP/nvlog configs from containers,
# test_params.sh, and a system-info snapshot (kernel, GPU driver, CUDA version).
# Failures are non-fatal — we always attempt a best-effort capture.
dump_configs() {
    local log_dir="$1"
    local cfg_dir="${log_dir}/configs"
    log_step "Dumping configs and system info to ${cfg_dir} ..."
    mkdir -p "${cfg_dir}"

    # test_params.sh — already on NFS
    [ -f "${TEST_PARAMS_HOST}" ] && cp "${TEST_PARAMS_HOST}" "${cfg_dir}/" 2>/dev/null || true

    # Extract the cuphycontroller YAML actually used from the phy screenlog.
    # cuphycontroller logs: [CTL.SCF] Config file: /path/to/cuphycontroller_X.yaml
    local phy_yaml
    phy_yaml=$(grep -o "Config file:.*\.yaml" "${log_dir}/screenlog_phy.log" 2>/dev/null \
               | awk '{print $NF}' | head -1)

    # Build the phy yaml copy command locally to avoid single quotes inside the
    # ssh_gnb string (single quotes break the docker exec bash -c '...' wrapper).
    local phy_yaml_cmd=""
    [ -n "${phy_yaml}" ] && phy_yaml_cmd="cp ${phy_yaml} ${log_dir}/configs/ 2>/dev/null || true;"

    # gNB container: the specific cuphycontroller yaml + testMAC + cuMAC-CP main configs
    ssh_gnb "mkdir -p ${log_dir}/configs && \
             ${phy_yaml_cmd} \
             cp ${CUBB_SDK}/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml   ${log_dir}/configs/ 2>/dev/null || true; \
             cp ${CUBB_SDK}/cuPHY-CP/testMAC/testMAC/test_cumac_config.yaml ${log_dir}/configs/ 2>/dev/null || true; \
             cp ${CUBB_SDK}/cuMAC-CP/config/cumac_cp.yaml                   ${log_dir}/configs/ 2>/dev/null || true; \
             cp ${CUBB_SDK}/cuPHY/nvlog/config/nvlog_config.yaml            ${log_dir}/configs/ 2>/dev/null || true" \
        2>/dev/null || true

    # RU container: ru-emulator main yaml only
    ssh_ru "cp ${CUBB_SDK}/cuPHY-CP/ru-emulator/config/ru_emulator_config.yaml \
               ${log_dir}/configs/ 2>/dev/null || true" 2>/dev/null || true

    # System info from gNB container
    {
        ssh_gnb "uname -a 2>/dev/null; \
                 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null; \
                 cat /proc/driver/nvidia/version 2>/dev/null; \
                 nvcc --version 2>/dev/null" 2>/dev/null || true
    } > "${cfg_dir}/sysinfo.txt" 2>&1 || true

    log_info "Config dump complete: ${cfg_dir}"
}

# ── Kill all test processes (local + remote) ──────────────────────────────────
# Build remote pgrep/pkill command lines with bracketed patterns expanded HERE
# (on the top-script host), e.g. pgrep -af "[c]uphycontroller_scf".
#
# Do NOT emit `for p in cuphycontroller_scf ...; do pgrep ...` on the remote
# side: that puts the unbracketed name into the bash -c argv, so
# pgrep -af "[c]uphycontroller_scf" matches the checker shell itself and
# _verify_killed always reports "still alive".
_bracket_pat() {
    # $1 = process name → "[c]uphycontroller_scf"
    local p="$1"
    echo "[${p:0:1}]${p:1}"
}

_alive_procs_cmd() {
    # $1 = space-separated process name patterns (matched via pgrep -af)
    local patterns="$1" p cmd=""
    for p in ${patterns}; do
        cmd="${cmd}pgrep -af \"$(_bracket_pat "${p}")\" 2>/dev/null || true; "
    done
    echo "${cmd}"
}

_kill_procs_cmd() {
    local patterns="$1" p cmd=""
    for p in ${patterns}; do
        cmd="${cmd}sudo pkill -9 -f \"$(_bracket_pat "${p}")\" 2>/dev/null || true; "
    done
    echo "${cmd}"
}

_signal_procs_cmd() {
    local patterns="$1" signal="$2" p cmd=""
    for p in ${patterns}; do
        cmd="${cmd}sudo pkill -${signal} -f \"$(_bracket_pat "${p}")\" 2>/dev/null || true; "
    done
    echo "${cmd}"
}

stop_all_gracefully() {
    log_info "Stopping test applications with SIGINT so runtime statistics are flushed ..."
    ssh_gnb "$(_signal_procs_cmd "${gnb_procs}" INT)" 2>/dev/null || true
    ssh_ru  "$(_signal_procs_cmd "${ru_procs}" INT)"  2>/dev/null || true
    sleep 10
    kill_all
}

# Verify target processes are gone on a host; retry kill once if any remain.
#   $1 = ssh helper (ssh_gnb | ssh_ru)
#   $2 = host label for logs
#   $3 = space-separated process name patterns
_verify_killed() {
    local ssh_fn="$1"; local label="$2"; local patterns="$3"
    local alive retry_alive
    local check_cmd kill_cmd
    check_cmd=$(_alive_procs_cmd "${patterns}")
    kill_cmd=$(_kill_procs_cmd "${patterns}")

    alive=$($ssh_fn "${check_cmd}" 2>/dev/null | grep -v '^\[' | tr -d '\r' | sed '/^$/d' || true)
    if [ -z "${alive}" ]; then
        log_check "${label}: all target processes killed."
        return 0
    fi

    log_info "${label}: processes still alive after kill — retrying:"
    echo "${alive}" | while IFS= read -r line; do log_info "  still: ${line}"; done
    $ssh_fn "${kill_cmd}" 2>/dev/null || true
    sleep 2

    retry_alive=$($ssh_fn "${check_cmd}" 2>/dev/null | grep -v '^\[' | tr -d '\r' | sed '/^$/d' || true)
    if [ -z "${retry_alive}" ]; then
        log_check "${label}: all target processes killed after retry."
        return 0
    fi

    log_err "${label}: processes STILL alive after kill retry:"
    echo "${retry_alive}" | while IFS= read -r line; do log_err "  alive: ${line}"; done
    return 1
}

kill_all() {
    log_info "Killing test processes on gNB and RU ..."
    # Use pkill -9 -f (matches full cmdline) so long binary names like
    # cuphycontroller_scf are not silently missed due to the 15-char
    # /proc/pid/comm truncation that makes killall fail.
    local gnb_cleanup_procs="${gnb_procs} nvidia-cuda-mps-server nvidia-cuda-mps-control"

    ssh_gnb "$(_kill_procs_cmd "${gnb_cleanup_procs}")" 2>/dev/null || true
    ssh_ru  "$(_kill_procs_cmd "${ru_procs}")"  2>/dev/null || true

    # Kill any local background jobs from this script
    jobs -p | xargs -r kill 2>/dev/null || true
    sleep 2

    local kill_ok=0
    _verify_killed ssh_gnb "gNB" "${gnb_cleanup_procs}" || kill_ok=1
    _verify_killed ssh_ru  "RU"  "${ru_procs}"  || kill_ok=1
    if [ "${kill_ok}" -eq 0 ]; then
        log_info "kill_all: all target processes confirmed dead."
    else
        log_err "kill_all: some processes survived — continuing, but next step may be unstable."
    fi
    return "${kill_ok}"
}

# ── Stop NVIDIA MPS on gNB container ──────────────────────────────────────────
# Force-kill any running MPS daemon before launching apps each step.
# run2_cuPHYcontroller.sh does its own restart_mps() internally; we just need to
# ensure no stale daemon is alive when it tries, regardless of which
# CUDA_MPS_PIPE_DIRECTORY it was originally started with.
stop_mps_gnb() {
    log_step "Stopping NVIDIA MPS on gNB container ..."
    ssh_gnb "sudo killall -9 nvidia-cuda-mps-server nvidia-cuda-mps-control 2>/dev/null || true; \
             sleep 1" || true
}

# ── Generate test_params.sh via parse_test_config_params.sh ──────────────────
# Runs inside gNB container (where cuBB_SDK is set and parse_test_config_params.py
# has its preset files).  Writes test_params.sh to the NFS CUBB_HOST root so it
# is immediately visible on the top-script server and all other containers.
# After writing, sources it locally for the host-agnostic params (SETUP*, TEST_CONFIG,
# COPY_TEST_FILES), then re-resolves the arch-specific params (BUILD/RUN*) per host.
generate_test_params() {
    log_step "Generating test_params.sh via parse_test_config_params.sh ..."
    log_cmd  "parse_test_config_params.sh \"${test_case_string}\" \"${HOST_CONFIG}\" ${TEST_PARAMS_SDK}"

    ssh_gnb "cd ${CUBB_SDK} && \
        ${phase4}/parse_test_config_params.sh \
        \"${test_case_string}\" \"${HOST_CONFIG}\" \"${TEST_PARAMS_SDK}\"" \
        || { log_err "parse_test_config_params.sh failed"; return 1; }

    log_info "Sourcing ${TEST_PARAMS_HOST} ..."
    # shellcheck disable=SC1090
    source "${TEST_PARAMS_HOST}" \
        || { log_err "Failed to source ${TEST_PARAMS_HOST}"; return 1; }

    log_info "test_params.sh loaded."
    log_info "  COPY_TEST_FILES_PARAMS = ${COPY_TEST_FILES_PARAMS:-<empty>}"
    log_info "  SETUP1_DU_PARAMS   = ${SETUP1_DU_PARAMS:-<empty>}"
    log_info "  SETUP2_RU_PARAMS   = ${SETUP2_RU_PARAMS:-<empty>}"
    log_info "  TEST_CONFIG_PARAMS = ${TEST_CONFIG_PARAMS:-<empty>}"
    # Re-resolve the per-host params to the values computed ON their executing host
    # and store them back into globals, so the steps can use them directly — no
    # per-step re-source over SSH, no escaping.  BUILD_AERIAL_PARAMS is built on
    # BOTH hosts (separate arch builds), so it needs one global per host; each RUN
    # param is used on a single host (run1→RU, run2/run3→gNB).
    BUILD_AERIAL_PARAMS_RU=$(_resolve_param_on_host ssh_ru  BUILD_AERIAL_PARAMS)
    BUILD_AERIAL_PARAMS_GNB=$(_resolve_param_on_host ssh_gnb BUILD_AERIAL_PARAMS)
    RUN1_RU_PARAMS=$(_resolve_param_on_host ssh_ru  RUN1_RU_PARAMS)
    RUN2_CUPHYCONTROLLER_PARAMS=$(_resolve_param_on_host ssh_gnb RUN2_CUPHYCONTROLLER_PARAMS)
    RUN3_TESTMAC_PARAMS=$(_resolve_param_on_host ssh_gnb RUN3_TESTMAC_PARAMS)
    log_info "  BUILD_AERIAL_PARAMS = ${BUILD_AERIAL_PARAMS_RU:-<empty>} (RU) | ${BUILD_AERIAL_PARAMS_GNB:-<empty>} (gNB)"
    log_info "  RUN1_RU_PARAMS     = ${RUN1_RU_PARAMS:-<empty>}  (resolved on RU)"
    log_info "  RUN2_CUPHYCONTROLLER_PARAMS = ${RUN2_CUPHYCONTROLLER_PARAMS:-<empty>}  (resolved on gNB)"
    log_info "  RUN3_TESTMAC_PARAMS = ${RUN3_TESTMAC_PARAMS:-<empty>}  (resolved on gNB)"

    local cumac_bdir; cumac_bdir=$(detect_cumac_build_dir)
    log_info "  cumac_bdir         = ${cumac_bdir}  (from RUN2_CUPHYCONTROLLER_PARAMS)"
    return 0
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 1 — Run cuBB test and dump SRS TV
# ────────────────────────────────────────────────────────────────────────────
run_cubb() {
    local step_start; step_start=$(date +%s)
    local step_label
    if [ "${DUMP_SRS_SLOT_NUM:-0}" -gt 0 ]; then
        step_label="cuBB Test + SRS TV Dump"
    else
        step_label="cuBB Test"
    fi
    log_step "====== ${step_label} START ======"
    log_info  "test_case_string=$test_case_string  duration=$duration"
    log_info  "DUMP_SRS_SLOT_NUM=$DUMP_SRS_SLOT_NUM"

    local step1_log
    if [ "$RUN_TEST" = "cubb" ]; then
        # A plain cuBB run has only one step, so keep its logs directly in the
        # run directory that will be renamed to log_dir_final.
        step1_log="$LOG_RUN_DIR"
    elif [ "${DUMP_SRS_SLOT_NUM:-0}" -gt 0 ]; then
        step1_log="$LOG_RUN_DIR/cubb_srs_dump"
        ssh_gnb "sudo rm -rf /tmp/cubb_srs_buffers_*.h5" || true
    else
        step1_log="$LOG_RUN_DIR/cubb"
    fi
    mkdir -p "$step1_log"
    export LOG_PATH="$step1_log"

    local _lp_file="${CUBB_HOST}/testVectors/multi-cell/launch_pattern_F08_${cell_num}C_${pattern}.yaml"
    if [ -z "${TV_SRC:-}" ] && [ -f "${_lp_file}" ]; then
        log_info "Launch pattern ${_lp_file} already present — skipping copy_test_files.sh"
    else
        # Copy test vectors (needed before build so test vector paths are available).
        # In QA each container has its own filesystem — copy on both gNB and RU.
        local _copy_tv_cmd="$phase4/copy_test_files.sh ${COPY_TEST_FILES_PARAMS}"
        [ -n "$TV_SRC" ] && _copy_tv_cmd="${_copy_tv_cmd} --src ${TV_SRC}"
        log_info "Copying test vectors ..."
        ssh_gnb "$_copy_tv_cmd" \
            || { log_err "copy_test_files.sh (gNB) failed"; return 1; }
        if [ "${gNB_RU_Share:-0}" != "1" ]; then
            ssh_ru "$_copy_tv_cmd" \
                || { log_err "copy_test_files.sh (RU) failed"; return 1; }
        fi
    fi

    build_if_needed || return 1
    # Restore baseline config files before applying test-specific setup.
    # Each test case may have different parameters so we start from config_ori
    # every run (mirrors Step 4 in cubb_phase4_config.sh).
    restore_config_files || return 1

    # Setup steps (synchronous).
    # setup1_DU.sh runs on the DU/gNB side; setup2_RU.sh MUST run on the RU side
    # so it reads the correct NIC MAC addresses from /sys/class/net/.
    # When gNB_RU_Share != 1 (gNB and RU do NOT share the NFS): setup scripts live inside containers (not NFS),
    # and setup1_DU.sh writes test_config_summary.sh into the gNB container.
    # Exchange it via NFS before setup2_RU.sh, and sync back to gNB afterward.
    log_info "Running setup scripts ..."
    ssh_gnb "$phase4/setup1_DU.sh ${SETUP1_DU_PARAMS}" \
        || { log_err "setup1_DU.sh failed"; return 1; }
    if [ "${gNB_RU_Share:-0}" != "1" ]; then
        _copy_du_artifacts_to_nfs  || return 1
        _copy_nfs_artifacts_to_ru  || return 1
    fi
    ssh_ru  "$phase4/setup2_RU.sh ${SETUP2_RU_PARAMS}" \
        || { log_err "setup2_RU.sh failed"; return 1; }
    if [ "${gNB_RU_Share:-0}" != "1" ]; then
        _sync_ru_artifacts_to_gnb  || return 1
    fi
    ssh_gnb "$phase4/test_config.sh ${TEST_CONFIG_PARAMS}" \
        || { log_err "test_config.sh (gNB) failed"; return 1; }
    if [ "${gNB_RU_Share:-0}" != "1" ]; then
        ssh_ru "$phase4/test_config.sh ${TEST_CONFIG_PARAMS}" \
            || { log_err "test_config.sh (RU) failed"; return 1; }
    fi
    configure_nvipc_pcap || return 1
    set_log_config || return 1

    # Kill any leftover processes from a previous run before launching.
    kill_all
    # Stop MPS so run2_cuPHYcontroller.sh can restart it cleanly.
    stop_mps_gnb

    # Launch apps in background; wait for each to be ready before starting the next.
    # Order: RU → PHY → MAC  (mirrors cubb_phase4_execute.sh wait_process pattern).
    log_info "Launching RU ..."
    ssh_ru_bg  "ru"  "$phase4/run1_RU.sh ${RUN1_RU_PARAMS}"
    wait_for_log_pattern "${step1_log}/screenlog_ru.log" \
        "Cell.*Seconds" "$RU_READY_TIMEOUT" "RU" \
        || { kill_all; return 1; }

    log_info "Launching cuPHYcontroller ..."
    ssh_gnb_bg "phy" "export DUMP_SRS_SLOT_NUM=${DUMP_SRS_SLOT_NUM} && \
                export LOG_PATH=${step1_log} && \
                $phase4/run2_cuPHYcontroller.sh ${RUN2_CUPHYCONTROLLER_PARAMS}"
    wait_for_log_pattern "${step1_log}/screenlog_phy.log" \
        "cuPHYController initialized, L1 is ready" "$PHY_READY_TIMEOUT" "cuPHYcontroller" \
        || { kill_all; return 1; }

    log_info "Launching testMAC ..."
    ssh_gnb_bg "mac" "export LOG_PATH=${step1_log} && \
                $phase4/run3_testMAC.sh ${RUN3_TESTMAC_PARAMS}"
    wait_for_log_pattern "${step1_log}/screenlog_mac.log" \
        "scheduler_thread_func" "$MAC_READY_TIMEOUT" "testMAC" \
        || { kill_all; return 1; }
    collect_core_snapshots "$step1_log"

    # Launch result checker locally (reads logs via NFS)
    log_info "Starting check_result_cumcp.py (Step 1: MAC + RU logs, no cuMAC-CP) ..."
    export LOG_PATH="$step1_log"
    python3 "$SCRIPT_DIR/check_result_cumcp.py" "$duration" \
        --ru-log "${step1_log}/screenlog_ru.log" \
        --mac-log "${step1_log}/screenlog_mac.log" &
    local checker_pid=$!
    log_info "check_result_cumcp.py pid=$checker_pid"

    # Monitor screens in background — kill checker if any process exits early.
    monitor_screens "$step1_log" "ru phy mac" "$checker_pid" &
    local monitor_pid=$!

    # Trap Ctrl+C
    trap "log_err 'Interrupted. Killing checker and processes.'; \
          kill -9 $checker_pid 2>/dev/null; kill $monitor_pid 2>/dev/null; kill_all; exit 1" SIGINT

    # Wait for checker
    wait $checker_pid
    local test_result=$?
    kill "$monitor_pid" 2>/dev/null
    wait "$monitor_pid" 2>/dev/null

    # Restore the global SIGINT handler (trap - would reset to default, losing coverage).
    trap 'log_err "Interrupted — running kill_all cleanup."; kill_all; exit 1' SIGINT

    # Stop applications gracefully before force cleanup so statistics files flush.
    stop_all_gracefully
    collect_runtime_logs "$step1_log"
    dump_configs "$step1_log"

    local elapsed=$(( $(date +%s) - step_start ))
    if [ $test_result -ne 0 ]; then
        log_step "====== ${step_label} END: FAILED (elapsed ${elapsed}s, exit_code=$test_result) ======"
        return $test_result
    fi

    # Verify SRS TV was dumped (only when dump was enabled)
    if [ "${DUMP_SRS_SLOT_NUM:-0}" -gt 0 ]; then
        check_srs_tv || return 1
    fi

    log_step "====== ${step_label} END: PASSED (elapsed ${elapsed}s) ======"
    return 0
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 2 — Generate cuMAC TV via run_cumcp_sa.sh (inside gNB container)
# ────────────────────────────────────────────────────────────────────────────
run_cumcp_sa() {
    local step_start; step_start=$(date +%s)
    local step_label="cuMAC-CP Standalone TV Generation"
    log_step "====== ${step_label} START ======"

    # Prerequisite: SRS TV must exist
    # SRS TV verification is only needed for gpu_share=2 (other modes don't use SRS TV).
    if [ "${gpu_share:-0}" -eq 2 ]; then
        check_srs_tv || return 1
    else
        log_info "SRS TV not needed for gpu_share=${gpu_share:-0}"
    fi
    local step2_log="$LOG_RUN_DIR/cumcp_sa"
    local step2_log_sdk
    if [[ "${step2_log}" == "${CUBB_HOST}/"* ]]; then
        step2_log_sdk="${CUBB_SDK}${step2_log#${CUBB_HOST}}"
    else
        log_err "cuMAC-CP SA log path must be under CUBB_HOST: ${step2_log}"
        return 1
    fi
    mkdir -p "$step2_log"
    local cumac_bdir; cumac_bdir=$(detect_cumac_build_dir)
    log_info "cuMAC-CP standalone build dir: ${cumac_bdir}"
    restore_config_files || return 1
    # Apply the TDD pattern after restore_config_files (which resets muUeGrp config.yaml to baseline) and before run_cumcp_sa.sh, so the pattern is live during TV generation. No-op when unset.
    if [[ -n "${TDD_PATTERN:-}" ]]; then
        local muUeGrp_cfg="${CUBB_SDK}/cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/config.yaml"
        log_cfg "yq: TDD_PATTERN = ${TDD_PATTERN}  →  ${muUeGrp_cfg}"
        ssh_gnb "yq -i \".TDD_PATTERN = \\\"${TDD_PATTERN}\\\"\" ${muUeGrp_cfg} && yq .TDD_PATTERN ${muUeGrp_cfg}" \
            || { log_err "Failed to set TDD_PATTERN=${TDD_PATTERN} in muUeGrp config"; return 1; }
    fi
    local run_cumcp_cmd="export LOG_PATH=${step2_log_sdk} && export BUILD=${cumac_bdir} && \
        $cumcp_scripts_sdk/run_cumcp_sa.sh $duration $test_group $test_case --alloc_type $alloc_type --gpu_share $gpu_share --task $CUMAC_TASK_MASK"

    if [ "${FORCE_RENEW_CUMAC_CP_TV}" = "1" ]; then
        run_cumcp_cmd="${run_cumcp_cmd} -r"
    fi

    # If the cumac TV symlink exists but points to the wrong target, remove it so
    # run_cumcp_sa.sh can create the correct one for this run's parameters.
    log_check "Checking cuMAC TV symlink before TV generation (expected: ${cumac_tv_dir}) ..."
    ssh_gnb "if [ -L ${cumac_tv_link} ]; then \
        target=\$(readlink -f ${cumac_tv_link} 2>/dev/null); \
        if [ \"\$target\" = \"${cumac_tv_dir}\" ]; then \
            echo \"cuMAC TV symlink already correct: \${target}\"; \
        else \
            echo \"cuMAC TV symlink mismatch (\${target} != ${cumac_tv_dir}), removing ...\"; \
            rm -f ${cumac_tv_link}; \
        fi; \
    else \
        echo \"cuMAC TV symlink absent — run_cumcp_sa.sh will create it\"; \
    fi" || true

    # Stop all processes
    kill_all
    log_info "Executing run_cumcp_sa.sh inside gNB container ..."
    log_cmd "run_cumcp_sa.sh: $run_cumcp_cmd"
    ssh_gnb "$run_cumcp_cmd"
    local rc=$?

    local elapsed=$(( $(date +%s) - step_start ))
    if [ $rc -ne 0 ]; then
        log_step "====== ${step_label} END: FAILED (elapsed ${elapsed}s, exit_code=$rc) ======"
        return $rc
    fi
    check_cumac_tv || return 1
    dump_configs "$step2_log"

    log_step "====== ${step_label} END: PASSED (elapsed ${elapsed}s) ======"
    return 0
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run cuBB + cuMAC-CP combined 4-app test
# ────────────────────────────────────────────────────────────────────────────
run_cubb_cumcp() {
    local step_start; step_start=$(date +%s)
    local step_label="cuBB + cuMAC-CP Combined Test"
    log_step "====== ${step_label} START ======"
    log_info "test_case_string=$test_case_string  duration=$duration"
    log_info "cell_num=$cell_num alloc_type=$alloc_type gpu_share=$gpu_share"
    log_info "CUMAC_TASK_MASK=$CUMAC_TASK_MASK"

    # No cuMAC TV prerequisite check here: run_cuMAC_CP.sh (Step 3) regenerates the
    # cuMAC TV via cumac_cp_tv.sh if it is missing, so the combined test is safe to
    # launch without a pre-existing symlink.

    local step3_log="$LOG_RUN_DIR/cubb_cumcp"
    mkdir -p "$step3_log"
    export LOG_PATH="$step3_log"

    # ── Restore baseline configs and re-run setup (same as Step 1) ───────────
    # Step 2 (run_cumcp_sa.sh) modifies YAML state; restore before applying
    # the combined-test configuration so cuBB sees the same setup as Step 1.
    build_if_needed || return 1
    restore_config_files || return 1

    log_info "Running setup scripts ..."
    ssh_gnb "$phase4/setup1_DU.sh ${SETUP1_DU_PARAMS}" \
        || { log_err "setup1_DU.sh failed"; return 1; }
    if [ "${gNB_RU_Share:-0}" != "1" ]; then
        _copy_du_artifacts_to_nfs  || return 1
        _copy_nfs_artifacts_to_ru  || return 1
    fi
    ssh_ru  "$phase4/setup2_RU.sh ${SETUP2_RU_PARAMS}" \
        || { log_err "setup2_RU.sh failed"; return 1; }
    if [ "${gNB_RU_Share:-0}" != "1" ]; then
        _sync_ru_artifacts_to_gnb  || return 1
    fi
    ssh_gnb "$phase4/test_config.sh ${TEST_CONFIG_PARAMS}" \
        || { log_err "test_config.sh (gNB) failed"; return 1; }
    if [ "${gNB_RU_Share:-0}" != "1" ]; then
        ssh_ru "$phase4/test_config.sh ${TEST_CONFIG_PARAMS}" \
            || { log_err "test_config.sh (RU) failed"; return 1; }
    fi

    # ── cuMAC-CP yaml configuration for combined test ─────────────────────────
    # run_cumcp_sa.sh (Step 2) leaves enable_cubb=0 and srs_slot_lag=0.
    # Apply cuMAC-CP-specific settings on top of the restored baseline.
    # YAML paths use CUBB_SDK; edits run via SSH inside the gNB container.
    # (gNB_RU_Share=1 shares the NFS, but YAML edits still target the container SDK.)
    local cumac_cp_yaml="$CUBB_SDK/cuMAC-CP/config/cumac_cp.yaml"
    local test_cumac_yaml="$CUBB_SDK/cuPHY-CP/testMAC/testMAC/test_cumac_config.yaml"
    # Derive the cuphycontroller yaml from the DU side of HOST_CONFIG (CG1/GL4/SPRK/…)
    # so the right file is edited on non-CG1 platforms, not a hardcoded CG1 one.
    local du_platform; du_platform=$(echo "$HOST_CONFIG" | cut -d_ -f1)
    local f08_yaml="$CUBB_SDK/cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_${du_platform}.yaml"
    if (( CUMAC_TASK_MASK == 0x20 )); then
        log_step "Configuring cuMAC-CP yamls (UE Group: enable_cubb=1, srs_slot_lag=${SRS_SLOT_LAG}) ..."
        cfg_set "enable_tv_test"   "1"             "$cumac_cp_yaml"   || return 1
        cfg_set "enable_cubb"      "1"             "$cumac_cp_yaml"   || return 1
        cfg_set "srs_slot_lag"     "$SRS_SLOT_LAG" "$cumac_cp_yaml"   || return 1
        cfg_set "srs_slot_lag"     "$SRS_SLOT_LAG" "$test_cumac_yaml" || return 1
        cfg_set "sendCPlane_dlbfw_backoff_th_ns"        "0"      "$f08_yaml"   || return 1
    else
        log_step "Configuring cuMAC-CP yamls (Other Tasks: enable_cubb=0) ..."
        cfg_set "enable_cubb"      "0"             "$cumac_cp_yaml"   || return 1
    fi
    cfg_set "cumac_cp_standalone"      "0"        "$test_cumac_yaml"  || return 1
    configure_nvipc_pcap || return 1
    set_log_config || return 1


    # Kill any leftover processes from a previous run before launching.
    kill_all
    # Stop MPS so run2_cuPHYcontroller.sh can restart it cleanly.
    stop_mps_gnb

    # Launch apps in order: cuMAC-CP → RU → PHY → MAC.
    # Each waits for a ready signal before the next is launched.
    # Mirrors cubb_phase4_execute.sh wait_process pattern for cuMAC=1.

    log_info "Launching RU ..."
    ssh_ru_bg "ru" "$phase4/run1_RU.sh ${RUN1_RU_PARAMS}"
    wait_for_log_pattern "${step3_log}/screenlog_ru.log" \
        "Cell.*Seconds" "$RU_READY_TIMEOUT" "RU" \
        || { kill_all; return 1; }

    log_info "Launching cuPHYcontroller ..."
    ssh_gnb_bg "phy" "export LOG_PATH=${step3_log} && \
                $phase4/run2_cuPHYcontroller.sh ${RUN2_CUPHYCONTROLLER_PARAMS}"
    wait_for_log_pattern "${step3_log}/screenlog_phy.log" \
        "cuPHYController initialized, L1 is ready" "$PHY_READY_TIMEOUT" "cuPHYcontroller" \
        || { kill_all; return 1; }

    log_info "Launching cuMAC-CP ..."
    local cumac_bdir; cumac_bdir=$(detect_cumac_build_dir)
    # Only pass -b when we resolved a build dir; otherwise omit it so run_cuMAC_CP.sh
    # uses its own build.$(uname -m) default ON THE gNB (correct arch).  Passing an
    # empty "-b" would let the next flag (-c) be swallowed as its value.
    local _bdir_flag=""
    [ -n "${cumac_bdir}" ] && _bdir_flag="-b ${cumac_bdir}"
    log_info "cuMAC-CP build dir: ${cumac_bdir:-<run_cuMAC_CP.sh default>}"
    ssh_gnb_bg "cum" "export LOG_PATH=${step3_log} && \
                $phase4/run_cuMAC_CP.sh ${_bdir_flag} -c $cell_num -a $alloc_type -g $gpu_share -m $CUMAC_TASK_MASK"
    wait_for_log_pattern "${step3_log}/screenlog_cum.log" \
        "cumac_receiver: initialized" "$CUMAC_READY_TIMEOUT" "cuMAC-CP" \
        || { kill_all; return 1; }

    log_info "Launching testMAC ..."
    ssh_gnb_bg "mac" "export LOG_PATH=${step3_log} && \
                $phase4/run3_testMAC.sh ${RUN3_TESTMAC_PARAMS}"
    wait_for_log_pattern "${step3_log}/screenlog_mac.log" \
        "scheduler_thread_func" "$MAC_READY_TIMEOUT" "testMAC" \
        || { kill_all; return 1; }
    collect_core_snapshots "$step3_log"

    # Launch result checker locally (monitors testMAC + RU + cuMAC-CP logs via NFS)
    log_info "Starting check_result_cumcp.py (Step 3: MAC + RU + cuMAC-CP logs) ..."
    python3 "$SCRIPT_DIR/check_result_cumcp.py" "$duration" \
        --ru-log "${step3_log}/screenlog_ru.log" \
        --cumcp-log "${step3_log}/screenlog_cum.log" \
        --mac-log "${step3_log}/screenlog_mac.log" &
    local checker_pid=$!
    log_info "check_result_cumcp.py pid=$checker_pid"

    # Monitor screens in background — kill checker if any process exits early.
    monitor_screens "$step3_log" "cum ru phy mac" "$checker_pid" &
    local monitor_pid=$!

    trap "log_err 'Interrupted. Killing checker and processes.'; \
          kill -9 $checker_pid 2>/dev/null; kill $monitor_pid 2>/dev/null; kill_all; exit 1" SIGINT

    # Wait for checker
    wait $checker_pid
    local test_result=$?
    kill "$monitor_pid" 2>/dev/null
    wait "$monitor_pid" 2>/dev/null

    # Restore the global SIGINT handler (trap - would reset to default, losing coverage).
    trap 'log_err "Interrupted — running kill_all cleanup."; kill_all; exit 1' SIGINT

    # Stop applications gracefully before force cleanup so statistics files flush.
    stop_all_gracefully
    collect_runtime_logs "$step3_log"
    dump_configs "$step3_log"

    local elapsed=$(( $(date +%s) - step_start ))
    if [ $test_result -ne 0 ]; then
        log_step "====== ${step_label} END: FAILED (elapsed ${elapsed}s, exit_code=$test_result) ======"
        return $test_result
    fi

    log_step "====== ${step_label} END: PASSED (elapsed ${elapsed}s) ======"
    return 0
}

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
script_start=$(date +%s)

# Top-level SIGINT safety net: ensure remote processes are cleaned up if Ctrl+C
# arrives between steps (when no per-function trap is active).
trap 'log_err "Interrupted — running kill_all cleanup."; kill_all; exit 1' SIGINT

log_info "========== run_cumcp_cubb_test.sh =========="
log_info "test_case_string=$test_case_string  (group=$test_group  cells=$cell_num  pattern=$pattern)"
log_info "duration=$duration  alloc_type=$alloc_type  gpu_share=$gpu_share"
log_info "GNB_SERVER=$GNB_SERVER  RU_SERVER=$RU_SERVER  HOST_CONFIG=$HOST_CONFIG"
log_info "CONTAINER=$CONTAINER  CUBB_HOST=$CUBB_HOST  CUBB_SDK=$CUBB_SDK"
log_info "DUMP_SRS_SLOT_NUM=$DUMP_SRS_SLOT_NUM  CUMAC_TASK_MASK=$CUMAC_TASK_MASK  SRS_SLOT_LAG=$SRS_SLOT_LAG  TV_SRC=${TV_SRC:-<auto>}  TDD_PATTERN=${TDD_PATTERN}"
log_info "SHM_LEVEL=${SHM_LEVEL:-<yaml-default>}  LOG_SIZE=${LOG_SIZE:-<yaml-default>}  NVIPC_PCAP_ENABLE=${NVIPC_PCAP_ENABLE:-0}  TRY=${TRY:-0}"
log_info "LOG_BASE=${LOG_BASE:-${CUBB_HOST}/logs}  LOG_RUN_DIR=$LOG_RUN_DIR"
log_info "Test mode: $RUN_TEST"
log_info "============================================"

# Generate test_params.sh and source it before any step runs.
# This populates $SETUP1_DU_PARAMS, $RUN1_RU_PARAMS, etc. used by all steps.
generate_test_params || exit 1

overall_result=0

case "$RUN_TEST" in
    cubb)
        # Normal cuBB Test — no SRS TV dump
        DUMP_SRS_SLOT_NUM=0
        CASE_TYPE="CUBB"
        run_cubb || overall_result=$?
        ;;
    cubb-srs)
        # cuBB Test with SRS TV Dump
        CASE_TYPE="CUBB_SRS_DUMP"
        run_cubb || overall_result=$?
        ;;
    cumcp-sa)
        # cuMAC-CP Standalone TV Generation
        CASE_TYPE="CUMCP_SA"
        run_cumcp_sa || overall_result=$?
        ;;
    ue-group)
        # cuBB + cuMAC-CP UE Group: [cuBB+SRS] → cuMAC-CP SA → cuBB+cuMAC-CP Combined
        # Step 1 (SRS dump) only runs when gpu_share=2
        CASE_TYPE="CUBB_CUMCP_UE_GROUP"
        if [ "${gpu_share:-0}" -eq 2 ]; then
            run_cubb || { overall_result=$?; log_err "cuBB Test + SRS TV Dump FAILED (exit $overall_result)"; }
        else
            log_info "Skipping cuBB Test + SRS TV Dump: only required for gpu_share=2 (current: ${gpu_share:-0})"
        fi
        if [ $overall_result -eq 0 ]; then
            run_cumcp_sa || { overall_result=$?; log_err "cuMAC-CP Standalone TV Generation FAILED (exit $overall_result)"; }
        fi
        if [ $overall_result -eq 0 ]; then
            run_cubb_cumcp || overall_result=$?
        fi
        ;;
    other)
        # cuBB + cuMAC-CP Other Tasks: Combined Test only (enable_cubb=0 applied in run_cubb_cumcp)
        CASE_TYPE="CUBB_CUMCP"
        run_cubb_cumcp || overall_result=$?
        ;;
    *)
        echo "ERROR: Unknown --test value: '$RUN_TEST'"
        echo "       Valid modes: cubb  cubb-srs  cumcp-sa  ue-group  other"
        exit 1
        ;;
esac

# ── Final log archival ────────────────────────────────────────────────────────
total_elapsed=$(( $(date +%s) - script_start ))
log_info "============================================"
log_info "Total elapsed: ${total_elapsed}s"

# Analyze checker output and screenlogs. Done before the mv so LOG_RUN_DIR exists.
# Classification priority: CRASH > FAIL > ERR > PASS.
# A crash marker is written only when an app exits while the checker is still
# active, so the expected SIGKILL exit codes emitted by cleanup are not crashes.
crash_hits=$(find "$LOG_RUN_DIR" -name '.crash_detected' -print -quit 2>/dev/null)
fail_hits=$(grep -rE "Test (FAIL|FAILED)|END: FAILED" \
    --include="check_result.log" "$LOG_RUN_DIR" 2>/dev/null | head -5)
pass_hits=$(grep -rE "Test PASS" \
    --include="check_result.log" "$LOG_RUN_DIR" 2>/dev/null | head -5)

# Scan screenlogs for ERR-level lines.
#   - All screenlogs EXCEPT phy: any ERR line is a failure.
#   - screenlog_phy.log: the PHY emits known-benign WRN/ERR during startup (e.g.
#     "Task aborted for Slot Map") and the first stats ticks read "DL 0.00 Mbps".
#     Only start counting ERR lines once a NON-ZERO "DL <x> Mbps" appears, i.e.
#     traffic is actually flowing.
err_hits=$(grep -rE "[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6} ERR" \
    --include="screenlog_*.log" --exclude="screenlog_phy.log" "$LOG_RUN_DIR" 2>/dev/null)
phy_err_hits=$(
    find "$LOG_RUN_DIR" -name 'screenlog_phy.log' 2>/dev/null | while IFS= read -r logf; do
        awk 'seen && /[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]+ ERR / { print FILENAME": "$0 }
             !seen && match($0, /DL[[:space:]]+[0-9.]+[[:space:]]+Mbps/) {
                 dl = substr($0, RSTART, RLENGTH); gsub(/[^0-9.]/, "", dl)
                 if (dl + 0 > 0) seen = 1
             }' "$logf"
    done
)
err_hits=$(printf '%s\n%s\n' "$err_hits" "$phy_err_hits" | grep -v '^[[:space:]]*$' | head -5)
if [ "$RUN_TEST" = "cubb" ]; then
    log_dir_final="$_log_parent/${ts}_${test_case_string}_${CASE_TYPE}_${duration}s"
elif [[ -n "${TDD_PATTERN:-}" ]]; then
    log_dir_final="$_log_parent/${ts}_${test_case_string}_${CASE_TYPE}_TASK${CUMAC_TASK_MASK}_${TDD_PATTERN}_TYPE${alloc_type}_SHARE${gpu_share}_${duration}s"
else
    log_dir_final="$_log_parent/${ts}_${test_case_string}_${CASE_TYPE}_TASK${CUMAC_TASK_MASK}_TYPE${alloc_type}_SHARE${gpu_share}_${duration}s"
fi
if [ -n "$crash_hits" ]; then
    log_info "OVERALL: CRASH (application exited before result checking completed)"
    overall_result=3
    mv "$LOG_RUN_DIR" "${log_dir_final}_CRASH" 2>/dev/null || true
    ln -sfn "${log_dir_final}_CRASH" "$_log_parent/latest"
    echo "Logs: ${log_dir_final}_CRASH"
elif [ $overall_result -ne 0 ] || [ -n "$fail_hits" ] || [ -z "$pass_hits" ]; then
    log_info "OVERALL: FAILED"
    [ -n "$fail_hits" ] && log_info "$fail_hits"
    [ -z "$pass_hits" ] && log_info "No 'Test PASS' result found in check_result.log"
    overall_result=2
    mv "$LOG_RUN_DIR" "${log_dir_final}_FAIL" 2>/dev/null || true
    ln -sfn "${log_dir_final}_FAIL" "$_log_parent/latest"
    echo "Logs: ${log_dir_final}_FAIL"
elif [ -n "$err_hits" ]; then
    log_info "OVERALL: ERR (ERR lines found in screenlogs)"
    log_info "$err_hits"
    overall_result=1
    mv "$LOG_RUN_DIR" "${log_dir_final}_ERR" 2>/dev/null || true
    ln -sfn "${log_dir_final}_ERR" "$_log_parent/latest"
    echo "Logs: ${log_dir_final}_ERR"
else
    log_info "OVERALL: PASSED"
    mv "$LOG_RUN_DIR" "${log_dir_final}_PASS" 2>/dev/null || true
    ln -sfn "${log_dir_final}_PASS" "$_log_parent/latest"
    echo "Logs: ${log_dir_final}_PASS"
fi

exit $overall_result
