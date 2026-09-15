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

# cuMAC-CP + cuBB Test Script — Environment Configuration
# Source this file from run_cumcp_cubb_test.sh (do not execute directly).

# ── Server hostnames / IPs ──────────────────────────────────────────────────
# GNB_SERVER: hostname or IP of the gNB/DU server
export GNB_SERVER="${GNB_SERVER:-}"

# RU_SERVER: hostname or IP of the RU server
# For QA the top-script server is the RU server, so this may equal $(hostname)
export RU_SERVER="${RU_SERVER:-}"

# ── SSH credentials ─────────────────────────────────────────────────────────
# Shared defaults — applied to both servers when per-server overrides are not set.
# When SSH key auth is configured, leave SSH_PASS unset.
export SSH_USER="${SSH_USER:-${USER}}"
export SSH_PASS="${SSH_PASS:-}"

# Per-server overrides — set when gNB and RU use different credentials.
# Leave unset to fall back to the shared SSH_USER / SSH_PASS above.
export GNB_SSH_USER="${GNB_SSH_USER:-${SSH_USER}}"
export GNB_SSH_PASS="${GNB_SSH_PASS:-${SSH_PASS}}"
export RU_SSH_USER="${RU_SSH_USER:-${SSH_USER}}"
export RU_SSH_PASS="${RU_SSH_PASS:-${SSH_PASS}}"

# ── Container naming ─────────────────────────────────────────────────────────
# Developer environment : c_aerial_${SSH_USER}
# QA environment        : cuBB_CICD
export CONTAINER="${CONTAINER:-c_aerial_${SSH_USER}}"

# ── SDK / NFS path ───────────────────────────────────────────────────────────
# CUBB_HOST: NFS mount path shared across the top-script server and all containers.
#   This path must be identical on every host (gNB, RU, top-script server)
#   and inside containers.  Used for data access: YAML configs, logs, test vectors.
#   Example (dev): /home/aerial/nfs/cubb/tot
#   Example (QA):  /home/aerial/nfs
export CUBB_HOST="${CUBB_HOST:-}"

# CUBB_SDK: value exported as $cuBB_SDK inside containers — the SDK install path.
#   Always set; defaults to /opt/nvidia/cuBB and is normally DIFFERENT from
#   CUBB_HOST (the NFS data path).  This variable is ONLY used inside container
#   SSH commands; all NFS file access (YAML edits, logs, test vectors) uses CUBB_HOST.
#   ENG (gNB_RU_Share=1): the NFS holds the full SDK, but cuBB_SDK is still the
#   in-container install path (default /opt/nvidia/cuBB), not CUBB_HOST.
export CUBB_SDK="${CUBB_SDK:-/opt/nvidia/cuBB}"

# gNB_RU_Share: set to 1 when the gNB and RU share the same NFS-mounted SDK (ENG
#   "share" setup).  The orchestrator then skips the redundant RU-side
#   copy_test_files / setup / test_config steps and the NFS artifact exchange.
#   Default 0 (gNB and RU are separate hosts): those RU-side steps run.
export gNB_RU_Share="${gNB_RU_Share:-0}"

# ── Host configuration ───────────────────────────────────────────────────────
# HOST_CONFIG: <DU_HOST>_<RU_HOST> platform string passed to parse_test_config_params.sh.
#   Valid combinations: CG1_R750  CG1_CG1  GL4_R750  SPRK_R750  (see phase4 README)
#   Example: SPRK_R750 (SPARK DU + x86 RU)
export HOST_CONFIG="${HOST_CONFIG:-}"

# SSH_LOG_EXEC: set to 1 to log the full expanded command being executed.
export SSH_LOG_EXEC="${SSH_LOG_EXEC:-0}"

# ── Validation ───────────────────────────────────────────────────────────────
_env_errors=0
for _var in GNB_SERVER RU_SERVER CUBB_HOST HOST_CONFIG; do
    if [ -z "${!_var}" ]; then
        echo "ERROR [env.sh]: ${_var} is not set. Edit env.sh or export it before running."
        _env_errors=$((_env_errors + 1))
    fi
done
if [ $_env_errors -gt 0 ]; then
    return 1 2>/dev/null || exit 1
fi

# Derived: per-server SSH user@ prefixes (empty string when username is unset)
_GNB_SSH_USER_AT="${GNB_SSH_USER:+${GNB_SSH_USER}@}"
_RU_SSH_USER_AT="${RU_SSH_USER:+${RU_SSH_USER}@}"
export _GNB_SSH_USER_AT _RU_SSH_USER_AT

# ── SSH logging hook ──────────────────────────────────────────────────────────
# Default: plain timestamped echo.
# Override in the calling script for structured logging, e.g.:
#   _ssh_log() { log_cmd "$1"; }
_ssh_log() { echo "[SSH ][$(date -u '+%H:%M:%S.%6N')] $*"; }

# Log the full expanded command being executed. Set SSH_LOG_EXEC=1 to enable.
_ssh_log_exec() {
    if [ "${SSH_LOG_EXEC}" = "1" ]; then
        echo "[EXEC][$(date -u '+%H:%M:%S.%6N')] $*";
    fi
    return 0;
}

# ── SSH helpers ───────────────────────────────────────────────────────────────
# Run cmd synchronously inside the named container on server (foreground).
# $4 = user@ prefix (e.g. "${_GNB_SSH_USER_AT}"), $5 = password (empty = key auth)
#
# ⚠️  Single-quote limitation: cmd is embedded inside bash -c '...', so any
# literal single quotes in cmd will break the shell quoting and cause the
# remote command to fail or misbehave.  Callers must NOT include single quotes
# in the cmd string.  Use double quotes or escape sequences instead.
_ssh_exec() {
    local server="$1"; local cmd="$2"; local flags="${3:--tt}"
    local user_at="${4:-}"; local pass="${5:-}"
    local full_cmd="export cuBB_SDK=${CUBB_SDK} && ${cmd}"
    local container_cmd="docker exec -i ${CONTAINER} bash -c '${full_cmd}'"
    if [ -n "${pass}" ]; then
        # Pass the password via the SSHPASS env var (sshpass -e) instead of -p so
        # it is not exposed in the process listing (/proc/*/cmdline, ps aux).
        # Do not interpolate ${pass} into the log line — that would leak credentials.
        _ssh_log_exec "SSHPASS=<redacted> sshpass -e ssh ${flags} \"${user_at}${server}\" \"${container_cmd}\""
        SSHPASS="${pass}" sshpass -e ssh ${flags} "${user_at}${server}" "${container_cmd}"
    else
        _ssh_log_exec "ssh ${flags} \"${user_at}${server}\" \"${container_cmd}\""
        ssh ${flags} "${user_at}${server}" "${container_cmd}"
    fi
}

# Execute cmd inside gNB container (foreground, returns exit code).
ssh_gnb() {
    local cmd="$1"
    _ssh_log "ssh gNB[${GNB_SERVER}]: ${cmd}"
    _ssh_exec "${GNB_SERVER}" "$cmd" "-tt" "${_GNB_SSH_USER_AT}" "${GNB_SSH_PASS}"
}

# Execute cmd inside RU container (foreground).
ssh_ru() {
    local cmd="$1"
    _ssh_log "ssh RU[${RU_SERVER}]: ${cmd}"
    _ssh_exec "${RU_SERVER}" "$cmd" "-tt" "${_RU_SSH_USER_AT}" "${RU_SSH_PASS}"
}

# Execute cmd inside the gNB container in the background.
# The SSH+docker-exec process is backgrounded on the TOP-SCRIPT server so its
# output is redirected to ${LOG_PATH}/screenlog_${title}.log via NFS — no
# dependency on screen being installed on the remote host, and no requirement
# for the remote host to have NFS mounted at the log path.
# title must match what check_result_cumcp.py expects: "mac" for testMAC, "cum" for cuMAC-CP.
#
# ⚠️  Single-quote limitation: cmd is embedded inside bash -c '...', so any
# literal single quotes in cmd will break the shell quoting and cause the
# remote command to fail or misbehave.  Callers must NOT include single quotes
# in the cmd string.  Use double quotes or escape sequences instead.
ssh_gnb_bg() {
    local title="$1"; local cmd="$2"
    _ssh_log "ssh gNB[${GNB_SERVER}] [bg:${title}]: ${cmd}"
    local logfile="${LOG_PATH}/screenlog_${title}.log"
    local full_cmd="export cuBB_SDK=${CUBB_SDK} && ${cmd}"
    if [ -n "${GNB_SSH_PASS:-}" ]; then
        SSHPASS="${GNB_SSH_PASS}" sshpass -e ssh "${_GNB_SSH_USER_AT}${GNB_SERVER}" \
            "docker exec -i ${CONTAINER} bash -c '${full_cmd}; echo ${title}_exit_code=\$?'" \
            > "${logfile}" 2>&1 &
    else
        ssh "${_GNB_SSH_USER_AT}${GNB_SERVER}" \
            "docker exec -i ${CONTAINER} bash -c '${full_cmd}; echo ${title}_exit_code=\$?'" \
            > "${logfile}" 2>&1 &
    fi
}

# Execute cmd inside the RU container in the background.
# Same pattern as ssh_gnb_bg: SSH is backgrounded on the top-script server,
# output redirected to NFS screenlog directly.
#
# ⚠️  Single-quote limitation: cmd is embedded inside bash -c '...', so any
# literal single quotes in cmd will break the shell quoting and cause the
# remote command to fail or misbehave.  Callers must NOT include single quotes
# in the cmd string.  Use double quotes or escape sequences instead.
ssh_ru_bg() {
    local title="$1"; local cmd="$2"
    _ssh_log "ssh RU[${RU_SERVER}] [bg:${title}]: ${cmd}"
    local logfile="${LOG_PATH}/screenlog_${title}.log"
    local full_cmd="export cuBB_SDK=${CUBB_SDK} && ${cmd}"
    if [ -n "${RU_SSH_PASS:-}" ]; then
        SSHPASS="${RU_SSH_PASS}" sshpass -e ssh "${_RU_SSH_USER_AT}${RU_SERVER}" \
            "docker exec -i ${CONTAINER} bash -c '${full_cmd}; echo ${title}_exit_code=\$?'" \
            > "${logfile}" 2>&1 &
    else
        ssh "${_RU_SSH_USER_AT}${RU_SERVER}" \
            "docker exec -i ${CONTAINER} bash -c '${full_cmd}; echo ${title}_exit_code=\$?'" \
            > "${logfile}" 2>&1 &
    fi
}
