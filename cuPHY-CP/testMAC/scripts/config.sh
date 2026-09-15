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

# cuMAC-CP + cuBB Test Script — Test Configuration and Configuration Helpers
# Source this file from run_cumcp_cubb_test.sh (do not execute directly).
# Provides: test parameter variables, cfg_set(), and NFS artifact-exchange helpers.

# ── Step 1 configuration ─────────────────────────────────────────────────────
# DUMP_SRS_SLOT_NUM: number of SRS slots to dump in Step 1.
#   Must be >= 1.  Typical value: 4.
#   This is passed as an env var to run2_cuPHYcontroller.sh to instruct
#   cuphycontroller to write SRS buffer snapshots to /tmp/cubb_srs_buffers_*.h5
#   inside the gNB container.
export DUMP_SRS_SLOT_NUM="${DUMP_SRS_SLOT_NUM:-4}"

# TV_SRC: source directory for copy_test_files.sh (--src argument).
#   ENG env:  leave unset — copy_test_files.sh locates TVs automatically via get_uuid.sh.
#   QA env:   set to the NFS TV path, e.g. /home/aerial/nfs/GPU_test_input
#   In QA the TV source is on NFS but the destination ($CUBB_SDK/testVectors) is inside
#   each container, so copy_test_files.sh must run on both gNB and RU containers.
export TV_SRC="${TV_SRC:-}"

# ── Step 2 configuration ─────────────────────────────────────────────────────
# TDD_PATTERN: TDD slot pattern used when Step 2 generates the cuMAC TVs.
#   Values: SDDDS / SSSSS.  Leave unset to use the baseline pattern (SDDDS).
export TDD_PATTERN="${TDD_PATTERN:-}"

# ── Step 3 configuration ─────────────────────────────────────────────────────
# CUMAC_TASK_MASK: cuMAC-CP task bitmask passed to run_cuMAC_CP.sh -m.
#   Default 0xF enables all tasks.  Override via env var or here.
export CUMAC_TASK_MASK="${CUMAC_TASK_MASK:-0xF}"

# RU_READY_TIMEOUT: seconds to wait for ru_emulator to report cell timing (Steps 1 and 3).
export RU_READY_TIMEOUT="${RU_READY_TIMEOUT:-180}"

# PHY_READY_TIMEOUT: seconds to wait for cuphycontroller to signal ready (Steps 1 and 3).
export PHY_READY_TIMEOUT="${PHY_READY_TIMEOUT:-120}"

# MAC_READY_TIMEOUT: seconds to wait for testMAC to signal ready (Steps 1 and 3).
export MAC_READY_TIMEOUT="${MAC_READY_TIMEOUT:-300}"

# CUMAC_READY_TIMEOUT: seconds to wait for cuMAC-CP to signal ready (Step 3).
export CUMAC_READY_TIMEOUT="${CUMAC_READY_TIMEOUT:-120}"

# TIMEOUT_BASE: extra buffer (seconds) added to duration for cumac_cp timeout.
export TIMEOUT_BASE="${TIMEOUT_BASE:-300}"

# ── Step 3 cuMAC-CP binary build directory ───────────────────────────────────
# CUMAC_BUILD_DIR: build subdirectory under $CUBB_SDK containing the cumac_cp binary.
#   Unset (default): uses the same build dir as cuBB (from RUN2_CUPHYCONTROLLER_PARAMS).
#   Set when cuMAC-CP is in a different build tree from cuBB (e.g. QA container).
#   Example: export CUMAC_BUILD_DIR=build
export CUMAC_BUILD_DIR="${CUMAC_BUILD_DIR:-}"

# ── Step 3 cuMAC-CP yaml configuration ───────────────────────────────────────
# SRS_SLOT_LAG: slot offset between SRS capture and cuMAC-CP processing in Step 3.
#   run_cumcp_sa.sh (Step 2) resets this to 0 for standalone TV generation.
#   Step 3 must restore it to the hardware-specific value (13 for CG1 platform).
export SRS_SLOT_LAG="${SRS_SLOT_LAG:-13}"

# ── nvlog overrides ───────────────────────────────────────────────────────────
# SHM_LEVEL: log level value to apply to nvlog_config.yaml.
#   - When FUNC_NAMES is also set: inserts shm_level per named function.
#   - When FUNC_NAMES is unset:   replaces the global shm_log_level line (default: 4).
#   Leave SHM_LEVEL unset to skip log level customization.
#   Example: export SHM_LEVEL=5
export SHM_LEVEL="${SHM_LEVEL:-}"

# LOG_SIZE: rotating log file size in bytes. Leave unset to retain the YAML value.
#   Example: export LOG_SIZE=32000000
export LOG_SIZE="${LOG_SIZE:-}"

# NVIPC_PCAP_ENABLE: enable nvIPC packet capture in the L2 adapter and collect
#   nvipc.pcap into the run log directory. Default: 0.
export NVIPC_PCAP_ENABLE="${NVIPC_PCAP_ENABLE:-0}"

# FUNC_NAMES: space-separated nvlog function names to update per-function.
#   Leave unset to fall back to the global shm_log_level replacement.
#   Example: export FUNC_NAMES="DRV.FUNC_UL DRV.FUNC_DL"
export FUNC_NAMES="${FUNC_NAMES:-}"

# ── Log path ─────────────────────────────────────────────────────────────────
# LOG_BASE: parent directory under which each run's timestamped log folder is created.
#   Dev env:  leave unset — defaults to ${CUBB_HOST}/logs
#   QA env:   set to a fixed path, e.g. ${CUBB_HOST}/Log/Sanity/cuMAC_CP
# In both cases, each run creates <LOG_BASE>/<timestamp>_<test_case>_<duration>s_PASS/FAIL.
export LOG_BASE="${LOG_BASE:-}"

# RESTORE_CONFIG_BY_GIT:
#   1 = restore tracked YAML files with git checkout on the gNB and RU.
#   0 = restore config files from the config_ori directories.
export RESTORE_CONFIG_BY_GIT="${RESTORE_CONFIG_BY_GIT:-0}"

# FORCE_REBUILD_CUBB:
#   1 = always rebuild Aerial SDK on gNB/RU even if the build folder exists.
#   0 = skip rebuild when the host build folder is already present (default).
export FORCE_REBUILD_CUBB="${FORCE_REBUILD_CUBB:-0}"

# FORCE_RENEW_CUMAC_CP_TV:
#   1 = pass -r to run_cumcp_sa.sh to force regenerate cuMAC-CP TV files.
#   0 = reuse existing TV when present (default).
export FORCE_RENEW_CUMAC_CP_TV="${FORCE_RENEW_CUMAC_CP_TV:-0}"

# ── Configuration helpers ─────────────────────────────────────────────────────
# These functions are called from run_cumcp_cubb_test.sh and depend on variables
# defined in env.sh (CUBB_SDK, CUBB_HOST, ssh_gnb, ssh_ru) and logging helpers
# defined in run_cumcp_cubb_test.sh (log_cfg, log_step, log_err).
# All references are resolved at call-time so sourcing order is not a concern.

# restore_config_files: reset config directories to baseline before applying any
# test-specific changes.  Must be called after build and before setup1_DU.sh so
# every test starts from a clean slate regardless of what the previous run left.
# Mirrors Step 4 in cubb_phase4_config.sh.
restore_config_files() {
    log_step "Restoring baseline config files ... RESTORE_CONFIG_BY_GIT=${RESTORE_CONFIG_BY_GIT} gNB_RU_Share=${gNB_RU_Share}"

    if [[ "${RESTORE_CONFIG_BY_GIT}" == "1" ]]; then
        local gnb_yamls="cuPHY/nvlog/config/nvlog_config.yaml"
        gnb_yamls+=" cuPHY-CP/cuphycontroller/config/"
        gnb_yamls+=" cuPHY-CP/testMAC/testMAC/test_mac_config.yaml"
        gnb_yamls+=" cuPHY-CP/testMAC/testMAC/test_cumac_config.yaml"
        gnb_yamls+=" cuMAC-CP/config/cumac_cp.yaml"
        gnb_yamls+=" cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/config.yaml"
        gnb_yamls+=" cuMAC/examples/parameters.yaml"
        gnb_yamls+=" cuMAC/examples/pfmSort/config.yaml"

        local ru_yamls="cuPHY-CP/ru-emulator/config/"
        if [[ "${gNB_RU_Share}" != "1" ]]; then
            ru_yamls+=" cuPHY/nvlog/config/nvlog_config.yaml"
        fi

        # Git checkout runs inside each container, so use its SDK root.
        ssh_gnb "cd ${CUBB_SDK} && git checkout -- ${gnb_yamls}" \
            || { log_err "git restore of gNB YAML files failed"; return 1; }
        ssh_ru "cd ${CUBB_SDK} && git checkout -- ${ru_yamls}" \
            || { log_err "git restore of RU YAML files failed"; return 1; }
        return 0
    fi

    # Snapshot restore (RESTORE_CONFIG_BY_GIT=0). Paths/snapshots match the
    # aerial_sdk config-restore table (config_ori / *_ori next to each target).
    # 1) cuphycontroller/config ← config_ori  (gNB)
    ssh_gnb "cd ${CUBB_SDK}/cuPHY-CP/cuphycontroller/config && cp -rf ../config_ori/*.yaml ./" \
        || { log_err "restore cuphycontroller config failed"; return 1; }
    # 2) testMAC/testMAC ← testMAC_ori  (gNB)
    ssh_gnb "cd ${CUBB_SDK}/cuPHY-CP/testMAC && cp testMAC_ori/*.yaml testMAC/" \
        || { log_err "restore testMAC yaml failed"; return 1; }
    # 3) ru-emulator/config ← config_ori  (RU)
    ssh_ru "cd ${CUBB_SDK}/cuPHY-CP/ru-emulator/config && cp -rf ../config_ori/*.yaml ./" \
        || { log_err "restore ru-emulator config failed"; return 1; }
    # 4) nvlog_config.yaml ← nvlog_config.yaml_ori  (gNB; RU when not NFS-shared)
    local nvlog_yaml="${CUBB_SDK}/cuPHY/nvlog/config/nvlog_config.yaml"
    local nvlog_ori="${nvlog_yaml}_ori"
    ssh_gnb "if [ -f ${nvlog_ori} ]; then cp -f ${nvlog_ori} ${nvlog_yaml}; fi" \
        || { log_err "restore nvlog_config.yaml (gNB) failed"; return 1; }
    if [[ "${gNB_RU_Share}" != "1" ]]; then
        ssh_ru "if [ -f ${nvlog_ori} ]; then cp -f ${nvlog_ori} ${nvlog_yaml}; fi" \
            || { log_err "restore nvlog_config.yaml (RU) failed"; return 1; }
    fi
    # 5) cuMAC/examples/parameters.yaml ← parameters.yaml_ori  (gNB)
    local cumac_ex="${CUBB_SDK}/cuMAC/examples"
    ssh_gnb "if [ -f ${cumac_ex}/parameters.yaml_ori ]; then cp -f ${cumac_ex}/parameters.yaml_ori ${cumac_ex}/parameters.yaml; fi" \
        || { log_err "restore cuMAC parameters.yaml failed"; return 1; }
    # 6) cuMAC/examples/pfmSort/config.yaml ← config.yaml_ori  (gNB)
    ssh_gnb "if [ -f ${cumac_ex}/pfmSort/config.yaml_ori ]; then cp -f ${cumac_ex}/pfmSort/config.yaml_ori ${cumac_ex}/pfmSort/config.yaml; fi" \
        || { log_err "restore cuMAC pfmSort config.yaml failed"; return 1; }
    # 7) cuMAC-CP/config ← config_ori  (gNB); drop nested config_ori/config leftover
    ssh_gnb "if [ -d ${CUBB_SDK}/cuMAC-CP/config_ori ]; then cp -rf ${CUBB_SDK}/cuMAC-CP/config_ori/*.yaml ${CUBB_SDK}/cuMAC-CP/config/; fi" \
        || { log_err "restore cuMAC-CP config failed"; return 1; }
    # 8) muMimoUeGrpL2Integration/yamlConfigFiles ← yamlConfigFiles_ori  (gNB)
    local uegrp="${cumac_ex}/muMimoUeGrpL2Integration"
    ssh_gnb "if [ -d ${uegrp}/yamlConfigFiles_ori ]; then cp -rf ${uegrp}/yamlConfigFiles_ori/*.yaml ${uegrp}/yamlConfigFiles/; fi" \
        || { log_err "restore muMimoUeGrp yamlConfigFiles failed"; return 1; }
}

# set_log_config: update nvlog_config.yaml log level/size on the test hosts.
# No-op when SHM_LEVEL and LOG_SIZE are both unset.
# Must be called AFTER restore_config_files and the setup steps so the baseline
# yaml is in place before edits are applied.
#
# Two modes:
#   FUNC_NAMES set   → per-function: append "shm_level: N" after each named function
#   FUNC_NAMES unset → global:       replace the "shm_log_level: ..." line in the yaml
set_log_config() {
    [ -z "${SHM_LEVEL:-}" ] && [ -z "${LOG_SIZE:-}" ] && return 0
    local nvlog_yaml="${CUBB_SDK}/cuPHY/nvlog/config/nvlog_config.yaml"
    local log_cmd=""

    if [ -n "${SHM_LEVEL:-}" ] && [ -n "${FUNC_NAMES:-}" ]; then
        log_step "Setting per-function shm_level=${SHM_LEVEL} for [${FUNC_NAMES}] in nvlog_config.yaml ..."
        local func_name
        for func_name in ${FUNC_NAMES}; do
            ssh_gnb "sed -i \"/${func_name}/a \\      shm_level: ${SHM_LEVEL}\" ${nvlog_yaml}" \
                || { log_err "set_log_level failed for ${func_name}"; return 1; }
        done
    elif [ -n "${SHM_LEVEL:-}" ]; then
        log_step "Setting global shm_log_level=${SHM_LEVEL} in nvlog_config.yaml ..."
        log_cmd="sed -i \"s/shm_log_level:.*/shm_log_level: ${SHM_LEVEL}/\" ${nvlog_yaml};"
    fi

    if [ -n "${LOG_SIZE:-}" ]; then
        if ! [[ "${LOG_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
            log_err "LOG_SIZE must be a positive integer byte count (got '${LOG_SIZE}')"
            return 1
        fi
        log_step "Setting max_file_size_bytes=${LOG_SIZE} in nvlog_config.yaml ..."
        log_cmd="${log_cmd} sed -i \"s/max_file_size_bytes:.*/max_file_size_bytes: ${LOG_SIZE}/\" ${nvlog_yaml};"
    fi

    if [ -n "${log_cmd}" ]; then
        ssh_gnb "${log_cmd}" || { log_err "set_log_config failed on gNB"; return 1; }
        if [ "${gNB_RU_Share}" != "1" ]; then
            ssh_ru "${log_cmd}" || { log_err "set_log_config failed on RU"; return 1; }
        fi
    fi
}

# Backward-compatible name for out-of-tree callers.
set_log_level() {
    set_log_config
}

# cfg_set: write a single YAML key=value.
#   ENG (CUBB_SDK == CUBB_HOST): YAML files are on shared NFS → edit locally.
#   QA  (CUBB_SDK != CUBB_HOST): YAML files are inside the gNB container → edit via SSH.
cfg_set() {
    local key="$1"; local value="$2"; local file="$3"
    log_cfg "sed: $key = $value  →  $file"
    if [ "$CUBB_SDK" = "$CUBB_HOST" ]; then
        sed -i "s/${key}[ ]*:.*/${key}: ${value}/g" "$file"
    else
        ssh_gnb "sed -i \"s/${key}[ ]*:.*/${key}: ${value}/g\" \"$file\""
    fi
}

# NFS artifact handoff — only needed when CUBB_SDK != CUBB_HOST (QA environment).
# setup1_DU.sh writes test_config_summary.sh into the gNB container.  The RU
# container needs a copy before setup2_RU.sh runs; NFS (CUBB_HOST) is the handoff.

_copy_du_artifacts_to_nfs() {
    log_step "QA: copying DU artifacts to NFS ($CUBB_HOST) ..."
    ssh_gnb "cp ${phase4}/test_config_summary.sh ${CUBB_HOST}/" \
        || { log_err "Failed to copy test_config_summary.sh from gNB container to NFS"; return 1; }
}

_copy_nfs_artifacts_to_ru() {
    log_step "QA: copying DU artifacts from NFS to RU container ..."
    ssh_ru "cp ${CUBB_HOST}/test_config_summary.sh ${phase4}/" \
        || { log_err "Failed to copy test_config_summary.sh from NFS to RU container"; return 1; }
}

_sync_ru_artifacts_to_gnb() {
    log_step "QA: syncing RU artifacts back to gNB via NFS ..."
    ssh_ru  "cp ${phase4}/test_config_summary.sh ${CUBB_HOST}/" \
        || { log_err "Failed to copy test_config_summary.sh from RU container to NFS"; return 1; }
    ssh_gnb "cp ${CUBB_HOST}/test_config_summary.sh ${phase4}/" \
        || { log_err "Failed to copy test_config_summary.sh from NFS to gNB container"; return 1; }
}
