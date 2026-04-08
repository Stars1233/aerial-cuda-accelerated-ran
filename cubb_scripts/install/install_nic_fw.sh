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

#
# install_nic_fw.sh - Install BlueField3 BFB firmware for Supermicro GH200
#
# This script installs BlueField3 NIC firmware via bfb-install.
# It is only applicable to Supermicro_ARS-111GL-NHR platforms;
# on all other platforms it exits immediately with no action.
#
# Prerequisites: DOCA/OFED must already be installed (provides bfb-install and mlxfwmanager).
#
# Usage: ./install_nic_fw.sh [--dry-run] [--verbose] [--check] [--rshim=N] [--help]
#
#   --dry-run        Show commands without executing
#   --verbose        Print commands before executing
#   --check          Check current FW version only; exit 0 if up-to-date, 1 if update needed
#   --rshim=N        Use /dev/rshimN (default: 0)
#   -h, --help       Show this help message
#

_SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
[[ -f "$_SCRIPT_DIR/includes.sh" ]] && source "$_SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $_SCRIPT_DIR/includes.sh" >&2; exit 1; }
[[ -f "$_SCRIPT_DIR/versions.sh" ]] && source "$_SCRIPT_DIR/versions.sh" || { echo "ERROR: versions.sh not found: $_SCRIPT_DIR/versions.sh" >&2; exit 1; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install BlueField3 BFB firmware for Supermicro GH200 (Supermicro_ARS-111GL-NHR only)"
    echo ""
    echo "Options:"
    echo "  --dry-run        Show commands without executing"
    echo "  --verbose        Print commands before executing"
    echo "  --check          Check FW version only (exit 0 if current, 1 if update needed)"
    echo "  --rshim=N        Use /dev/rshimN (default: 0)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Expected BFB: ${BFB_FILE:-not set}"
    echo "BFB URL:      https://content.mellanox.com/BlueField/FW-Bundle/${BFB_FILE:-<not set>}"
    exit "${1:-0}"
}

# Guard: skip entirely on non-Supermicro platforms
if [[ -z "${BFB_FILE:-}" ]]; then
    echo_and_log "[INFO] BFB_FILE is not set for platform '$PLATFORM' — NIC firmware install skipped"
    exit 0
fi

# Parse common arguments (--dry-run, --verbose) and populate REMAINING_ARGS
parse_common_args "$@"
verify_secure_boot_disabled

CHECK_ONLY=0
RSHIM_NUM=0

set -- "${REMAINING_ARGS[@]}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --check) CHECK_ONLY=1; shift ;;
        --rshim=*) RSHIM_NUM="${1#*=}"; shift ;;
        --rshim) RSHIM_NUM="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

RSHIM_DEV="/dev/rshim${RSHIM_NUM}"
BFB_URL="https://content.mellanox.com/BlueField/FW-Bundle/${BFB_FILE}"

# Query current BlueField3 FW version via mlxfwmanager
# Returns the running FW version string, or empty if unavailable.
get_current_fw_version() {
    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] sudo mlxfwmanager --query (would report current BF3 FW)"
        echo ""
        return
    fi

    local fw_output
    fw_output=$(sudo mlxfwmanager --query 2>/dev/null | grep -A5 "BlueField\|MT416\|mt41692" | grep "FW Version" | head -1 | awk '{print $NF}') || true
    echo "$fw_output"
}

# Check whether the installed FW matches BFB_VERSION.
# Returns 0 if up-to-date, 1 if update is needed.
check_nic_fw_version() {
    echo_and_log "[INFO] Checking BlueField3 FW version..."
    echo_and_log "[INFO] Expected BFB version: ${BFB_VERSION}"

    local current_fw
    current_fw=$(get_current_fw_version)

    if [[ -z "$current_fw" ]]; then
        if [[ $DRYRUN -eq 1 ]]; then
            echo_and_log "[DRY-RUN] Cannot determine current FW version — assuming update needed"
            return 1
        fi
        echo_and_log "[WARN] Could not determine current BlueField3 FW version"
        echo_and_log "[WARN] Assuming update is needed"
        return 1
    fi

    echo_and_log "[INFO] Current FW version: ${current_fw}"

    # BFB_VERSION format: "3.2.1-34_25.11-prod"; FW version reported by mlxfwmanager
    # typically matches the build portion (e.g., 32.2411.2050). Do a substring match on
    # the numeric part before the first underscore/dash to catch common versioning styles.
    local bfb_base="${BFB_VERSION%%-*}"
    if echo "$current_fw" | grep -qF "$bfb_base"; then
        echo_and_log "[INFO] BlueField3 FW is up-to-date (${current_fw})"
        return 0
    else
        echo_and_log "[INFO] FW update required: current=${current_fw}, expected base=${bfb_base}"
        return 1
    fi
}

# Download BFB file
download_bfb() {
    if [[ -f "${BFB_FILE}" && -s "${BFB_FILE}" ]]; then
        echo_and_log "[INFO] Using existing ${BFB_FILE} ($(du -h "${BFB_FILE}" | cut -f1))"
    else
        echo_and_log "[INFO] Downloading BFB from ${BFB_URL}..."
        execute_or_die "wget -nv -O ${BFB_FILE} ${BFB_URL}"
    fi
}

# Install BFB via bfb-install
install_bfb() {
    echo_and_log "[INFO] Installing BFB firmware via bfb-install on ${RSHIM_DEV}..."

    # Start the rshim daemon if the device isn't present yet.
    # rshim (userspace) creates /dev/rshimN as a directory; it is started via systemd.
    if [[ ! -e "${RSHIM_DEV}" ]]; then
        echo_and_log "[INFO] ${RSHIM_DEV} not found — starting rshim service..."
        execute "sudo systemctl start rshim"
        sleep 3
    fi

    if [[ ! -e "${RSHIM_DEV}" ]] && [[ $DRYRUN -eq 0 ]]; then
        echo_and_log "[ERROR] rshim device not found after starting rshim service: ${RSHIM_DEV}"
        echo_and_log "[ERROR] Available rshim devices: $(ls /dev/rshim* 2>/dev/null || echo 'none')"
        echo_and_log "[ERROR] Check rshim service: sudo systemctl status rshim"
        exit 1
    fi

    execute_or_die "sudo bfb-install -r ${RSHIM_DEV} -b ${BFB_FILE}"
}

# Wait for BlueField3 to come back online after BFB install by polling mst status
wait_for_bfb() {
    echo_and_log "[INFO] Waiting for BlueField3 to come back online (up to 120s)..."

    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] Would poll: sudo mst status | grep -q mt41692"
        return 0
    fi

    local elapsed=0
    local max_wait=120
    while [[ $elapsed -lt $max_wait ]]; do
        if sudo mst status 2>/dev/null | grep -q "mt41692"; then
            echo_and_log "[INFO] BlueField3 is back online after ${elapsed}s"
            return 0
        fi
        sleep 5
        ((elapsed += 5))
        echo_and_log "[INFO] Still waiting... (${elapsed}s elapsed)"
    done

    echo_and_log "[WARN] BlueField3 did not come back online within ${max_wait}s"
    echo_and_log "[WARN] A full power cycle may be required"
    return 1
}

# Main execution
main() {
    echo "============================================"
    echo_and_log "BlueField3 BFB Firmware Installation"
    echo_and_log "Platform: ${PLATFORM}"
    echo_and_log "BFB file: ${BFB_FILE}"
    echo_and_log "rshim:    ${RSHIM_DEV}"
    echo "============================================"
    echo ""

    if [[ $CHECK_ONLY -eq 1 ]]; then
        check_nic_fw_version
        exit $?
    fi

    # Check if update is needed
    if check_nic_fw_version; then
        echo_and_log "[INFO] BlueField3 FW already up-to-date. Nothing to do."
        exit 0
    fi

    download_bfb
    install_bfb
    wait_for_bfb

    echo ""
    echo "============================================"
    echo_and_log "[INFO] BFB firmware installed successfully!"
    echo ""
    echo_and_log "[IMPORTANT] A full system POWER CYCLE is required for the new firmware to take effect."
    echo_and_log "[IMPORTANT] A soft reboot is NOT sufficient — please power off and power on the system."
    echo "============================================"
}

main "$@"
