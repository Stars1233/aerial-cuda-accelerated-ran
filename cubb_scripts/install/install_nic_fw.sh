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

# Install and verify NIC firmware declared by versions.sh.
# BlueField-3 platforms install a BFB through every configured rshim device.
# ConnectX platforms flash an explicit firmware image with flint when one is
# configured; otherwise they rely on mlnx-fw-updater. Both paths verify the
# documented final firmware version.

_SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
[[ -f "$_SCRIPT_DIR/includes.sh" ]] && source "$_SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $_SCRIPT_DIR/includes.sh" >&2; exit 1; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install or verify the platform NIC firmware"
    echo ""
    echo "Options:"
    echo "  --dry-run        Show commands without executing"
    echo "  --verbose        Print commands before executing"
    echo "  --check          Verify the exact final firmware version only"
    echo "  --rshim=N        Override the configured rshim list with rshimN"
    echo "  -h, --help       Show this help message"
    exit "${1:-0}"
}

parse_common_args "$@"
verify_secure_boot_disabled

CHECK_ONLY=0
RSHIM_NUMS="${BFB_RSHIMS:-}"
NIC_FW_RESTART_MARKER="${_SCRIPT_DIR}/.stamps/nic_fw_power_cycle_required"

set -- "${REMAINING_ARGS[@]}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --check) CHECK_ONLY=1; shift ;;
        --rshim=*) RSHIM_NUMS="${1#*=}"; shift ;;
        --rshim) RSHIM_NUMS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

current_boot_id() {
    cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown
}

check_pending_firmware_restart() {
    [[ -f "$NIC_FW_RESTART_MARKER" ]] || return 0

    local flash_boot current_boot
    flash_boot=$(cat "$NIC_FW_RESTART_MARKER" 2>/dev/null || true)
    current_boot=$(current_boot_id)
    if [[ "$flash_boot" == "$current_boot" ]]; then
        if [[ "$PLATFORM" == "DGX-Spark" ]]; then
            echo_and_log "[ERROR] NIC firmware was updated during this boot. Reboot the host, then re-run make install."
        else
            echo_and_log "[ERROR] NIC firmware was updated during this boot. A full BMC/host cold power cycle is required."
            echo_and_log "[ERROR] A warm reboot is not sufficient. Re-run make install after the host returns."
        fi
        return 2
    fi

    if [[ "$PLATFORM" != "DGX-Spark" ]]; then
        # A new boot ID proves only that Linux restarted; operators must ensure
        # the documented full BMC/host cold power cycle was performed.
        echo_and_log "[INFO] Detected a new boot after the NIC firmware update; ensure the required cold power cycle was performed"
    else
        echo_and_log "[INFO] Detected the required reboot after the NIC firmware update"
    fi
    execute "rm -f '$NIC_FW_RESTART_MARKER'"
}

firmware_devices() {
    if [[ -n "${NIC_CONFIG_DEVICES:-}" ]]; then
        printf '%s\n' $NIC_CONFIG_DEVICES
    elif [[ -n "${NIC_DEV:-}" ]]; then
        printf '%s\n' "$NIC_DEV"
    fi
}

get_fw_version() {
    local device="$1"
    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] sudo mlxfwmanager -d $device --query (would report current firmware)" >&2
        return 1
    fi

    sudo mlxfwmanager -d "$device" --query 2>/dev/null \
        | awk '$1 == "FW" { print $2; exit }'
}

check_nic_psid() {
    [[ -n "${NIC_PSID:-}" ]] || return 0

    local device actual failed=0
    while read -r device; do
        [[ -n "$device" ]] || continue
        if [[ $DRYRUN -eq 1 ]]; then
            echo_and_log "[DRY-RUN] Would verify $device PSID is $NIC_PSID"
            continue
        fi
        actual=$(sudo mlxfwmanager -d "$device" --query 2>/dev/null \
            | awk -F: '/PSID/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}')
        if [[ "$actual" != "$NIC_PSID" ]]; then
            echo_and_log "[ERROR] $device PSID is ${actual:-unknown}; this profile only supports $NIC_PSID"
            failed=1
        else
            echo_and_log "[INFO] $device PSID: $actual"
        fi
    done < <(firmware_devices)
    return "$failed"
}

check_nic_fw_version() {
    if [[ -z "${NIC_FW_VERSION:-}" ]]; then
        echo_and_log "[INFO] No explicit NIC firmware target is defined for $PLATFORM; check skipped"
        return 0
    fi

    local device current failed=0 found=0
    echo_and_log "[INFO] Verifying NIC firmware target ${NIC_FW_VERSION}"
    while read -r device; do
        [[ -n "$device" ]] || continue
        found=1
        current=$(get_fw_version "$device" || true)
        if [[ $DRYRUN -eq 1 ]]; then
            echo_and_log "[DRY-RUN] $device firmware state is unknown; update path will be shown"
            failed=1
        elif [[ "$current" == "$NIC_FW_VERSION" ]]; then
            echo_and_log "[INFO] $device firmware: $current (expected)"
        else
            echo_and_log "[WARN] $device firmware: ${current:-unknown}; expected $NIC_FW_VERSION"
            failed=1
        fi
    done < <(firmware_devices)

    if [[ $found -eq 0 ]]; then
        echo_and_log "[ERROR] No NIC devices are configured for $PLATFORM"
        return 1
    fi
    return "$failed"
}

download_bfb() {
    if [[ -f "$BFB_FILE" && -s "$BFB_FILE" ]]; then
        echo_and_log "[INFO] Using existing $BFB_FILE ($(du -h "$BFB_FILE" | cut -f1))"
    else
        echo_and_log "[INFO] Downloading $BFB_FILE from $BFB_URL"
        execute_or_die "wget -nv -O '$BFB_FILE' '$BFB_URL'"
    fi
}

install_bfb() {
    local rshim_num rshim_dev
    execute "sudo systemctl restart rshim"
    # rshim creates /dev/rshimN asynchronously after the service starts.
    sleep 3

    for rshim_num in $RSHIM_NUMS; do
        rshim_dev="/dev/rshim${rshim_num}"
        echo_and_log "[INFO] Installing $BFB_FILE through $rshim_dev"
        if [[ ! -e "$rshim_dev" && $DRYRUN -eq 0 ]]; then
            echo_and_log "[ERROR] Missing $rshim_dev; available devices: $(ls -d /dev/rshim* 2>/dev/null || echo none)"
            return 1
        fi
        execute_or_die "sudo bfb-install -r '$rshim_dev' -b '$BFB_FILE'"
    done

    local wait_seconds="${BFB_WAIT_SECONDS:-120}"
    echo_and_log "[INFO] Allowing BlueField initialization to finish (${wait_seconds}s)"
    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] sleep $wait_seconds"
    else
        sleep "$wait_seconds"
    fi
}

record_firmware_restart_required() {
    if [[ $DRYRUN -eq 1 ]]; then
        if [[ "$PLATFORM" == "DGX-Spark" ]]; then
            echo_and_log "[DRY-RUN] Would record that a host reboot is required"
        else
            echo_and_log "[DRY-RUN] Would record that a full cold power cycle is required"
        fi
        return
    fi
    mkdir -p "$(dirname "$NIC_FW_RESTART_MARKER")"
    current_boot_id > "$NIC_FW_RESTART_MARKER"
}

get_flint_fw_version() {
    local device="$1"
    sudo flint -d "$device" q 2>/dev/null \
        | awk -F: '/FW Version:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}'
}

check_flint_fw_versions() {
    local device current_fw failed=0
    echo_and_log "[INFO] Verifying ${NIC_DEVICE_TYPE:-ConnectX} firmware target ${NIC_FW_VERSION}"
    for device in $NIC_FW_DEVICES; do
        if [[ $DRYRUN -eq 1 ]]; then
            echo_and_log "[DRY-RUN] Would query $device with flint"
            failed=1
            continue
        fi
        if [[ ! -e "$device" ]]; then
            echo_and_log "[ERROR] ${NIC_DEVICE_TYPE:-ConnectX} device not found: $device"
            failed=1
            continue
        fi
        current_fw=$(get_flint_fw_version "$device")
        if [[ "$current_fw" == "$NIC_FW_VERSION" ]]; then
            echo_and_log "[INFO] $device firmware: $current_fw (expected)"
        else
            echo_and_log "[WARN] $device firmware: ${current_fw:-unknown}; expected $NIC_FW_VERSION"
            failed=1
        fi
    done
    return "$failed"
}

install_flint_firmware() {
    if check_flint_fw_versions; then
        echo_and_log "[INFO] All ${NIC_DEVICE_TYPE:-ConnectX} devices are at the documented target"
        return 0
    fi
    [[ $CHECK_ONLY -eq 1 ]] && return 1
    if [[ $DRYRUN -eq 1 ]]; then
        echo_and_log "[DRY-RUN] Would download $NIC_FW_FILE and flash all devices: $NIC_FW_DEVICES"
        return 0
    fi
    if [[ ! -f "$NIC_FW_FILE" || ! -s "$NIC_FW_FILE" ]]; then
        echo_and_log "[INFO] Downloading ${NIC_DEVICE_TYPE:-ConnectX} firmware from $NIC_FW_URL"
        execute_or_die "wget -nv -O '$NIC_FW_FILE' '$NIC_FW_URL'"
    fi
    local device
    for device in $NIC_FW_DEVICES; do
        [[ -e "$device" ]] || { echo_and_log "[ERROR] ${NIC_DEVICE_TYPE:-ConnectX} device not found: $device"; return 1; }
        echo_and_log "[INFO] Flashing ${NIC_FW_VERSION} to $device"
        execute_or_die "sudo flint -d '$device' -i '$NIC_FW_FILE' -y b"
    done
    record_firmware_restart_required
    if [[ "$PLATFORM" == "DGX-Spark" ]]; then
        echo_and_log "[IMPORTANT] ${NIC_DEVICE_TYPE:-ConnectX} firmware was updated; reboot the host, then re-run make install."
    else
        echo_and_log "[IMPORTANT] ${NIC_DEVICE_TYPE:-ConnectX} firmware was updated; perform a full BMC/host cold power cycle, then re-run make install."
    fi
    return 2
}

main() {
    echo "============================================"
    echo_and_log "NIC Firmware Installation"
    echo_and_log "Platform: $PLATFORM"
    echo_and_log "Target:   ${NIC_FW_VERSION:-package managed}"
    echo_and_log "BFB:      ${BFB_FILE:-not used}"
    echo "============================================"

    check_pending_firmware_restart || exit $?
    # mlxfwmanager requires the /dev/mst device nodes created by MST, including
    # when this script is invoked directly by make check after a reboot.
    execute_or_die "sudo mst start"
    if [[ -n "${NIC_FW_FILE:-}" ]]; then
        check_nic_psid || return 1
        install_flint_firmware
        return $?
    fi
    check_nic_psid || return 1

    if check_nic_fw_version; then
        echo_and_log "[INFO] NIC firmware is at the documented target"
        exit 0
    fi

    if [[ $CHECK_ONLY -eq 1 ]]; then
        exit 1
    fi

    if [[ -z "${BFB_FILE:-}" ]]; then
        if [[ $DRYRUN -eq 1 ]]; then
            echo_and_log "[DRY-RUN] mlnx-fw-updater would install the compatible image; make check will require ${NIC_FW_VERSION:-the documented target}"
            return 0
        fi
        # Installing mlnx-fw-updater attempts a firmware update in its post-install
        # script. The firmware version is still mismatched, so retry the updater and
        # verify the documented target below.
        echo_and_log "[INFO] Updating ConnectX firmware with mlnx-fw-updater"
        execute "sudo mlnx-fw-updater"
        if check_nic_fw_version; then
            echo_and_log "[INFO] NIC firmware is now at the documented target"
            return 0
        fi
        echo_and_log "[ERROR] mlnx-fw-updater did not install the documented firmware target ${NIC_FW_VERSION:-unknown}"
        return 1
    fi
    if [[ -z "$RSHIM_NUMS" ]]; then
        echo_and_log "[ERROR] BFB_RSHIMS is empty for BlueField platform $PLATFORM"
        return 1
    fi

    download_bfb || return $?
    install_bfb || return $?
    record_firmware_restart_required

    echo ""
    echo_and_log "[IMPORTANT] The BFB was installed on rshim device(s): $RSHIM_NUMS"
    echo_and_log "[IMPORTANT] Perform a full BMC/host cold power cycle, then re-run make install."
    [[ $DRYRUN -eq 1 ]] && return 0
    return 2
}

main "$@"
