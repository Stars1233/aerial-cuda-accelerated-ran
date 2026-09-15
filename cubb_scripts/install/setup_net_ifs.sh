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

# Source common functions
_SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
[[ -f "$_SCRIPT_DIR/includes.sh" ]] && source "$_SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $_SCRIPT_DIR/includes.sh" >&2; exit 1; }
parse_common_args "$@"

# Match the site.yaml network_interfaces role: only supported Mellanox products
# are renamed, with newer adapters ordered first and PCI bus address used
# as the tie-breaker.
SUPPORTED_MLX_PRODUCTS="ConnectX-6 Dx|ConnectX-7|ConnectX-8|BlueField-2|BlueField-3"

get_network_hardware() {
    if [[ $DRYRUN -eq 1 ]]; then
        # Discovery does not require root on supported hosts. Avoid prompting
        # for sudo during a non-mutating preview.
        lshw -json -class network 2>/dev/null
    else
        sudo lshw -json -class network 2>/dev/null
    fi
}

mapfile -t nic_rows < <(
    get_network_hardware \
    | jq -r \
        --arg supported "$SUPPORTED_MLX_PRODUCTS" \
        --arg businfo_regex "${AERIAL_NET_BUSINFO_REGEX:-}" '
        def product_sort:
            if ((.product // "") | test("ConnectX-8")) then 0
            elif ((.product // "") | test("ConnectX-7|BlueField-3")) then 1
            elif ((.product // "") | test("ConnectX-6 Dx|BlueField-2")) then 2
            else 9 end;
        if type == "array" then . else [.] end
        | map(
            select(.configuration.driver? == "mlx5_core")
            | select((.product // "") | test($supported))
            | select($businfo_regex == "" or ((.businfo // "") | test($businfo_regex)))
            | select(.serial != null and .serial != "00:00:00:00:00:00")
        )
        | sort_by([product_sort, .businfo])
        | .[]
        | [.serial, (.product // "unknown"), (.businfo // "unknown")]
        | @tsv
    '
)

expected_interface_count="${AERIAL_NET_EXPECTED_COUNT:-${EXPECTED_AERIAL_INTERFACES:-}}"
if [[ -n "$expected_interface_count" && ${#nic_rows[@]} -ne $expected_interface_count ]]; then
    if [[ -n "${AERIAL_NET_EXPECTED_COUNT:-}" ]]; then
        echo_and_log "[ERROR] Expected ${AERIAL_NET_EXPECTED_COUNT} fronthaul interface(s) for ${PLATFORM}, found ${#nic_rows[@]}"
        echo_and_log "[ERROR] lshw businfo filter: ${AERIAL_NET_BUSINFO_REGEX:-<none>}"
    else
    echo_and_log "[ERROR] Detected ${#nic_rows[@]} supported mlx5 interface(s); $PLATFORM requires $EXPECTED_AERIAL_INTERFACES"
    fi
    exit 1
fi

for i in "${!nic_rows[@]}"; do
    IFS=$'\t' read -r mac_addr product businfo <<< "${nic_rows[$i]}"
    ifname="aerial0${i}"
    filename="/etc/systemd/network/20-${ifname}.link"
    if [[ $DRYRUN -eq 1 ]]; then
        echo "[DRY-RUN] Would create $filename for $ifname with ${mac_addr} (${product}, ${businfo})"
    else
        echo_and_log "[INFO] Creating $filename for $ifname with ${mac_addr} (${product}, ${businfo})"
        sudo tee "$filename" > /dev/null <<EOF
[Match]
MACAddress=${mac_addr}

[Link]
Name=$ifname
EOF
    fi
done

if [[ ${#nic_rows[@]} -gt 0 ]]; then
    echo_and_log "[INFO] Created ${#nic_rows[@]} aerial interface link file(s)"
    execute sudo udevadm control --reload-rules
    execute sudo udevadm trigger --action=add --subsystem-match=net
    execute sudo netplan apply
else
    echo_and_log "[WARN] No mlx5_core interfaces detected; no aerial interfaces configured"
fi
