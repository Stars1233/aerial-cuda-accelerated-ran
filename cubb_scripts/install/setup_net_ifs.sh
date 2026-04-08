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

# Find MAC addresses of all interfaces bound to mlx5_core (ConnectX-7 and BlueField-3),
# ordered by PCI address so that aerial00/01/02/03 map to ascending PCI slots.
mapfile -t macAddrs < <(
    sudo lshw -json -class network 2>/dev/null \
    | jq -r '
        if type == "array" then . else [.] end
        | map(select(.configuration.driver == "mlx5_core" and .serial != null and .serial != "00:00:00:00:00:00"))
        | sort_by(.businfo)
        | .[].serial
    '
)

for i in "${!macAddrs[@]}"; do
    ifname="aerial0${i}"
    filename="/etc/systemd/network/20-${ifname}.link"
    if [[ $DRYRUN -eq 1 ]]; then
        echo "[DRY-RUN] Would create $filename for $ifname with ${macAddrs[$i]}"
    else
        echo_and_log "[INFO] Creating $filename for $ifname with ${macAddrs[$i]}"
        sudo tee "$filename" > /dev/null <<EOF
[Match]
MACAddress=${macAddrs[$i]}

[Link]
Name=$ifname
EOF
    fi
done

if [[ ${#macAddrs[@]} -gt 0 ]]; then
    echo_and_log "[INFO] Created ${#macAddrs[@]} aerial interface link file(s)"
    execute sudo udevadm control --reload-rules
    execute sudo udevadm trigger --action=add --subsystem-match=net
    execute sudo netplan apply
else
    echo_and_log "[WARN] No mlx5_core interfaces detected; no aerial interfaces configured"
fi
