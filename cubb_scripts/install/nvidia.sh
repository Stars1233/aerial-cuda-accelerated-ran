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
# nvidia.sh - GPU and system optimizations for NVIDIA Aerial
#

FAILED=0

# Start Mellanox Software Tools
mst start || { echo "[ERROR] Failed to start MST"; FAILED=1; }

# Bring up aerial0* interfaces and set ethtool options (detect which exist)
AERIAL_INTERFACES=$(ip -o link show 2>/dev/null | grep -oE 'aerial0[0-9]+' | sort -u)
for iface in $AERIAL_INTERFACES; do
    ip link set "$iface" up || { echo "[WARN] Failed to bring up $iface"; }
    ethtool --set-priv-flags "$iface" tx_port_ts on 2>/dev/null || echo "[WARN] tx_port_ts not set on $iface (may not be supported)"
    ethtool -A "$iface" rx off tx off 2>/dev/null || true
done

# Force max frequency when the platform has a GPU (command is substituted during install).
if command -v nvidia-smi >/dev/null 2>&1; then
    @GPU_CLOCK_CMD@ || { echo "[ERROR] Failed to lock GPU frequency"; FAILED=1; }
    @GPU_MIG_CMD@ || { echo "[ERROR] Failed to configure MIG mode"; FAILED=1; }
fi

# Allow real-time tasks to take 100% CPU
echo -1 > /proc/sys/kernel/sched_rt_runtime_us || { echo "[ERROR] Failed to configure RT scheduling"; FAILED=1; }

# Disable timer migration
echo 0 | tee /proc/sys/kernel/timer_migration > /dev/null || { echo "[ERROR] Failed to disable timer migration"; FAILED=1; }

# Disable fair server on isolated CPUs (runtime=0 suppresses the DL fair server tick)
FAIR_SERVER_BASE="/sys/kernel/debug/sched/fair_server"
ISOLATED_CPUS_FILE="/sys/devices/system/cpu/isolated"
if [[ -d "$FAIR_SERVER_BASE" ]]; then
    isolated_cpus=""
    if [[ -r "$ISOLATED_CPUS_FILE" ]]; then
        read -r isolated_cpus < "$ISOLATED_CPUS_FILE" || isolated_cpus=""
    else
        echo "[WARN] Cannot read $ISOLATED_CPUS_FILE (non-fatal)"
    fi
    if [[ -n "$isolated_cpus" ]]; then
        for part in ${isolated_cpus//,/ }; do
            if [[ "$part" == *-* ]]; then
                cpu_range_start=${part%-*}
                cpu_range_end=${part#*-}
                for ((cpu=cpu_range_start; cpu<=cpu_range_end; cpu++)); do
                    runtime_path="${FAIR_SERVER_BASE}/cpu${cpu}/runtime"
                    if [[ -f "$runtime_path" ]]; then
                        echo 0 > "$runtime_path" || { echo "[ERROR] Failed to disable fair_server on CPU $cpu"; FAILED=1; }
                    else
                        echo "[WARN] Missing $runtime_path"
                    fi
                done
            else
                runtime_path="${FAIR_SERVER_BASE}/cpu${part}/runtime"
                if [[ -f "$runtime_path" ]]; then
                    echo 0 > "$runtime_path" || { echo "[ERROR] Failed to disable fair_server on CPU $part"; FAILED=1; }
                else
                    echo "[WARN] Missing $runtime_path"
                fi
            fi
        done
    fi
else
    echo "[WARN] fair_server debugfs not available (non-fatal)"
fi

# Pin all RCU processes to core 1
[[ -x /usr/local/bin/rcu_affinity_manager.sh ]] && \
    /usr/local/bin/rcu_affinity_manager.sh -w -c 1 || echo "[WARN] RCU pinning failed (non-fatal)"

# Enable DPDK mapping of GPU memory on GPU platforms.
if command -v nvidia-smi >/dev/null 2>&1; then
    modprobe nvidia-peermem 2>/dev/null || lsmod | grep -q nvidia_peermem || { echo "[ERROR] Failed to load nvidia-peermem"; FAILED=1; }
fi

# The CPU DMA latency service is intentionally Spark-only.
if systemctl list-unit-files cpu-latency.service --no-legend 2>/dev/null | grep -q '^cpu-latency.service'; then
    systemctl start cpu-latency.service || { echo "[ERROR] Failed to start cpu-latency.service"; FAILED=1; }
fi

# Exit with status
[[ $FAILED -eq 1 ]] && { echo "[ERROR] Completed with errors"; exit 1; }
echo "[INFO] Completed successfully"
exit 0
