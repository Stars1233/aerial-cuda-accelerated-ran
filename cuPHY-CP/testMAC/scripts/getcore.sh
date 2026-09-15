#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if [ "$1" != "" ]; then
    process_names="$*"
else
    process_names="gdb phc2sys ptp4l ru_emulator l2_adapter_cuphycontroller l2_adapter_cuphycontroller_scf test_mac cuphycontroller cuphycontroller_scf tick_unit_test nvlog_observer"
    process_names="${process_names} nvipc_cunit test_ipc sched_fifo_poll cumac_cp"
    process_names="${process_names} l2_test cumcp_test"
fi

echo "process_names: $process_names"
echo "----------------------------------------"

# RSS: resident set size, the non-swapped physical memory that a task has used (in kiloBytes)
# VSZ: virtual memory size of the process in KiB (1024-byte units)
# DRS: data resident set size, the amount of physical memory devoted to other than executable code
# VSZ = DRS ?

print_cores () {
  pnames=$1
  for pname in $pnames
  do
    pid=$(pidof ${pname})
    if [ "${pid}" != "" ]; then
      echo "===== NAME: ${pname} PID: ${pid} ====="
      ps H -o 'pid tid comm policy psr priority %cpu %mem vsz rss' ${pid}
      echo ""
    fi
  done
}

print_cores "${process_names}"
