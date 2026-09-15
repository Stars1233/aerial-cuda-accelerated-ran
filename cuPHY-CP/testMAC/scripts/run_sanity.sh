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

# Mandatory configurations
export GNB_SERVER="${GNB_SERVER:-}"
export RU_SERVER="${RU_SERVER:-}"
export HOST_CONFIG="${HOST_CONFIG:-CG1_R750}"
export CUBB_HOST="${CUBB_HOST:-${cuBB_SDK}}"

# Mandatory configurations
# Using NFS to share the cuBB files between gNB and RU
export gNB_RU_Share=1
# Restore the initial config yaml files by git
export RESTORE_CONFIG_BY_GIT=1

# Optional configurations for debug the test scripts
export SSH_LOG_EXEC=1

# Runtime diagnostics. Callers can override any value before invoking this script.
export SHM_LEVEL="${SHM_LEVEL:-5}"
export LOG_SIZE="${LOG_SIZE:-32000000}"
export NVIPC_PCAP_ENABLE="${NVIPC_PCAP_ENABLE:-0}"
export TRY="${TRY:-0}"

# Optional configurations
export FORCE_REBUILD_CUBB=1

# cuMAC-CP test configurations
export FORCE_RENEW_CUMAC_CP_TV=1

################################################################################
# Test start from here
################################################################################
if [[ -z "$CUBB_HOST" || ! -d "$CUBB_HOST" ]]; then
    echo "Error: CUBB_HOST=${CUBB_HOST} is not set or not exists"
    exit 1
fi

if [ "$GNB_SERVER" == "" ]; then
    echo "Error: GNB_SERVER is not set"
    exit 1
fi

if [ "$RU_SERVER" == "" ]; then
    echo "Error: RU_SERVER is not set"
    exit 1
fi

cd $CUBB_HOST
export PATH=$PATH:$CUBB_HOST/cuPHY-CP/testMAC/scripts

################################################################################
# cuBB tests
################################################################################

# F08 4C 59c BFP9 STT455000 EH 1P
run_cumcp_cubb_test.sh "F08_4C_59c_BFP9_STT455000_EH_1P" 30 --test cubb

################################################################################
# cuMAC-CP tests
################################################################################

# 4T4R
run_cumcp_cubb_test.sh "F08_4C_60c_BFP9_STT455000_EH_1P" 20 1 0 --cumac_task 0xF --build_dir build.perf.aarch64 --test cumcp-sa
run_cumcp_cubb_test.sh "F08_4C_60c_BFP9_STT455000_EH_1P" 20 1 0 --cumac_task 0xF --build_dir build.perf.aarch64 --test other

# PFM SORT
run_cumcp_cubb_test.sh "F08_4C_60c_BFP9_STT455000_EH_1P" 20 1 0 --cumac_task 0x10 --build_dir build.perf.aarch64 --test cumcp-sa
run_cumcp_cubb_test.sh "F08_2C_60c_BFP9_STT455000_EH_1P" 20 1 0 --cumac_task 0x10 --build_dir build.perf.aarch64 --test other

# UE GROUP GPU_SHARE=0
run_cumcp_cubb_test.sh "F08_4C_66c_BFP9_STT455000_EH_1P" 20 1 0 --cumac_task 0x20 --build_dir build.perf.aarch64 --test cumcp-sa

# UE GROUP GPU_SHARE=1
run_cumcp_cubb_test.sh "F08_4C_66c_BFP9_STT455000_EH_1P" 20 1 1 --cumac_task 0x20 --build_dir build.perf.aarch64 --test cumcp-sa

# UE GROUP GPU_SHARE=2
run_cumcp_cubb_test.sh "F08_4C_66c_BFP9_STT455000_EH_1P" 20 1 2 --cumac_task 0x20 --build_dir build.perf.aarch64

# UE GROUP separate test steps
run_cumcp_cubb_test.sh "F08_4C_66c_BFP9_STT455000_EH_1P" 20 1 2 --cumac_task 0x20 --build_dir build.perf.aarch64 --test cubb-srs
run_cumcp_cubb_test.sh "F08_4C_66c_BFP9_STT455000_EH_1P" 20 1 2 --cumac_task 0x20 --build_dir build.perf.aarch64 --test cumcp-sa
run_cumcp_cubb_test.sh "F08_4C_66c_BFP9_STT455000_EH_1P" 20 1 2 --cumac_task 0x20 --build_dir build.perf.aarch64 --test other
