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

# This script to be used after getting into the docker image, or as a docker entrypoint,
# with aerial compiled as in testBenches/phase4_test_scripts/build_aerial_sdk.sh

export cuBB_SDK=${cuBB_SDK:-$(pwd)}
export CUDA_DEVICE_MAX_CONNECTIONS=8
config_file=""
if [[ -n "$1" ]]; then
    config_file=$1
fi

stop_mps() {
    # Stop existing MPS
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
    echo quit | sudo -E nvidia-cuda-mps-control || true
}

restart_mps() {
    # Stop existing MPS
    stop_mps

    # Start MPS
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
    export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
    sudo -E nvidia-cuda-mps-control -d
    echo start_server -uid 0 | sudo -E nvidia-cuda-mps-control
    sleep 1  # Give daemon time to start
    if ! pgrep -f nvidia-cuda-mps > /dev/null; then
        echo "Error: MPS daemon failed to start" >&2
        return 1
    fi
}

get_platform_name() {
    local _dmi=/sys/devices/virtual/dmi/id
    if [[ ! -r "$_dmi/board_vendor" || ! -r "$_dmi/board_name" || ! -r "$_dmi/board_version" ]]; then
        echo "Error: Could not read DMI files: Vendor: $_dmi/board_vendor, Name: $_dmi/board_name, Version: $_dmi/board_version" >&2
        return
    fi
    local _platform_name
    _platform_name=$(cat $_dmi/board_vendor)-$(cat $_dmi/board_name)-$(cat $_dmi/board_version)
    echo "${_platform_name}"
}

_arch=$(uname -m)
cuphy_bin="cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf"
l1_bin="${cuBB_SDK}/build.${_arch}/${cuphy_bin}"
if [[ ! -x "${l1_bin}" ]]; then
    l1_bin="${cuBB_SDK}/build/${cuphy_bin}"
fi
if [[ ! -x "${l1_bin}" ]]; then
    echo "Error: Binary not found or not executable (checked build and build.${_arch})" >&2
    exit 1
fi

_platform_name=$(get_platform_name)
if [[ -z "$_platform_name" ]]; then
    echo "Error: Failed to detect platform" >&2
    exit 1
fi

case $_platform_name in
    "NVIDIA-P4242-A04")
        export UCX_REG_MT_THRESH=inf
        config_file=${config_file:-P5G_WNC_DGX}
        ;;
    "Supermicro-G1SMH-G-1.02")
        config_file=${config_file:-P5G_WNC_GH}
        ;;
    *)
        echo "Unknown platform: $_platform_name"
        config_file=${config_file:-P5G_WNC_GH}
        echo "Using default config file: cuphycontroller_${config_file}.yaml"
        ;;
esac

cuphy_yaml="${cuBB_SDK}/cuPHY-CP/cuphycontroller/config/cuphycontroller_${config_file}.yaml"
if [[ ! -f "$cuphy_yaml" ]]; then
    echo "Error: cuphycontroller YAML not found: $cuphy_yaml" >&2
    exit 1
fi

if [[ -n "${RU_MAC:-}" ]]; then
    if ! [[ "$RU_MAC" =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]; then
        echo "Error: RU_MAC must be an Ethernet MAC like aa:bb:cc:dd:ee:ff (got: $RU_MAC)" >&2
        exit 1
    fi
    echo "Applying RU_MAC=$RU_MAC to first dst_mac_addr in $cuphy_yaml"
    sudo sed -i "0,/dst_mac_addr:/{s/dst_mac_addr:.*/dst_mac_addr: $RU_MAC/}" "$cuphy_yaml"
fi

# MPS is incompatible with cuphycontroller when use_green_contexts is enabled.
_use_green_contexts=$(yq -r '.cuphydriver_config.use_green_contexts // 0' "$cuphy_yaml")
_use_green_contexts=${_use_green_contexts,,}
if [[ "$_use_green_contexts" == "true" || "$_use_green_contexts" == "1" ]]; then
    echo "run_l1: use_green_contexts enabled ($cuphy_yaml) — stopping MPS"
    stop_mps
else
    restart_mps
fi

sudo -E "${l1_bin}" "${config_file}"
