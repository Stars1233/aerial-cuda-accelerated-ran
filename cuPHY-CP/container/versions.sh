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
# versions.sh - Shared version definitions for Aerial host install and container builds.
#
# setup.sh is the source of truth for Aerial container identity:
#   AERIAL_VERSION_TAG, AERIAL_REPO, AERIAL_IMAGE_NAME, AERIAL_PLATFORM
#
# This file is the source of truth for software version pins. Host install scripts
# source it through cubb_scripts/install/includes.sh, which requires a supported
# host platform. Container build scripts source it directly and can use the common
# pins without running on SMC or DGX hardware.
#
# Usage:
#   source cuPHY-CP/container/versions.sh
#   AERIAL_REQUIRE_HOST_PLATFORM=1 source cuPHY-CP/container/versions.sh
#   PLATFORM=SMC-GraceHopper bash cuPHY-CP/container/versions.sh
#

if [[ "${BASH_SOURCE[0]:-$0}" != "${0}" ]] && [[ -n "${VERSIONS_SH_LOADED:-}" ]]; then
    return 0
fi
VERSIONS_SH_LOADED=1

_VERSIONS_SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")"
source "$_VERSIONS_SCRIPT_DIR/setup.sh"

# =============================================================================
# Common software versions
# =============================================================================

MANIFEST_VERSION="v26_2"
DOCA_VERSION="3.3.0"
DOCA_BUILD="088000-26.01"
DOCA_OFED_VERSION="OFED-internal-26.01-1.0.0"
NVIDIA_CONTAINER_TOOLKIT_VERSION="1.19.1"
NVIDIA_CONTAINER_TOOLKIT_VERSION_SUFFIX="${NVIDIA_CONTAINER_TOOLKIT_VERSION_SUFFIX:--1}"
MFT_VERSION="4.35.0-159"
MLNX_FW_UPDATER_VERSION="26.01-1.0.0.0"
DOCKER_VERSION="29.1.5"
GPU_DRIVER_VERSION="610.57.04"
GPU_DRIVER_ARCH="aarch64"
CUDA_VERSION="13.3.0"
CUDA_RUN_FILE_NAME="cuda_${CUDA_VERSION}_${GPU_DRIVER_VERSION}_linux_sbsa.run"
GDRDRV_VERSION="2.6-1"
GDRDRV_CUDA_VERSION="13.0"
GDRCOPY_UBUNTU_VER="ubuntu24_04"
LINUXPTP_VERSION="4.2"
YQ_VERSION="${YQ_VERSION:-4.50.1}"

# =============================================================================
# Container build versions
# =============================================================================

CONTAINER_CUDA_IMAGE="nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04"
CONTAINER_UBUNTU_DISTRO="ubuntu24"
CONTAINER_DOCA_UBUNTU_VERSION="ubuntu2404"
CONTAINER_GDRCOPY_VERSION="2.6"
MATHDX_VERSION="26.03.0"
MATHDX_CUDA_VERSION="cuda13"
TENSORRT_VERSION="10.14.1.48"
TENSORRT_CUDA_VERSION="13.0"
DOCA_FOR_AERIAL_VERSION="26-2"
LDPC_DECODER_CUBIN_VERSION="c3e5b9"
DOCA_TAG="3.3.0109-1"
DOCA_HASH="461bb22fcb0488faf14ee7b9758928ba58118e8f"

export_container_build_versions() {
    export CONTAINER_CUDA_IMAGE
    export CONTAINER_UBUNTU_DISTRO
    export CONTAINER_GDRCOPY_VERSION
    export DOCA_VERSION
    export DOCA_BUILD
    export CONTAINER_DOCA_UBUNTU_VERSION
    export MATHDX_VERSION
    export MATHDX_CUDA_VERSION
    export TENSORRT_VERSION
    export TENSORRT_CUDA_VERSION
    export DOCA_TAG
    export DOCA_HASH
    export DOCA_FOR_AERIAL_VERSION
    export LDPC_DECODER_CUBIN_VERSION
}

# =============================================================================
# Platform detection
# =============================================================================

if [[ -z "${PLATFORM_ID:-}" ]]; then
    DMI_PATH="${DMI_PATH:-/sys/devices/virtual/dmi/id}"
    _read_dmi() {
        local value
        value="$(cat "$DMI_PATH/$1" 2>/dev/null || true)"
        if [[ -n "$value" ]]; then
            printf "%s" "$value" | tr ' ' '_' | tr -s '_'
        else
            printf "unknown"
        fi
    }
    _vb="$(_read_dmi board_vendor)_$(_read_dmi product_family)_$(_read_dmi product_name)_$(_read_dmi board_name)"
    case "$_vb" in
        # There are multiple variants of the Supermicro ARS-111GL-NHR server.
        Supermicro_Family_Super_Server_G1SMH-G) PLATFORM_ID="SMC-GraceHopper" ;;
        Supermicro_Family_ARS-111GL-NHR_G1SMH-G) PLATFORM_ID="SMC-GraceHopper" ;;
        NVIDIA_DGX_Spark_NVIDIA_DGX_Spark_P4242) PLATFORM_ID="DGX-Spark" ;;
        *_MGX_QuantaEdge_EGN77C-2U_QuantaEdge_EGN77C-2U) PLATFORM_ID="MGX-ARC-Pro" ;;
        Dell_Inc._PowerEdge_PowerEdge_R750_*) PLATFORM_ID="Dell-R750" ;;
        *) PLATFORM_ID="$_vb" ;;
    esac
    VERBOSE_PLATFORM_ID="$_vb"
fi

PLATFORM="${PLATFORM:-$PLATFORM_ID}"
VERBOSE_PLATFORM_ID="${VERBOSE_PLATFORM_ID:-$PLATFORM}"
HOST_PLATFORM_SUPPORTED=1

# =============================================================================
# Host install platform-specific versions
# =============================================================================

case "$PLATFORM" in
    "DGX-Spark")
        KERNEL_VERSION="6.17.0-1018-nvidia"
        UBUNTU_VERSION="ubuntu2404"
        ARCH="arm64"

        HUGEPAGES="32"
        HUGEPAGE_SIZE="1G"
        NIC_DEV="/dev/mst/mt4129_pciconf0"
        NIC_DEVICE_TYPE="ConnectX7"
        NIC_FW_VERSION="28.47.1088"
        NIC_PSID="NVD0000000087"
        # The DOCA 3.3 updater does not carry an image for this PSID.
        NIC_FW_FILE="fw-ConnectX7-rel-28_47_1088-cx7_P4242_HORIZON_PK_Ax-UEFI-14.40.10-FlexBoot-3.8.201.signed.bin"
        NIC_FW_URL="https://nbu-nfs.gtm.nvidia.com/auto/host_fw_release/fw-4129/fw-4129-rel-28_47_1088-build-001/etc/bin/signed/${NIC_FW_FILE}"
        NIC_FW_DEVICES="$NIC_DEV"
        EXPECTED_AERIAL_INTERFACES="4"
        ISOLCPUS="4-19"
        IRQ_AFFINITY_CPUS="0-3"
        GRUB_PLATFORM_SPECIFIC_PARAMS=(
            "earlycon"
            "acpi_power_meter.force_cap_on=y"
            "init_on_alloc=0"
            "preempt=none"
        )
        PTP_CPU_AFFINITY="${PTP_CPU_AFFINITY:-4}"
        PTP_INTERFACE_DEFAULT="aerial00"
        AERIAL_BUILD_FLAGS="${AERIAL_BUILD_FLAGS:- --cuda-archs 121}"
        ;;

    "SMC-GraceHopper")
        KERNEL_VERSION="6.17.0-1018-nvidia-64k"
        UBUNTU_VERSION="ubuntu2404"
        ARCH="arm64"

        HUGEPAGES="48"
        HUGEPAGE_SIZE="512M"
        BFB_VERSION="3.3.0-202_26.01-prod"
        BFB_RSHIMS="0 1"
        BFB_WAIT_SECONDS="120"
        NIC_DEV="/dev/mst/mt41692_pciconf0"
        NIC_CONFIG_DEVICES="/dev/mst/mt41692_pciconf0 /dev/mst/mt41692_pciconf1"
        NIC_DEVICE_TYPE="BlueField3"
        NIC_FW_VERSION="32.48.1000"
        NIC_PSID="MT_0000000884"
        EXPECTED_AERIAL_INTERFACES="4"
        ISOLCPUS="4-64"
        IRQ_AFFINITY_CPUS="0-3"
        GRUB_PLATFORM_SPECIFIC_PARAMS=(
            "pci=pcie_bus_safe"
            "numa_balancing=disable"
            "earlycon"
            "acpi_power_meter.force_cap_on=y"
            "init_on_alloc=0"
            "preempt=none"
        )
        PTP_CPU_AFFINITY="${PTP_CPU_AFFINITY:-}"
        PTP_INTERFACE_DEFAULT="aerial00"
        MIG_MODE="0"
        AERIAL_BUILD_FLAGS="${AERIAL_BUILD_FLAGS:- --cuda-archs 90}"
        NVIDIA_MODULE_OPTIONS='options nvidia NVreg_RegistryDwords="RMNvLinkDisableLinks=0x3FFFF;"'
        ;;

    "MGX-ARC-Pro")
        KERNEL_VERSION="6.17.0-1018-nvidia-64k"
        UBUNTU_VERSION="ubuntu2404"
        ARCH="arm64"

        HUGEPAGES="48"
        HUGEPAGE_SIZE="512M"
        NIC_DEV="/dev/mst/mt4131_pciconf0"
        NIC_DEVICE_TYPE="ConnectX8"
        ISOLCPUS="4-64"
        IRQ_AFFINITY_CPUS="0-3"
        GRUB_PLATFORM_SPECIFIC_PARAMS=(
            "pci=pcie_bus_safe"
            "numa_balancing=disable"
            "earlycon"
            "acpi_power_meter.force_cap_on=y"
            "init_on_alloc=0"
            "preempt=none"
        )
        PTP_CPU_AFFINITY="${PTP_CPU_AFFINITY:-}"
        MIG_MODE="0"
        AERIAL_BUILD_FLAGS="${AERIAL_BUILD_FLAGS:- --cuda-archs 120}"

        # CX8-0 is the two-port QSFP adapter used by the LLS-C3 fronthaul
        # topology. Do not rename the 16 SFP ports on CX8-1/CX8-2.
        AERIAL_NET_BUSINFO_REGEX='^pci@0002:03:00\.[01]$'
        AERIAL_NET_EXPECTED_COUNT="2"
        AERIAL_FH_INTERFACES="aerial00 aerial01"

        # QP firmware for EVT systems. DVT/PVT systems require the signed PK image.
        NIC_FW_VERSION="40.97.5452"
        NIC_FW_FILE="fw-ConnectX8-rel-40_97_5452-cx8_P4180_MGX_ARC_QP_Ax-UEFI-14.41.14-FlexBoot-3.9.101.bin"
        NIC_FW_URL="https://nbu-nfs.gtm.nvidia.com/auto/host_fw_release/fw-4131/fw-4131-rel-40_97_5452-build-001/etc/bin/${NIC_FW_FILE}"
        NIC_FW_DEVICES="/dev/mst/mt4131_pciconf0 /dev/mst/mt4131_pciconf1 /dev/mst/mt4131_pciconf2"

        PCI_CONFIG_ACS='xx000x0@0000:00:00.0;xx000x0@0002:00:00.0;xx000x0@0002:02:00.0;xx000x0@0002:02:01.0;xx000x0@0002:02:03.0;xx000x0@0002:05:00.0;xx000x0@0002:08:00.0;xx000x0@0002:08:08.0;xx000x0@0004:00:00.0;xx000x0@0005:00:00.0;xx000x0@0006:00:00.0;xx000x0@0009:00:00.0'
        NVIDIA_MODULE_OPTIONS='options nvidia NVreg_RegistryDwords="RMNvLinkDisableLinks=0x3FFFF;"'
        ;;

    "Dell-R750")
        KERNEL_VERSION="6.8.0-1058-nvidia-lowlatency"
        UBUNTU_VERSION="ubuntu2404"
        ARCH="amd64"

        HUGEPAGES="16"
        HUGEPAGE_SIZE="1G"
        BFB_VERSION="3.3.0-202_26.01-prod"
        BFB_RSHIMS="0"
        BFB_WAIT_SECONDS="600"
        NIC_DEV="/dev/mst/mt41692_pciconf0"
        NIC_CONFIG_DEVICES="$NIC_DEV"
        NIC_DEVICE_TYPE="BlueField3"
        NIC_FW_VERSION="32.48.1000"
        NIC_PSID="MT_0000000884"
        EXPECTED_AERIAL_INTERFACES="2"
        ISOLCPUS="4-47"
        IRQ_AFFINITY_CPUS="0-3"
        GRUB_PLATFORM_SPECIFIC_PARAMS=(
            "clocksource=tsc"
            "intel_idle.max_cstate=0"
            "mce=ignore_ce"
            "intel_pstate=disable"
            "iommu=off"
            "noht"
            "numa_balancing=disable"
        )
        PTP_CPU_AFFINITY="${PTP_CPU_AFFINITY:-}"
        PTP_INTERFACE_DEFAULT="aerial00"
        AERIAL_FH_INTERFACES="aerial00 aerial01"
        INSTALL_GPU="0"
        INSTALL_DOCA_OFED="0"
        AERIAL_BUILD_FLAGS="${AERIAL_BUILD_FLAGS:- --toolchain r750}"
        ;;

    *)
        HOST_PLATFORM_SUPPORTED=0
        if [[ "${AERIAL_REQUIRE_HOST_PLATFORM:-0}" == "1" ]]; then
            echo "[ERROR] Unknown platform DMI ID: $PLATFORM"
            echo "[ERROR] Supported platforms and their DMI board_vendor_family_name_board_name:"
            echo "        DGX-Spark:"
            echo "           NVIDIA_DGX_Spark_NVIDIA_DGX_Spark_P4242"
            echo "        SMC-GraceHopper:"
            echo "           Supermicro_Family_Super_Server_G1SMH-G"
            echo "           Supermicro_Family_ARS-111GL-NHR_G1SMH-G"
            echo "        MGX-ARC-Pro:"
            echo "           unknown_MGX_QuantaEdge_EGN77C-2U_QuantaEdge_EGN77C-2U"
            echo "        Dell-R750:"
            echo "           Dell_Inc._PowerEdge_PowerEdge_R750_<board>"
            exit 1
        fi
        ;;
esac

# The installer calls this after parsing optional overrides, and version checks
# use the default generated value.
build_aerial_grub_cmdline_params() {
    local -a common_params params
    local IFS=' '

    [[ "$HOST_PLATFORM_SUPPORTED" == "1" ]] || return 0

    common_params=(
        "pci=realloc=off"
        "default_hugepagesz=${HUGEPAGE_SIZE}"
        "hugepagesz=${HUGEPAGE_SIZE}"
        "hugepages=${HUGEPAGES}"
        "tsc=reliable"
        "processor.max_cstate=0"
        "audit=0"
        "idle=poll"
        "rcu_nocb_poll"
        "nosoftlockup"
        "irqaffinity=${IRQ_AFFINITY_CPUS}"
        "isolcpus=managed_irq,domain,${ISOLCPUS}"
        "nohz_full=${ISOLCPUS}"
        "rcu_nocbs=${ISOLCPUS}"
        "module_blacklist=nouveau"
    )

    params=("${common_params[@]}" "${GRUB_PLATFORM_SPECIFIC_PARAMS[@]}")

    printf '%s\n' "${params[*]}"
}

AERIAL_GRUB_CMDLINE_PARAMS="$(build_aerial_grub_cmdline_params)"

export AERIAL_VERSION_TAG

# ARCH for binary downloads (e.g. yq); set per-platform above or default from dpkg.
ARCH="${ARCH:-$(dpkg --print-architecture 2>/dev/null || true)}"
INSTALL_GPU="${INSTALL_GPU:-1}"
INSTALL_DOCA_OFED="${INSTALL_DOCA_OFED:-1}"
NIC_CONFIG_DEVICES="${NIC_CONFIG_DEVICES:-${NIC_DEV:-}}"
PTP_ROLE_DEFAULT="${PTP_ROLE_DEFAULT:-client}"
PTP_INTERFACE_DEFAULT="${PTP_INTERFACE_DEFAULT:-aerial00}"

# =============================================================================
# Derived host install values
# =============================================================================

NVIDIA_CUDA_BASE_URL="https://developer.download.nvidia.com/compute/cuda/"
NVIDIA_TESLA_DRIVER_BASE_URL="https://us.download.nvidia.com/tesla"

if [[ "$HOST_PLATFORM_SUPPORTED" == "1" ]]; then
    DOCA_DEB="doca-host_${DOCA_VERSION}-${DOCA_BUILD}-${UBUNTU_VERSION}_${ARCH}.deb"
    DOCA_URL="https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/${DOCA_DEB}"

    if [[ "$INSTALL_GPU" == "1" ]]; then
        GPU_DRIVER_FILE="NVIDIA-Linux-${GPU_DRIVER_ARCH}-${GPU_DRIVER_VERSION}.run"
        GPU_DRIVER_URL="${NVIDIA_TESLA_DRIVER_BASE_URL}/${GPU_DRIVER_VERSION}/${GPU_DRIVER_FILE}"
        GPU_DRIVER_DOWNLOAD_FILE="$GPU_DRIVER_FILE"

        _gdrcopy_label="${GDRCOPY_UBUNTU_VER^}"
        GDRDRV_FILE="gdrdrv-dkms_${GDRDRV_VERSION}_${ARCH}.${_gdrcopy_label}.deb"
        GDRDRV_URL="https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%20${GDRDRV_CUDA_VERSION}/${GDRCOPY_UBUNTU_VER}/${GPU_DRIVER_ARCH}/${GDRDRV_FILE}"
    fi

    BFB_FILE="${BFB_VERSION:+bf-fwbundle-${BFB_VERSION}.bfb}"
    BFB_URL="${BFB_URL:-https://content.mellanox.com/BlueField/FW-Bundle/${BFB_FILE}}"
fi

show_versions() {
    echo "Platform ID: $PLATFORM"
    echo "Platform DMI ID: $VERBOSE_PLATFORM_ID"
    echo ""
    echo "Common Versions:"
    echo "  Manifest:     $MANIFEST_VERSION"
    echo "  DOCA:         $DOCA_VERSION"
    echo "  DOCA Build:   $DOCA_BUILD"
    echo "  DOCA/OFED:    $DOCA_OFED_VERSION"
    echo "  Docker:       $DOCKER_VERSION"
    echo "  NCT:          ${NVIDIA_CONTAINER_TOOLKIT_VERSION}${NVIDIA_CONTAINER_TOOLKIT_VERSION_SUFFIX}"
    echo "  MFT:          $MFT_VERSION"
    echo "  MLNX FW:      $MLNX_FW_UPDATER_VERSION"
    echo "  GPU Driver:   $GPU_DRIVER_VERSION"
    echo "  CUDA:         $CUDA_VERSION"
    echo "  GDRCopy:      $GDRDRV_VERSION (CUDA $GDRDRV_CUDA_VERSION)"
    echo "  linuxptp:     $LINUXPTP_VERSION"
    echo ""
    echo "Aerial Container:"
    echo "  Version Tag:  AERIAL_VERSION_TAG=$AERIAL_VERSION_TAG"
    echo "  Repository:   AERIAL_REPO=$AERIAL_REPO"
    echo "  Image Name:   AERIAL_IMAGE_NAME=$AERIAL_IMAGE_NAME"
    echo "  Platform:     ${AERIAL_PLATFORM:-not set}"
    echo "  CUDA Image:   $CONTAINER_CUDA_IMAGE"
    echo "  Ubuntu:       $CONTAINER_UBUNTU_DISTRO"
    echo "  DOCA Ubuntu:  $CONTAINER_DOCA_UBUNTU_VERSION"
    echo "  TensorRT:     $TENSORRT_VERSION (CUDA $TENSORRT_CUDA_VERSION)"
    echo "  DOCA Aerial:  $DOCA_FOR_AERIAL_VERSION"
    echo "  LDPC Cubin:   $LDPC_DECODER_CUBIN_VERSION"
    echo ""

    if [[ "$HOST_PLATFORM_SUPPORTED" != "1" ]]; then
        echo "Host Install:"
        echo "  Unsupported platform; set PLATFORM to a supported host to show host install pins."
        return
    fi

    echo "Host Install:"
    echo "  Kernel:       $KERNEL_VERSION"
    echo "  ISOLCPUS:     ${ISOLCPUS:-not set}"
    echo "  PTP CPU aff:  ${PTP_CPU_AFFINITY:-<unpinned>}"
    echo "  Hugepages:    ${HUGEPAGES:-not set}"
    echo "  Ubuntu:       $UBUNTU_VERSION"
    echo "  Arch:         $ARCH"
    echo "  NIC device:   ${NIC_DEVICE_TYPE:-${NIC_DEV:-auto-detect}}"
    echo "  NIC firmware: ${NIC_FW_VERSION:-platform managed}"
    echo "  FH interfaces:${AERIAL_FH_INTERFACES:+ }${AERIAL_FH_INTERFACES:-auto-detect}"
    echo "  PTP role:     ${PTP_ROLE_DEFAULT}"
    echo "  PTP port:     ${PTP_INTERFACE_DEFAULT}"
    echo "  GPU install:  ${INSTALL_GPU}"
    echo "  Aerial build: ${AERIAL_BUILD_FLAGS:-<none>}"
    echo ""
    echo "Derived Host Install Values:"
    echo "  DOCA DEB:     $DOCA_DEB"
    echo "  DOCA URL:     $DOCA_URL"
    echo "  GPU File:     $GPU_DRIVER_FILE"
    echo "  GPU URL:      $GPU_DRIVER_URL"
    if [[ "$GPU_DRIVER_DOWNLOAD_FILE" == "$CUDA_RUN_FILE_NAME" ]]; then
        echo "  CUDA Run:     $CUDA_RUN_FILE_NAME (extract then run $GPU_DRIVER_FILE)"
    else
        echo "  CUDA Run:     not used (standalone driver download)"
    fi
    echo "  Download as:  $GPU_DRIVER_DOWNLOAD_FILE"
    echo "  GDRDRV File:  $GDRDRV_FILE"
    echo "  GDRDRV URL:   $GDRDRV_URL"
    echo "  BFB File:     ${BFB_FILE:-not set (non-Supermicro platform)}"
    echo "  NIC FW File:  ${NIC_FW_FILE:-not set}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    show_versions
fi
