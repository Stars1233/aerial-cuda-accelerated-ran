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
# install_aerial_kernel.sh
# Installs and configures the NVIDIA Aerial CUDA kernel with optimized settings
#
# Usage: ./install_aerial_kernel.sh [--dry-run] [--verbose] [--check] [--remove-other-kernels] [--isolcpus LIST]
#   --isolcpus LIST   CPU list for isolation. Defaults in versions.sh. Overridable via ISOLCPUS env or --isolcpus
#

# Source common functions and versions
_SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
[[ -f "$_SCRIPT_DIR/includes.sh" ]] && source "$_SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $_SCRIPT_DIR/includes.sh" >&2; exit 1; }

# Parse common arguments (--dry-run, --verbose) and populate REMAINING_ARGS
parse_common_args "$@"
verify_secure_boot_disabled

CHECK_ONLY=0
REMOVE_OTHER_ONLY=0
ISOLCPUS="${ISOLCPUS:-4-19}"

# Re-set positional parameters to REMAINING_ARGS for shift-based parsing for use with parse_common_args
set -- "${REMAINING_ARGS[@]}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --check) CHECK_ONLY=1; shift ;;
        --remove-other-kernels) REMOVE_OTHER_ONLY=1; shift ;;
        --isolcpus=*) ISOLCPUS="${1#*=}"; shift ;;
        --isolcpus) ISOLCPUS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${ISOLCPUS}" ]]; then
    echo "ERROR: ISOLCPUS must be non-empty (e.g. 4-19, 4-15, 2,4,5,7,8). Use --isolcpus=LIST or ISOLCPUS=LIST." >&2
    exit 1
fi

# Check if running kernel matches expected version
check_kernel_version() {
    local current_kernel
    current_kernel=$(uname -r)
    
    echo_and_log "[INFO] Kernel version check:"
    echo_and_log "  Expected: ${KERNEL_VERSION}"
    echo_and_log "  Running:  ${current_kernel}"
    
    if [[ $current_kernel == "$KERNEL_VERSION" ]]; then
        echo_and_log "[INFO] Kernel version OK"
        return 0
    else
        echo_and_log "[WARN] Kernel version mismatch"
        echo_and_log "[WARN] Reboot required to load kernel ${KERNEL_VERSION}"
        return 1
    fi
}

# Handle --check flag: just check kernel version and exit
if [[ $CHECK_ONLY -eq 1 ]]; then
    check_kernel_version
    exit $?
fi

# Check if kernel headers are already installed
check_existing_kernel() {
    local headers_pkg="linux-headers-${KERNEL_VERSION}"
    
    if dpkg -l "$headers_pkg" 2>/dev/null | grep -q "^ii"; then
        echo_and_log "[WARN] Kernel headers already installed: $headers_pkg"
        echo ""
        echo "Options:"
        echo "  [p] Purge existing kernel packages and reinstall"
        echo "  [c] Continue with installation anyway"
        echo "  [q] Quit"
        echo ""
        read -p "Choice [p/c/q]: " choice
        echo_and_log "Choice: $choice"
        
        case "$choice" in
            p|P)
                echo_and_log "[INFO] Purging existing kernel packages..."
                execute_or_die "sudo apt purge -y linux-image-${KERNEL_VERSION} linux-headers-${KERNEL_VERSION} linux-modules-${KERNEL_VERSION}"
                ;;
            c|C)
                echo_and_log "[INFO] Continuing with installation..."
                ;;
            q|Q|*)
                echo_and_log "[INFO] Exiting."
                exit 0
                ;;
        esac
    fi
}

remove_other_kernels_and_headers() {
    local RUNNING_KERNEL
    RUNNING_KERNEL=$(uname -r)
    # Never purge the currently running kernel or the Aerial kernel we're installing
    # Purge linux-headers for all other kernels
    local other_headers
    # Use tr to get space-separated list so eval in execute_or_die doesn't run newlines as new commands
    other_headers=$(dpkg -l 'linux-headers-*' 2>/dev/null | awk '/^ii/ || /^rc/ || /^rF/ {print $2}' | grep -v "^linux-headers-${KERNEL_VERSION}" | grep -v "^linux-headers-${RUNNING_KERNEL}" | tr '\n' ' ' || true)
    if [[ -n "${other_headers// /}" ]]; then
        echo_and_log "[INFO] Purging other kernel headers: $other_headers"
        execute "sudo apt purge -y $other_headers"
    fi

    # For each linux-image-* we purge, also add linux-image-unsigned-* (unsigned may appear only during removal)
    local other_images other_unsigned pkg suffix
    other_images=$(dpkg -l 'linux-image-*' 2>/dev/null | awk '/^ii/ || /^rc/ || /^rF/ {print $2}' | grep -v "^linux-image-${KERNEL_VERSION}" | grep -v "^linux-image-${RUNNING_KERNEL}" | tr '\n' ' ' || true)
    other_unsigned=""
    for pkg in $other_images; do
        if [[ $pkg == linux-image-* ]]; then
            suffix="${pkg#linux-image-}"
            if dpkg-query -W "linux-image-unsigned-${suffix}" &> /dev/null; then
                echo_and_log "[INFO] Adding linux-image-unsigned-${suffix} to purge list"
                other_unsigned="${other_unsigned} linux-image-unsigned-${suffix}"
            else
                echo_and_log "[INFO] linux-image-unsigned-${suffix} not found, not adding to purge list"
            fi
        fi
    done
    other_images="${other_images} ${other_unsigned}"
    if [[ -n "${other_images// /}" ]]; then
        echo_and_log "[INFO] Purging other kernel images: $other_images"
        execute_or_die "sudo apt purge -y $other_images"
    fi
    autoremove_until_clean --purge
}

# Handle --remove-other-kernels: purge other kernel images/headers (keep running + Aerial kernel), then exit
if [[ $REMOVE_OTHER_ONLY -eq 1 ]]; then
    echo_and_log "[INFO] Removing other kernels and headers (keeping running kernel and ${KERNEL_VERSION})..."
    remove_other_kernels_and_headers
    echo_and_log "[INFO] Done."
    exit 0
fi

# Check for existing installation
check_existing_kernel
add_user_to_docker

# Disable automatic updates to prevent interference with kernel installation
echo_and_log "[INFO] Disabling automatic updates..."
write_file /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Unattended-Upgrade "0";
EOF
# Disable fwupd-refresh timer to prevent fwupdmgr from auto-updating firmware
execute "sudo systemctl mask fwupd-refresh.timer"

# Stop and disable the timers first (they trigger the services); then stop and mask the services
execute "sudo systemctl stop apt-daily.service apt-daily-upgrade.service apt-daily.timer apt-daily-upgrade.timer"
execute "sudo systemctl mask apt-daily.service apt-daily-upgrade.service apt-daily.timer apt-daily-upgrade.timer"
execute "sudo systemctl disable apt-daily.service apt-daily-upgrade.service apt-daily.timer apt-daily-upgrade.timer"
execute "sudo systemctl daemon-reload"

# Install the kernel and its suggested packages (headers, modules-extra, perf, nvidia-tools)
execute_or_die "sudo apt update"
remove_other_kernels_and_headers
execute_or_die "sudo apt install -y --install-suggests linux-image-${KERNEL_VERSION}"
autoremove_until_clean

# Update GRUB to generate menu entries
execute_or_die "sudo update-grub"

# Find GRUB menu entries for this kernel (exclude recovery)
echo_and_log "[INFO] Finding GRUB menu entries for kernel ${KERNEL_VERSION}..."

# Get the advanced submenu ID and kernel menuentry (requires sudo to read /boot/grub/grub.cfg)
if [[ $DRYRUN -eq 1 ]]; then
    run_sudo_grep=""
    read -t 5 -p "Run sudo to read GRUB config? [y/N]: " run_sudo_grep
    read_rc=$?
    # Timeout (exit >128) or "n" means skip
    [[ $read_rc -gt 128 ]] && run_sudo_grep="n"
    run_sudo_grep="${run_sudo_grep:-n}"
    if [[ "${run_sudo_grep,,}" == "n" ]]; then
        [[ $read_rc -gt 128 ]] && echo # Handle no newline on timeout
        echo -e "\n[DRY-RUN] Using placeholder GRUB entries for 6.17.0-1014-nvidia\n"
        GRUB_SUBMENU="gnulinux-advanced-b937ead4-f4cb-42b9-a72d-a7a7e8b9d5b4"
        GRUB_MENUENTRY="gnulinux-6.17.0-1014-nvidia-advanced-b937ead4-f4cb-42b9-a72d-a7a7e8b9d5b4"
    else
        echo_and_log "[DRY-RUN] Running sudo grep to get GRUB menu entries for kernel ${KERNEL_VERSION}"
        GRUB_SUBMENU=$(sudo grep "submenu.*gnulinux-advanced" /boot/grub/grub.cfg | head -1 | grep -oE "'[^']+'" | tail -1 | tr -d "'")
        GRUB_MENUENTRY=$(sudo grep "menuentry.*${KERNEL_VERSION}" /boot/grub/grub.cfg | grep -v recovery | head -1 | grep -oE "'[^']+'" | tail -1 | tr -d "'")
        if [[ -z $GRUB_SUBMENU || -z $GRUB_MENUENTRY ]]; then
            echo_and_log "[DRY-RUN] [WARNING] Could not find GRUB entries for kernel ${KERNEL_VERSION}. This is expected if the kernel has not yet been installed."
            echo_and_log -e "[DRY-RUN] Using placeholder GRUB entries for 6.17.0-1014-nvidia"
            GRUB_SUBMENU="gnulinux-advanced-b937ead4-f4cb-42b9-a72d-a7a7e8b9d5b4"
            GRUB_MENUENTRY="gnulinux-6.17.0-1014-nvidia-advanced-b937ead4-f4cb-42b9-a72d-a7a7e8b9d5b4"
        fi
    fi
else
    GRUB_SUBMENU=$(sudo grep "submenu.*gnulinux-advanced" /boot/grub/grub.cfg | head -1 | grep -oE "'[^']+'" | tail -1 | tr -d "'")
    GRUB_MENUENTRY=$(sudo grep "menuentry.*${KERNEL_VERSION}" /boot/grub/grub.cfg | grep -v recovery | head -1 | grep -oE "'[^']+'" | tail -1 | tr -d "'")
fi

if [[ $DRYRUN -eq 0 && (-z $GRUB_SUBMENU || -z $GRUB_MENUENTRY) ]]; then
    echo_and_log "[ERROR] Could not find GRUB entries for kernel ${KERNEL_VERSION}"
    echo_and_log "[ERROR] GRUB_SUBMENU: ${GRUB_SUBMENU:-not found}"
    echo_and_log "[ERROR] GRUB_MENUENTRY: ${GRUB_MENUENTRY:-not found}"
    echo_and_log "[ERROR] Check /boot/grub/grub.cfg manually"
    FAILED=1
    exit 1
else
    GRUB_DEFAULT_VALUE="${GRUB_SUBMENU}>${GRUB_MENUENTRY}"
    
    echo_and_log "[INFO] Setting GRUB_DEFAULT to: ${GRUB_DEFAULT_VALUE}"
    execute_or_die "sudo sed -i 's|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"${GRUB_DEFAULT_VALUE}\"|' /etc/default/grub"
fi

# Remove iommu.passthrough=y from grub config files if present
FOUND_FILES=$(grep -rls "iommu.passthrough=y" /etc/default/grub* 2>/dev/null || true)
for file in $FOUND_FILES; do
    execute_or_die "sudo sed -i 's/ iommu.passthrough=y//g; s/iommu.passthrough=y //g; s/iommu.passthrough=y//g' $file"
done

# Create kernel command-line configuration (ISOLCPUS used in isolcpus=, nohz_full=, rcu_nocbs=)
_HUGEPAGES="${HUGEPAGES:-24}"

if [[ $PLATFORM == "Supermicro_ARS-111GL-NHR" ]]; then
    # GH200: write GRUB_CMDLINE_LINUX to /etc/default/grub.d/cmdline.cfg
    # and serial/terminal settings to /etc/default/grub.d/menu.cfg
    execute_or_die "sudo mkdir -p /etc/default/grub.d"
    execute_or_die "cat <<EOF | sudo tee /etc/default/grub.d/cmdline.cfg
GRUB_CMDLINE_LINUX=\"\$GRUB_CMDLINE_LINUX pci=realloc=off pci=pcie_bus_safe default_hugepagesz=512M hugepagesz=512M hugepages=${_HUGEPAGES} tsc=reliable processor.max_cstate=0 audit=0 idle=poll rcu_nocb_poll nosoftlockup irqaffinity=0 isolcpus=managed_irq,domain,${ISOLCPUS} nohz_full=${ISOLCPUS} rcu_nocbs=${ISOLCPUS} earlycon module_blacklist=nouveau acpi_power_meter.force_cap_on=y numa_balancing=disable init_on_alloc=0 preempt=none\"
EOF"
    write_file /etc/default/grub.d/menu.cfg << 'MENU_EOF'
GRUB_SERIAL_COMMAND="serial --speed=115200 --unit=0 --word=8 --parity=no --stop=1"
GRUB_TERMINAL="serial console"
MENU_EOF
else
    execute_or_die "cat <<EOF | sudo tee /etc/default/grub.d/cmdline.cfg
GRUB_CMDLINE_LINUX=\"\$GRUB_CMDLINE_LINUX pci=realloc=off default_hugepagesz=1G hugepagesz=1G hugepages=${_HUGEPAGES} tsc=reliable processor.max_cstate=0 audit=0 idle=poll rcu_nocb_poll nosoftlockup irqaffinity=0-3 kthread_cpus=0-3 isolcpus=managed_irq,domain,${ISOLCPUS} nohz_full=${ISOLCPUS} rcu_nocbs=${ISOLCPUS} earlycon module_blacklist=nouveau acpi_power_meter.force_cap_on=y init_on_alloc=0 preempt=none\"
EOF"
fi

# Update GRUB
execute_or_die "sudo update-grub"

echo ""
echo "============================================"
echo_and_log "Installation complete!"
echo "============================================"
echo ""

# Check current kernel status
check_kernel_version

echo_and_log "To activate the new kernel, run: sudo reboot"
echo_and_log "After reboot, verify with: ./install_aerial_kernel.sh --check"
