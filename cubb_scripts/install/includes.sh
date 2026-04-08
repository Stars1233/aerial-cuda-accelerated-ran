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
# includes.sh - Common functions and variables for install scripts
#
# Source this file at the beginning of install scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/includes.sh"
#

# Common variables
DRYRUN=0
VERBOSE=0
SHOW_TIME=${SHOW_TIME:-0}  # Can be set via environment variable or --show-time flag
OVERRIDE_SB=${OVERRIDE_SB:-0}
FAILED=0
SYSTEM_STATUS_ONLY=0
# Logging: install_* scripts log here by default. Set via env (LOGFILE=) or --log=FILE in parse_common_args.
# Set LOGFILE= to disable logging. Build commands (make build_*) use build-<target>.log instead.
LOGFILE=${LOGFILE:-/var/log/aerial-installer.log}
# Set by log_init: 1 if we need sudo to write to LOGFILE, 0 if not (so tee vs sudo tee).
LOG_NEEDS_SUDO=1

# Source versions.sh (platform-specific and common versions). One-way: includes.sh -> versions.sh only.
_INCLUDES_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
[[ -f "$_INCLUDES_DIR/versions.sh" ]] && source "$_INCLUDES_DIR/versions.sh" || { echo "ERROR: versions.sh not found: $_INCLUDES_DIR/versions.sh" >&2; exit 1; }

# Parse common arguments and populate REMAINING_ARGS with unhandled arguments
# Usage: parse_common_args "$@"
#        Then use ${REMAINING_ARGS[@]} for script-specific argument parsing
parse_common_args() {
    REMAINING_ARGS=()
    for arg in "$@"; do
        case $arg in
            --dry-run) DRYRUN=1 ;;
            --verbose) VERBOSE=1 ;;
            --show-time) SHOW_TIME=1 ;;
            --override-sb) OVERRIDE_SB=1 ;;
            --system-status) SYSTEM_STATUS_ONLY=1 ;;
            --log=*) LOGFILE="${arg#--log=}" ;;
            *) REMAINING_ARGS+=("$arg") ;;
        esac
    done
    # With --dry-run, use dry-run.log in cwd so logging does not require sudo
    if [[ $DRYRUN -eq 1 ]] && [[ "$LOGFILE" == "/var/log/aerial-installer.log" ]]; then
        LOGFILE="dry-run.log"
    fi
    log_init
    if [[ $SYSTEM_STATUS_ONLY -eq 1 ]]; then
        get_system_status
        exit 0
    fi
    check_and_install_dependencies
}

# Docker wrapper. init_docker_cmd is called when this file is sourced; then use "docker" as usual.
# Only defines docker() when a prefix is needed (sg or sudo); otherwise the real docker is used.
init_docker_cmd() {
    # Only initialize docker command once
    [[ -n "${INIT_DOCKER_CMD_DONE:-}" ]] && return 0

    local current_user="${SUDO_USER:-$USER}"
    if command docker ps &>/dev/null; then
        # In case this is the second time through, let docker be docker.
        unset -f docker 2>/dev/null || true
        INIT_DOCKER_CMD_DONE=1
        export INIT_DOCKER_CMD_DONE
        return
    fi
    if id -nG "$current_user" 2>/dev/null | grep -qw docker; then
        # User is in docker group but docker isn't working in this session (e.g. newgrp not done, or not logged back in). Use sg.
        docker() {
            local inner
            inner=$(printf '%q ' docker "$@")
            sg docker -c "$inner"
        }
        export -f docker 2>/dev/null || true
        INIT_DOCKER_CMD_DONE=1
        export INIT_DOCKER_CMD_DONE
        return
    fi
    if ! command -v docker &>/dev/null; then
        # Docker not installed yet (e.g. install_drivers.sh will install it). Defer so sourcing does not abort.
        # Lazy-init: first use of docker will re-run init_docker_cmd, then run the real docker.
        docker() {
            init_docker_cmd
            docker "$@"
        }
        export -f docker 2>/dev/null || true
        return
    fi
    # Docker is installed but user cannot run it (not in docker group).
    echo "User $current_user is not in the 'docker' group and cannot run docker commands."
    echo "ERROR: User $current_user is not in the 'docker' group and cannot run docker commands." >&2
    exit 1
}

# Add current user to docker group if not already a member. Uses SUDO_USER or USER.
add_user_to_docker() {
    local current_user="${SUDO_USER:-$USER}"
    if id -nG "$current_user" 2>/dev/null | grep -qw docker; then
        echo_and_log "[INFO] User '${current_user}' already in docker group"
    else
        echo_and_log "[INFO] Adding user '${current_user}' to docker group..."
        execute sudo usermod -aG docker "$current_user"
        echo_and_log "[INFO] NOTE: Log out and back in for docker group membership to take effect"
    fi
}

# Wait for apt/dpkg lock to be released before running apt or dpkg.
# Uses fuser to detect processes holding /var/lib/dpkg/lock-frontend, lock, or archives/lock.
# Usage: wait_for_apt_lock [timeout_sec]
#   timeout_sec  Max seconds to wait (default 60). Exit 1 on timeout, 0 when lock is free.
#   In dry-run mode returns 0 immediately without waiting.
check_and_release_apt_lock() {
    local timeout_sec="${1:-60}"
    local elapsed=0
    local pid
    local pids
    local lock_files="/var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock"
    if [[ $DRYRUN -eq 1 ]]; then
        execute sudo fuser $lock_files 2>/dev/null
        return 0
    fi
    while true; do
        # fuser returns 0 when any process is using the lock files. Use sudo so we see root-held locks (e.g. apt).
        if ! sudo fuser $lock_files >/dev/null 2>&1; then
            return 0
        fi
        if [[ $elapsed -ge $timeout_sec ]]; then
            echo_and_log "[WARN] Timeout waiting for apt/dpkg lock after ${timeout_sec}s. Killing processes holding the lock..."
            pids=$(sudo fuser $lock_files 2>/dev/null | tr -s ' \n' '\n' | grep -oE '[0-9]+' | sort -u)
            for pid in $pids; do
                echo_and_log "[INFO] Killing PID $pid ($(ps -p "$pid" -o cmd= 2>/dev/null || echo '?'))"
                sudo kill -9 "$pid" 2>/dev/null || true
            done
            sleep 2
            if sudo fuser $lock_files >/dev/null 2>&1; then
                echo_and_log "[ERROR] Lock still held after killing processes. Cannot proceed."
                return 1
            fi
            echo_and_log "[INFO] Lock released. You may need to run 'sudo dpkg --configure -a' if package state was interrupted."
            return 0
        fi
        echo_and_log "[INFO] Waiting for apt/dpkg lock... ${elapsed}/${timeout_sec}s elapsed"
        for lock_file in $lock_files; do
            pid=$(sudo fuser $lock_file 2>/dev/null)
            if [[ -n "$pid" ]]; then
                echo_and_log "[INFO] $lock_file held by: $pid: $(ps -p $pid -o cmd=)"
            fi
        done
        sleep 5
        elapsed=$((elapsed + 5))
    done
}

# Append stdin to LOGFILE (and pass through to stdout). Uses sudo or plain tee based on LOG_NEEDS_SUDO.
# Usage: echo "..." | log_command   or   some_command 2>&1 | log_command. No-op if LOGFILE is empty.
log_command() {
    [[ -z "$LOGFILE" ]] && { cat; return 0; }
    if [[ $LOG_NEEDS_SUDO -eq 1 ]]; then
        sudo tee -a "$LOGFILE"
    else
        tee -a "$LOGFILE"
    fi
}

# Echo arguments to stdout and append to LOGFILE. Convenience for "echo ... | log_command".
# Usage: echo_and_log "message"   or   echo_and_log "[INFO] something"
echo_and_log() {
    echo "$@" | log_command
}

# Write a log header: script name and UTC timestamp. Uses BASH_SOURCE to find the top-level script (the one invoked).
# Creates log dir and file so -w check passes for user-writable paths (e.g. build-*.log). Called from parse_common_args.
log_init() {
    [[ -z "$LOGFILE" ]] && return 0
    local log_dir
    log_dir=$(dirname "$LOGFILE")
    if [[ -n "$log_dir" && "$log_dir" != "." ]]; then
        mkdir -p "$log_dir" 2>/dev/null || true
    fi
    # -w fails if the file doesn't exist
    touch "$LOGFILE" 2>/dev/null || true
    if [[ -w $LOGFILE ]]; then
        LOG_NEEDS_SUDO=0
    else
        echo "[INFO] LOGFILE: $LOGFILE requires sudo to write"
        LOG_NEEDS_SUDO=1
    fi
    # In dry-run skip writing the header so we don't create log content; LOG_NEEDS_SUDO is already set above.
    [[ $DRYRUN -eq 1 ]] && return 0
    # BASH_SOURCE[1] is the direct caller (often includes.sh); use last element = top-level script that was invoked
    local script_name
    script_name=$(basename "${BASH_SOURCE[${#BASH_SOURCE[@]}-1]:-$0}")
    local line="=== $script_name @ $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
    echo "$line" | log_command
}

JOURNAL_FIRST_LINES=30
JOURNAL_LAST_LINES=30

# Collect system status for debugging: journalctl -k (kernel), journal (this boot, UTC, limited per unit),
# .service unit states, and loaded modules.
get_system_status() {
    echo "[Capturing system status to $LOGFILE ...]" | log_command
    {
        echo ""
        echo "=== SYSTEM STATUS @ $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
        echo "--- loaded modules (lsmod) ---"
        lsmod 2>&1
        echo "--- journalctl -k (kernel, this boot, UTC) ---"
        TZ=UTC journalctl -k -b --no-pager 2>&1
        echo "--- journalctl (this boot, UTC; per-unit first/last lines only) ---"
        local unit
        for unit in $(TZ=UTC journalctl -b -F _SYSTEMD_UNIT 2>/dev/null | sort -u); do
            case "$unit" in
                session*.scope) continue ;;
                *)
                    echo "--- $unit (first ${JOURNAL_FIRST_LINES} / last ${JOURNAL_LAST_LINES} lines) ---"
                    local journal_out
                    journal_out=$(TZ=UTC journalctl -b -u "$unit" --no-pager 2>&1)
                    echo "$journal_out" | head -n "$JOURNAL_FIRST_LINES"
                    echo "$journal_out" | tail -n "$JOURNAL_LAST_LINES"
                    ;;
            esac
        done
        echo "--- systemctl list-units --all (type=service) ---"
        systemctl list-units --all --type=service --no-pager 2>&1
        echo "--- systemctl status (failed .service units) ---"
        systemctl list-units --state=failed --type=service --no-pager 2>&1
        local u
        for u in $(systemctl list-units --state=failed --type=service --no-pager --plain --no-legend 2>/dev/null | awk '{print $1}' | grep -v '^$'); do
            systemctl status "$u" --no-pager 2>&1
        done
    } | log_command > /dev/null
}

# Run apt autoremove; on "Unmet dependencies", parse " pkg : Depends" from output,
# purge those packages with apt purge -y (all at once), then repeat (max 5 rounds).
# Usage: autoremove_until_clean [--purge]
#   --purge  use "apt --purge autoremove -y"; otherwise "apt autoremove -y"
autoremove_until_clean() {
    local use_purge=0
    [[ "${1:-}" == "--purge" ]] && use_purge=1
    local round=1 max_rounds=5 out broken_pkgs
    while [[ $round -le $max_rounds ]]; do
        echo_and_log "[INFO] Autoremove round $round/$max_rounds, this may take a moment..."
        if [[ $DRYRUN -eq 1 ]]; then
            [[ $use_purge -eq 1 ]] && echo_and_log "[DRY-RUN] sudo apt --purge autoremove -y" || echo_and_log "[DRY-RUN] sudo apt autoremove -y"
            break
        fi
        if [[ $use_purge -eq 1 ]]; then
            out=$(sudo apt --purge autoremove -y 2>&1) || true
        else
            out=$(sudo apt autoremove -y 2>&1) || true
        fi
        echo_and_log "$out"
        if echo "$out" | grep -q "Unmet dependencies"; then
            broken_pkgs=$(echo "$out" | sed -n 's/^[[:space:]]*\([^[:space:]:]*\)[[:space:]]*:[[:space:]]*Depends.*/\1/p' | sort -u || true)
            if [[ -z "$broken_pkgs" ]]; then
                echo_and_log "[WARN] Autoremove reported unmet dependencies but could not parse package names."
                break
            fi
            echo_and_log "[INFO] Purging packages with unmet dependencies: $broken_pkgs"
            execute "sudo apt purge -y $broken_pkgs"
        else
            echo_and_log "[INFO] Autoremove completed successfully."
            break
        fi
        round=$((round + 1))
    done
}

# Dependency list (apt packages; linux-headers-$(uname -r) and yq are handled separately)
INSTALL_DEPS_APT=(
    build-essential
    dkms
    unzip
    pv
    apt-utils
    net-tools
    wget
    ca-certificates
    curl
    jq
    gnupg
    ethtool
    tcpdump
)

# Check which dependencies are already installed and which need installing.
check_and_install_dependencies() {
    check_and_release_apt_lock 60 || { echo "[ERROR] Cannot proceed: apt/dpkg lock held. Try again later." >&2; exit 1; }
    local pkg linux_headers_pkg
    linux_headers_pkg="linux-headers-$(uname -r)"
    FULFILLED_APT=()
    TO_INSTALL_APT=()

    for pkg in "${INSTALL_DEPS_APT[@]}" "$linux_headers_pkg"; do
        if dpkg -l "$pkg" 2>/dev/null | grep -q '^ii'; then
            FULFILLED_APT+=("$pkg")
        else
            TO_INSTALL_APT+=("$pkg")
        fi
    done

    local installed_yq_version
    if command -v yq &>/dev/null; then
        installed_yq_version=$(yq --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if [[ -n "${YQ_VERSION:-}" && "${installed_yq_version:-}" != "${YQ_VERSION}" ]]; then
            echo_and_log "[INFO] yq installed version (${installed_yq_version}) does not match desired version (${YQ_VERSION}), will reinstall"
            YQ_NEEDED=1
        else
            YQ_NEEDED=0
        fi
    else
        YQ_NEEDED=1
    fi

    # Fix for bug found in https://forums.developer.nvidia.com/t/dgx-spark-apt-update-failure-expkeysig/354539
    ai_workbench_key=/usr/share/keyrings/ai-workbench-desktop-key.gpg
    if [[ -f "$ai_workbench_key" ]] && gpg --show-keys "$ai_workbench_key" 2>/dev/null | grep -q '\[expired:'; then
        echo_and_log "[WARNING] NVIDIA AI Workbench key is expired, reinstalling. This will require sudo."
        execute_or_die "curl -fsSL https://workbench.download.nvidia.com/stable/linux/gpgkey | gpg --dearmor | sudo tee $ai_workbench_key > /dev/null"
        execute_or_die "gpg --show-keys $ai_workbench_key"
    fi

    if [[ ${#FULFILLED_APT[@]} -gt 0 || $YQ_NEEDED -eq 0 ]]; then
        local fulfilled_list="${FULFILLED_APT[*]}"
        [[ $YQ_NEEDED -eq 0 ]] && fulfilled_list="${fulfilled_list:+$fulfilled_list }yq"
        echo_and_log "[INFO] Dependencies already fulfilled: ${fulfilled_list}"
    fi
    if [[ ${#TO_INSTALL_APT[@]} -gt 0 || $YQ_NEEDED -eq 1 ]]; then
        local to_install_list="${TO_INSTALL_APT[*]}"
        [[ $YQ_NEEDED -eq 1 ]] && to_install_list="${to_install_list:+$to_install_list }yq"
        echo_and_log "[INFO] Dependencies to install: ${to_install_list}"
    fi

    if [[ ${#TO_INSTALL_APT[@]} -gt 0 ]]; then
        echo_and_log "[INFO] Installing dependency packages..."
        execute "sudo apt update && sudo apt install -y --no-install-recommends ${TO_INSTALL_APT[*]}"
    fi

    if [[ $YQ_NEEDED -eq 1 ]]; then
        echo_and_log "[INFO] Installing yq ${YQ_VERSION} (${ARCH})..."
        execute sudo "wget -nv https://github.com/mikefarah/yq/releases/download/v${YQ_VERSION}/yq_linux_${ARCH} -O /usr/bin/yq"
        execute sudo chmod +x /usr/bin/yq
        if [[ $DRYRUN -eq 0 ]] && command -v yq &>/dev/null; then
            local actual_version
            actual_version=$(yq --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
            if [[ "${actual_version}" == "${YQ_VERSION}" ]]; then
                echo_and_log "[INFO] Verified yq updated to version: ${actual_version}"
            else
                echo_and_log "[WARN] yq version mismatch: expected ${YQ_VERSION}, got ${actual_version}"
            fi
        fi
    fi
}

# Execute a command with dry-run and verbose support
# Sets FAILED=1 on error but does not exit (allows script to continue)
function execute() {
    if [[ $DRYRUN -eq 1 ]]; then
        echo "[DRY-RUN] $@" | log_command
    else
        echo "Executing: $@" | log_command
        local ret_val
        if [[ $SHOW_TIME -eq 1 ]]; then
            ( time eval "$@" ) 2>&1 | log_command
        else
            ( eval "$@" ) 2>&1 | log_command
        fi
        ret_val=${PIPESTATUS[0]}
        if [[ $ret_val -ne 0 ]]; then
            echo "[ERROR] Command failed with exit code $ret_val" | log_command
            FAILED=1
        fi
    fi
}

# Execute a command and exit immediately on failure
# Use this for critical commands where continuing doesn't make sense
# On failure, get_system_status() is called to capture journalctl, service status, and loaded modules.
function execute_or_die() {
    if [[ $DRYRUN -eq 1 ]]; then
        echo "[DRY-RUN] $@" | log_command
    else
        echo "Executing: $@" | log_command
        local ret_val
        if [[ $SHOW_TIME -eq 1 ]]; then
            ( time eval "$@" ) 2>&1 | log_command
        else
            ( eval "$@" ) 2>&1 | log_command
        fi
        ret_val=${PIPESTATUS[0]}
        if [[ $ret_val -ne 0 ]]; then
            echo "[ERROR] Command failed with exit code $ret_val" | log_command
            get_system_status
            exit $ret_val
        fi
    fi
}

# Execute a command with retries; on final failure exit (like execute_or_die).
# Usage: execute_retry_or_die max_tries "command"
function execute_retry_or_die() {
    local max_tries="$1"
    local cmd="$2"
    if [[ $DRYRUN -eq 1 ]]; then
        echo "[DRY-RUN] (retry up to ${max_tries}x) $cmd" | log_command
        return 0
    fi
    local attempt=1
    local ret_val
    while [[ $attempt -le $max_tries ]]; do
        echo "Executing (attempt $attempt/$max_tries): $cmd" | log_command
        if [[ $SHOW_TIME -eq 1 ]]; then
            ( time eval "$cmd" ) 2>&1 | log_command
        else
            ( eval "$cmd" ) 2>&1 | log_command
        fi
        ret_val=${PIPESTATUS[0]}
        if [[ $ret_val -eq 0 ]]; then
            return 0
        fi
        echo "[WARN] Command failed with exit code $ret_val (attempt $attempt/$max_tries)" | log_command
        attempt=$((attempt + 1))
    done
    echo "[ERROR] Command failed after $max_tries attempts" | log_command
    get_system_status
    exit $ret_val
}

# Execute a command, exit on failure, and print its output so the caller can capture it.
# Usage: out=$(capture_or_die "cmd args"). Output is also logged via log_command.
# DRYRUN: command is not run; logs "[DRY-RUN] ..." and returns success with no output.
function capture_or_die() {
    if [[ $DRYRUN -eq 1 ]]; then
        echo "[DRY-RUN] $@" | log_command
        return 0
    fi
    local output
    # Run with bash -c "$1" so the whole command string (pipes, redirections) is executed by a shell.
    if [[ $SHOW_TIME -eq 1 ]]; then
        output=$( ( time bash -c "$1" ) 2>&1 )
    else
        output=$( bash -c "$1" 2>&1 )
    fi
    local ret_val=$?
    # Log to file only so captured output is not duplicated.
    echo "$output" | log_command > /dev/null
    if [[ $ret_val -ne 0 ]]; then
        echo "[ERROR] Command failed with exit code $ret_val" | log_command
        exit $ret_val
    fi
    printf '%s' "$output"
}

# Write content to a file with sudo, respecting DRYRUN
# Usage: write_file "/path/to/file" "content"
#        write_file "/path/to/file" < heredoc
function write_file() {
    local dest_file="$1"
    local content

    if [[ -n $2 ]]; then
        content="$2"
    else
        content=$(cat)
    fi

    if [[ $VERBOSE -eq 1 ]]; then
        echo "Writing to: $dest_file"
    fi

    if [[ $DRYRUN -eq 1 ]]; then
        echo "[DRY-RUN] Would write to $dest_file"
    else
        echo "$content" | sudo tee "$dest_file" > /dev/null
        echo "  [write_file] $dest_file ($(echo -n "$content" | wc -c) bytes)" | log_command
    fi
}

# Copy a file with sudo, respecting DRYRUN
# Usage: copy_file "/src" "/dest"
function copy_file() {
    local src_file="$1"
    local dest_file="$2"

    if [[ ! -f $src_file ]]; then
        echo "[ERROR] Source file not found: $src_file" | log_command
        FAILED=1
        return 1
    fi

    if [[ $VERBOSE -eq 1 ]]; then
        echo "Copying: $src_file -> $dest_file"
    fi

    if [[ $DRYRUN -eq 1 ]]; then
        echo "[DRY-RUN] Would copy $src_file to $dest_file"
    else
        sudo cp "$src_file" "$dest_file"
        sudo chmod +x "$dest_file"
        echo "  [copy_file] $src_file -> $dest_file" | log_command
    fi
}

# Check if Secure Boot is disabled
# Returns 0 if disabled, 1 if enabled
# Exits with code 2 if cannot determine
function is_secure_boot_disabled() {
    if ! command -v mokutil &> /dev/null; then
        echo "[WARNING] mokutil not found, cannot determine Secure Boot state" | log_command
        exit 2
    fi

    local output
    output=$(mokutil --sb-state 2>&1)

    if [[ $VERBOSE -eq 1 ]]; then
        echo "mokutil output: $output"
    fi

    # grep returns 0 when it finds a match
    if echo "$output" | grep -qi "SecureBoot disabled"; then
        if [[ $VERBOSE -eq 1 ]]; then
            echo "Secure Boot is disabled"
        fi
        return 0  # Secure Boot is disabled
    elif echo "$output" | grep -qi "SecureBoot enabled"; then
        if [[ $VERBOSE -eq 1 ]]; then
            echo "Secure Boot is enabled"
        fi
        return 1  # Secure Boot is enabled
    else
        echo "[WARNING] Cannot determine Secure Boot state from mokutil output" | log_command
        exit 2
    fi
}

# Verify that Secure Boot is disabled; exit if enabled unless override flag is set
function verify_secure_boot_disabled() {
    if is_secure_boot_disabled; then
        return
    fi

    if [[ $OVERRIDE_SB -eq 1 ]]; then
        echo "[WARNING] Secure Boot is enabled but --override-sb flag is set, proceeding anyway" | log_command
        return
    fi

    echo "[ERROR] Secure Boot is enabled. Please disable Secure Boot in BIOS or use --override-sb to continue anyway -- this is not supported and strongly discouraged." | log_command
    exit 1
}
