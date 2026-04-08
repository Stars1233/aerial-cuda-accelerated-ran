# Server Version Check

A suite of tools for verifying and managing software versions across Aerial 5G cluster nodes (DU and RU systems).

## Overview

This directory contains three main tools:

1. **`server_version_check.py`** - Verify versions on a single node against a manifest file, or generate a new manifest
2. **`server_version_check_all.py`** - Run version checks across multiple cluster nodes
3. **Manifest CSV files** - Define expected software versions for different system configurations

## Features

- **Automated Version Detection**: Checks BMC, BIOS, kernel, GPU drivers, CUDA, and more
- **Service Health Checks**: Validates PTP sync status, system services
- **System Type Auto-Detection**: Automatically identifies DU vs RU systems
- **Manifest Generation**: Create version manifests from running systems
- **Cluster-Wide Checks**: Verify versions across multiple nodes simultaneously
- **Detailed Reporting**: Clear pass/fail status with detailed failure information

## Prerequisites

- Python 3.6+
- `sudo` access (required for some version checks and service health checks)
- `sshpass` (for cluster-wide checks): `sudo apt install sshpass`

### Password File (for cluster-wide checks)

It is assumed your SSH password is stored in `~/aerial_pw` with secure permissions (chmod 600).

Load it into the environment variable:
```bash
export SSHPASS=$(cat ~/aerial_pw)
```

## Quick Start

### 1. Check a Single Server

Run version check against a manifest file:

```bash
sudo ./server_version_check.py manifest_cg1_r750_25.3.csv
```

**Example Output:**

```
Detected System Type: DU

========================================================================================================================
Component              Expected                          Actual                            Optional   Status
========================================================================================================================
BMC                    1.03                              1.03                              No         PASS
BIOS                   2.0c                              2.0c                              No         PASS
VBIOS                  96.00.8D.00.03                    96.00.8D.00.03                    No         PASS
Ubuntu                 22.04                             22.04                             No         PASS
Kernel                 6.8.0-1025-nvidia-64k             6.8.0-1025-nvidia-64k             No         PASS
GPU Driver             575.57.08                         575.57.08                         No         PASS
CUDA                   12.9.0                            12.9.0                            No         PASS
DOCA OFED              25.04-0.6.1                       25.04-0.6.1                       No         PASS
PTP4L                  4.2                               4.2                               No         PASS
GDRCOPY                2.5                               2.5                               No         PASS
BlueField FW           32.45.1020                        32.45.1020                        No         PASS
Cmdline                BOOT_IMAGE=/vmlinuz-6.8.0-1025... BOOT_IMAGE=/vmlinuz-6.8.0-1025... No         PASS
Container Toolkit      1.17.7                            1.17.7                            Yes        PASS
========================================================================================================================

✓ OVERALL: PASS - All required versions match

========================================================================================================================
SERVICE HEALTH CHECKS
========================================================================================================================
Service Check                                 Status                                                  Result
========================================================================================================================
ptp4l Service Running                         Running                                                 PASS
ptp4l Locked                                  Locked as timeReceiver (avg rms: 2.4ns (20 samples))    PASS
phc2sys Service Running                       Running                                                 PASS
phc2sys Syncing                               Syncing (avg rms: 7.8ns (20 samples))                   PASS
ptp4l configured for aerial00                 Configured                                              PASS
nvidia.service has run successfully           Ran successfully                                        PASS
========================================================================================================================

✓ SERVICE HEALTH: All checks passed

========================================================================================================================
✓ OVERALL: All version and service health checks passed
========================================================================================================================
```

### 2. Generate a Manifest File

Generate manifests from current system configurations:

```bash
# On a DU system (e.g., aerial-smc-15)
./server_version_check.py -o manifest_du.csv

# On an RU system (e.g., aerial-r750-15)
./server_version_check.py -o manifest_ru.csv

# Combine DU and RU manifests into a single file
(head -1 manifest_du.csv && tail -n +2 manifest_du.csv && tail -n +2 manifest_ru.csv) > manifest_combined.csv
```

The system type (DU/RU) is automatically detected based on kernel version and GPU presence.

### 3. Check Multiple Servers

Run checks across multiple cluster nodes with filtering:

```bash
# Set SSH password securely from file
export SSHPASS=$(cat ~/aerial_pw)

# Run checks on specific nodes
./server_version_check_all.py \
    ~/nfs/gitlab/cicd-scripts/cicd_test_nodes.py \
    manifest_cg1_r750_25.3.csv \
    ~/nfs/cuBB_0102/ \
    aerial \
    --filter "aerial-smc-15|aerial-smc-16|aerial-r750-15|aerial-r750-16"
```

**Example Output:**

```
Parsing nodes from: /home/user/nfs/gitlab/cicd-scripts/cicd_test_nodes.py
Found 58 nodes
Filter 'aerial-smc-15|aerial-smc-16|aerial-r750-15|aerial-r750-16' matched 4/58 nodes

Running version checks on 4 nodes...
(This may take several minutes)

[1/4] Checking aerial-r750-15.nvidia.com...
[2/4] Checking aerial-r750-16.nvidia.com...
[3/4] Checking aerial-smc-15.nvidia.com...
[4/4] Checking aerial-smc-16.nvidia.com...

============================================================================================================================================
CLUSTER VERSION CHECK SUMMARY
============================================================================================================================================
Node                                Type     Version Check        Service Health       Overall         Failures
============================================================================================================================================
aerial-r750-15                      RU       PASS (8/8)           PASS (6/6)           ✓ PASS
aerial-r750-16                      RU       FAIL (6/8)           PASS (6/6)           ✗ FAIL          Cmdline
aerial-smc-15                       DU       PASS (13/13)         PASS (6/6)           ✓ PASS
aerial-smc-16                       DU       PASS (13/13)         PASS (6/6)           ✓ PASS
============================================================================================================================================

Summary:
  Total nodes: 4
  Passed: 3
  Failed: 1
  Unreachable: 0

Failed nodes:
  - aerial-r750-16: Cmdline

Run 'python3 testBenches/phase4_test_scripts/server_version_check/server_version_check.py <manifest>' on failed nodes for details.
```

## Command Reference

### server_version_check.py

**Verify versions against a manifest:**
```bash
sudo ./server_version_check.py <manifest_file>
```

**Generate a manifest from current system:**
```bash
./server_version_check.py -o <output_file>
```

**List all detectable components:**
```bash
./server_version_check.py -l
```

**Options:**
- `-o, --output FILE` - Generate manifest CSV from current system
- `-l, --list` - List all available components that can be checked
- `-h, --help` - Show help message

### server_version_check_all.py

**Run checks across cluster nodes:**
```bash
export SSHPASS=$(cat ~/aerial_pw)
./server_version_check_all.py <nodes_file> <manifest> <nfs_path> <username> [options]
```

**Arguments:**
- `nodes_file` - Path to cicd_test_nodes.py file
- `manifest` - Manifest CSV filename (in same directory as script)
- `nfs_path` - Shared NFS path accessible on all nodes
- `username` - SSH username for connecting to nodes

**Options:**
- `--filter PATTERN` - Filter nodes by hostname pattern (regex)
- `-t, --test` - Test mode: show SSH commands without executing
- `-v, --verbose` - Verbose mode: show full output from each node
- `-h, --help` - Show help message

**Examples:**

```bash
# Set password from file (do this once before running checks)
export SSHPASS=$(cat ~/aerial_pw)

# Check all nodes
./server_version_check_all.py \
    ~/nfs/gitlab/cicd-scripts/cicd_test_nodes.py \
    manifest_cg1_r750_25.3.csv \
    ~/nfs/cuBB_0102/ \
    aerial

# Check specific nodes with filter
./server_version_check_all.py \
    ~/nfs/gitlab/cicd-scripts/cicd_test_nodes.py \
    manifest_cg1_r750_25.3.csv \
    ~/nfs/cuBB_0102/ \
    aerial \
    --filter "smc-1[5-9]|r750-1[5-9]"

# Test mode (show commands without executing)
./server_version_check_all.py \
    ~/nfs/gitlab/cicd-scripts/cicd_test_nodes.py \
    manifest_cg1_r750_25.3.csv \
    ~/nfs/cuBB_0102/ \
    aerial \
    --test
```

## Manifest File Format

Manifest CSV files define expected versions for system components:

```csv
system_type,component,version,optional
DU,BMC,1.03,n
DU,BIOS,2.0c,n
DU,Kernel,6.8.0-1025-nvidia-64k,n
DU,Container Toolkit,1.17.7,y
RU,BMC,7.20,n
RU,BIOS,1.17.2,n
RU,Kernel,5.15.0-1042-nvidia-lowlatency,n
```

**Columns:**
- `system_type` - DU (Distributed Unit) or RU (Radio Unit)
- `component` - Component name (e.g., BMC, BIOS, Kernel)
- `version` - Expected version string
- `optional` - `y` if optional, `n` if required

