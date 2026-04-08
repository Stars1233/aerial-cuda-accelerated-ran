#!/usr/bin/env python3

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

"""Server version verification script.

This script verifies that the current machine matches the versions specified
in a manifest CSV file.
"""

import argparse
import csv
import re
import subprocess
import sys
from typing import Dict, Optional, Set


# Mapping of component names to their version check commands
# NOTE: Commands with escaped quotes (\" or \') need escaping only in Python.
#       To test on command line, replace \" with " and \' with '
VERSION_COMMANDS: Dict[str, str] = {
    # Copy-paste: cat /sys/devices/platform/ipmi_bmc.0/firmware_revision 2>/dev/null | awk -F'.' '{printf "%d.%02d\n", $1, $2}'
    "BMC": "cat /sys/devices/platform/ipmi_bmc.0/firmware_revision 2>/dev/null | awk -F'.' '{printf \"%d.%02d\\n\", $1, $2}'",
    
    # Copy-paste: cat /sys/class/dmi/id/bios_version 2>/dev/null
    "BIOS": "cat /sys/class/dmi/id/bios_version 2>/dev/null",
    
    # Copy-paste: nvidia-smi --query-gpu=vbios_version --format=csv,noheader
    "VBIOS": "nvidia-smi --query-gpu=vbios_version --format=csv,noheader",
    
    # Copy-paste: lsb_release -rs
    "Ubuntu": "lsb_release -rs",
    
    # Copy-paste: uname -r
    "Kernel": "uname -r",
    
    # Copy-paste: nvidia-smi --query-gpu=driver_version --format=csv,noheader
    "GPU Driver": "nvidia-smi --query-gpu=driver_version --format=csv,noheader",
    
    # Copy-paste: nvidia-smi | grep 'CUDA Version' | awk '{print $9}' | awk -F'.' '{print $1"."$2".0"}'
    "CUDA": "nvidia-smi | grep 'CUDA Version' | awk '{print $9}' | awk -F'.' '{print $1\".\"$2\".0\"}'",
    
    # Copy-paste: ofed_info -s 2>/dev/null | cut -d'-' -f3- | sed 's/:$//'
    "DOCA OFED": "ofed_info -s 2>/dev/null | cut -d'-' -f3- | sed 's/:$//'",
    
    # Copy-paste: ptp4l -v 2>&1 | head -1
    "PTP4L": "ptp4l -v 2>&1 | head -1",
    
    # Copy-paste: modinfo gdrdrv 2>/dev/null | grep ^version | awk '{print $2}'
    "GDRCOPY": "modinfo gdrdrv 2>/dev/null | grep ^version | awk '{print $2}'",
    
    # Copy-paste: cat /sys/class/net/aerial00/device/infiniband/*/fw_ver 2>/dev/null
    "BlueField FW": "cat /sys/class/net/aerial00/device/infiniband/*/fw_ver 2>/dev/null",
    
    # Copy-paste: ethtool aerial00 2>/dev/null | awk '/Speed:/{speed=$2; gsub(/[^0-9]/,"",speed); print speed/1000"G"}'
    "aerial00 Link Speed": "ethtool aerial00 2>/dev/null | awk '/Speed:/{speed=$2; gsub(/[^0-9]/,\"\",speed); print speed/1000\"G\"}'",
    
    # Copy-paste: nvidia-smi -i 0 --query-gpu=clocks.gr,clocks.max.gr --format=csv,noheader | awk -F', ' '{gsub(/ MHz/,"",$1); gsub(/ MHz/,"",$2); if ($1 == $2) print "Locked at " $1 "MHz (Max)"; else print "Not Locked at " $2 "MHz (Max)"}'
    "GPU Clock Lock": "nvidia-smi -i 0 --query-gpu=clocks.gr,clocks.max.gr --format=csv,noheader | awk -F', ' '{gsub(/ MHz/,\"\",$1); gsub(/ MHz/,\"\",$2); if ($1 == $2) print \"Locked at \" $1 \"MHz (Max)\"; else print \"Not Locked at \" $2 \"MHz (Max)\"}'",
    
    # Copy-paste: cat /proc/cmdline | tr -d '\n'
    "Cmdline": "cat /proc/cmdline | tr -d '\\n'",
    
    # Copy-paste: nvidia-ctk --version 2>/dev/null | head -1 | awk '{print $NF}'
    "Container Toolkit": "nvidia-ctk --version 2>/dev/null | head -1 | awk '{print $NF}'",
}


def get_version(component: str) -> Optional[str]:
    """Get the current version of a component.
    
    Args:
        component: The name of the component to check
        
    Returns:
        The version string, or None if detection failed
    """
    command = VERSION_COMMANDS.get(component)
    if not command:
        return None
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None


def parse_cmdline_params(cmdline: str) -> Set[str]:
    """Parse cmdline into individual parameters.
    
    Args:
        cmdline: The kernel command line string
        
    Returns:
        Set of parameter strings
    """
    # Strip out root= parameter since it's system-specific
    cmdline_normalized = re.sub(r'root=\S+\s*', '', cmdline)
    # Split on spaces and return as set
    return set(param for param in cmdline_normalized.split() if param)


def get_cmdline_diff(expected: str, actual: str) -> tuple[list[str], list[str]]:
    """Get differences between expected and actual cmdline parameters.
    
    Args:
        expected: Expected cmdline string
        actual: Actual cmdline string
        
    Returns:
        Tuple of (missing_from_actual, extra_in_actual)
    """
    expected_params = parse_cmdline_params(expected)
    actual_params = parse_cmdline_params(actual)
    
    missing = sorted(expected_params - actual_params)
    extra = sorted(actual_params - expected_params)
    
    return missing, extra


def check_service_running(service_name: str) -> tuple[bool, str]:
    """Check if a systemd service is running.
    
    Args:
        service_name: Name of the systemd service
        
    Returns:
        Tuple of (passed, status_message)
    """
    result = subprocess.run(
        f"systemctl is-active {service_name}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    is_running = result.stdout.strip() == "active"
    status = "Running" if is_running else "Not Running"
    return is_running, status


def check_ptp4l_locked() -> tuple[bool, str]:
    """Check if ptp4l is locked/operating correctly.
    
    For DU (timeReceiver): Check RMS values are low (< 10ns average)
    For RU (Grandmaster): Check for "assuming the grand master role"
    
    Returns:
        Tuple of (passed, status_message)
    """
    # First check if we can access journals
    result_check = subprocess.run(
        "journalctl -u ptp4l -n 1 --no-pager 2>&1",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Check if journalctl failed due to permissions
    if "systemd-journal" in result_check.stdout or "Hint:" in result_check.stdout:
        return False, "Requires sudo (run script with sudo)"
    
    # Check for Grandmaster role (RU case)
    result_gm = subprocess.run(
        "journalctl -u ptp4l -n 100 --no-pager 2>&1 | grep 'assuming the grand master role' | tail -1",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    if "assuming the grand master role" in result_gm.stdout:
        return True, "Grandmaster (acting as PTP source)"
    
    # Check for timeReceiver with RMS data (DU case)
    result = subprocess.run(
        "journalctl -u ptp4l -n 100 --no-pager 2>&1 | grep 'rms' | tail -20",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    lines = [line for line in result.stdout.strip().split('\n') if 'rms' in line and line.strip()]
    if not lines:
        return False, "No recent sync data (not Grandmaster or timeReceiver)"
    
    # Parse RMS values: "rms    3 max    7"
    rms_values = []
    for line in lines:
        match = re.search(r'rms\s+(\d+)', line)
        if match:
            rms_values.append(int(match.group(1)))
    
    if not rms_values:
        return False, "Could not parse RMS values"
    
    avg_rms = sum(rms_values) / len(rms_values)
    status = f"avg rms: {avg_rms:.1f}ns ({len(rms_values)} samples)"
    
    if avg_rms < 10:
        return True, f"Locked as timeReceiver ({status})"
    else:
        return False, f"Not locked ({status})"


def check_phc2sys_locked() -> tuple[bool, str]:
    """Check if phc2sys is syncing by analyzing RMS values from recent logs.
    
    Returns:
        Tuple of (passed, status_message)
    """
    # First check if we can access journals
    result_check = subprocess.run(
        "journalctl -u phc2sys -n 1 --no-pager 2>&1",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Check if journalctl failed due to permissions
    if "systemd-journal" in result_check.stdout or "Hint:" in result_check.stdout:
        return False, "Requires sudo (run script with sudo)"
    
    # Now get the actual rms data
    result = subprocess.run(
        "journalctl -u phc2sys -n 100 --no-pager 2>&1 | grep 'CLOCK_REALTIME rms' | tail -20",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    lines = [line for line in result.stdout.strip().split('\n') if 'rms' in line and line.strip()]
    if not lines:
        return False, "No recent sync data"
    
    # Parse RMS values: "CLOCK_REALTIME rms    6 max   20"
    rms_values = []
    for line in lines:
        match = re.search(r'rms\s+(\d+)', line)
        if match:
            rms_values.append(int(match.group(1)))
    
    if not rms_values:
        return False, "Could not parse RMS values"
    
    avg_rms = sum(rms_values) / len(rms_values)
    status = f"avg rms: {avg_rms:.1f}ns ({len(rms_values)} samples)"
    
    if avg_rms < 10:
        return True, f"Syncing ({status})"
    else:
        return False, f"Not syncing ({status})"


def check_ptp_config_aerial00() -> tuple[bool, str]:
    """Check if ptp4l is configured for aerial00 interface.
    
    Returns:
        Tuple of (passed, status_message)
    """
    try:
        with open('/etc/ptp.conf', 'r') as f:
            content = f.read()
            if '[aerial00]' in content:
                return True, "Configured"
            else:
                return False, "Not configured for aerial00"
    except FileNotFoundError:
        return False, "Config file not found"
    except PermissionError:
        return False, "Permission denied"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_nvidia_service_ran() -> tuple[bool, str]:
    """Check if nvidia.service has run successfully.
    
    Returns:
        Tuple of (passed, status_message)
    """
    result = subprocess.run(
        "systemctl show nvidia -p Result",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    if "Result=success" in result.stdout:
        return True, "Ran successfully"
    elif "Result=failure" in result.stdout:
        return False, "Failed on last run"
    else:
        return False, "Never run or unknown state"


def detect_system_type(manifest_file: str) -> str:
    """Detect system type based on kernel version from manifest.
    
    Args:
        manifest_file: Path to the CSV manifest file
        
    Returns:
        System type string (e.g., "DU" or "RU") based on kernel match
    """
    actual_kernel = get_version("Kernel")
    if not actual_kernel:
        print("Error: Unable to detect kernel version")
        sys.exit(1)
    
    # Read manifest and find which system_type matches the kernel
    try:
        with open(manifest_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['component'] == 'Kernel':
                    if row['version'] == actual_kernel:
                        return row['system_type']
        
        # If no match found, warn and default to DU
        print(f"Warning: Current kernel version '{actual_kernel}' does not match any system type in manifest", file=sys.stderr)
        print("Available kernel versions in manifest:", file=sys.stderr)
        
        with open(manifest_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['component'] == 'Kernel':
                    print(f"  - {row['system_type']}: {row['version']}", file=sys.stderr)
        
        print("Defaulting to DU system type for comparison...", file=sys.stderr)
        return "DU"
        
    except FileNotFoundError:
        print(f"Error: Manifest file '{manifest_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading manifest for system type detection: {e}")
        sys.exit(1)


def collect_system_versions() -> Dict[str, Optional[str]]:
    """Collect all component versions from the current system.
    
    Returns:
        Dictionary mapping component names to their versions (or None if not detected)
    """
    versions = {}
    for component in VERSION_COMMANDS.keys():
        versions[component] = get_version(component)
    return versions


def list_components():
    """List all available components that can be checked."""
    print("\nAvailable Components:")
    print("=" * 50)
    for component in sorted(VERSION_COMMANDS.keys()):
        if component == "Kernel":
            print(f"  - {component} (REQUIRED - used for system type detection)")
        else:
            print(f"  - {component}")
    print("=" * 50)
    print(f"\nTotal: {len(VERSION_COMMANDS)} components")
    print("\nNote: Kernel component is required and must match a version in the manifest")


def check_versions(manifest_file: str) -> bool:
    """Check versions against manifest file.
    
    Args:
        manifest_file: Path to the CSV manifest file
        
    Returns:
        True if all required versions match, False otherwise
    """
    results = []
    all_required_pass = True
    has_invalid_components = False
    cmdline_diff = None
    failed_components = []
    
    # Detect system type based on kernel version in manifest
    system_type = detect_system_type(manifest_file)
    
    # Collect all system versions once
    system_versions = collect_system_versions()
    
    # Read and process manifest
    try:
        with open(manifest_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip rows that don't match our system type
                if 'system_type' in row and row['system_type'] != system_type:
                    continue
                
                name = row['component']
                expected = row['version']
                optional = row['optional'].lower() == 'y'
                
                # Get actual version from collected data
                actual = system_versions.get(name)
                
                # For Cmdline, strip out the root= parameter before comparing
                # since it's system-specific (different volume group names)
                if name == "Cmdline" and actual:
                    # Remove root=... parameter from both
                    actual_compare = re.sub(r'root=\S+\s*', '', actual)
                    expected_compare = re.sub(r'root=\S+\s*', '', expected)
                else:
                    actual_compare = actual
                    expected_compare = expected
                
                # Check if versions match
                if actual is None:
                    status = "FAIL"
                    passed = False
                    has_invalid_components = True
                elif actual_compare == expected_compare:
                    status = "PASS"
                    passed = True
                else:
                    status = "FAIL"
                    passed = False
                    # Store cmdline diff for later display
                    if name == "Cmdline" and actual:
                        cmdline_diff = get_cmdline_diff(expected, actual)
                
                # Track overall status and failures
                if not optional and not passed:
                    all_required_pass = False
                    failed_components.append(name)
                
                # Truncate long values for display
                if name == "Cmdline":
                    expected_display = expected[:30] + "..." if len(expected) > 33 else expected
                    actual_display = actual[:30] + "..." if actual and len(actual) > 33 else (actual if actual else 'Invalid Component')
                else:
                    expected_display = expected
                    actual_display = actual if actual else 'Invalid Component'
                
                results.append({
                    'name': name,
                    'expected': expected_display,
                    'actual': actual_display,
                    'optional': optional,
                    'status': status,
                    'passed': passed
                })
    
    except FileNotFoundError:
        print(f"Error: Manifest file '{manifest_file}' not found")
        return False
    except KeyError as e:
        print(f"Invalid CSV format. Expected columns: system_type, component, version, optional")
        print(f"Missing column: {e}")
        return False
    except Exception as e:
        print(f"Error reading manifest: {e}")
        return False
    
    # Display results
    print(f"\nDetected System Type: {system_type}")
    print("\n" + "=" * 120)
    print(f"{'Component':<22} {'Expected':<33} {'Actual':<33} {'Optional':<10} {'Status':<10}")
    print("=" * 120)
    
    for result in results:
        optional_str = "Yes" if result['optional'] else "No"
        print(f"{result['name']:<22} {result['expected']:<33} {result['actual']:<33} {optional_str:<10} {result['status']:<10}")
    
    print("=" * 120)
    
    # Overall result
    if all_required_pass:
        print("\n✓ OVERALL: PASS - All required versions match")
    else:
        print("\n✗ OVERALL: FAIL - One or more required versions do not match")
    
    # Display note about invalid components if any were found
    if has_invalid_components:
        print("\nNOTE: 'Invalid Component' indicates that the script does not currently know how to")
        print("      retrieve version information for that component. The component may not be")
        print("      installed, or a detection command needs to be added to the script.")
    
    # Display cmdline parameter differences if any were found
    if cmdline_diff:
        missing, extra = cmdline_diff
        print("\nNOTE: Cmdline parameters that differ (root= parameter excluded from comparison):")
        if missing:
            print("      Missing from actual:")
            for param in missing:
                print(f"        - {param}")
        else:
            print("      Missing from actual: (none)")
        
        if extra:
            print("      Extra in actual:")
            for param in extra:
                print(f"        - {param}")
        else:
            print("      Extra in actual: (none)")
    
    # Output failures list for parsing by other scripts
    if failed_components:
        print(f"\nVERSION_FAILURES: {', '.join(failed_components)}")
    
    return all_required_pass


def check_service_health() -> bool:
    """Run service health checks and display results.
    
    Returns:
        True if all checks passed, False otherwise
    """
    print("\n" + "=" * 120)
    print("SERVICE HEALTH CHECKS")
    print("=" * 120)
    print(f"{'Service Check':<45} {'Status':<55} {'Result':<20}")
    print("=" * 120)
    
    all_checks_pass = True
    requires_sudo = False
    failed_checks = []
    
    checks = [
        ("ptp4l Service Running", check_service_running, ("ptp4l",)),
        ("ptp4l Locked", check_ptp4l_locked, ()),
        ("phc2sys Service Running", check_service_running, ("phc2sys",)),
        ("phc2sys Syncing", check_phc2sys_locked, ()),
        ("ptp4l configured for aerial00", check_ptp_config_aerial00, ()),
        ("nvidia.service has run successfully", check_nvidia_service_ran, ()),
    ]
    
    for check_name, check_func, args in checks:
        try:
            passed, status = check_func(*args)
            result = "PASS" if passed else "FAIL"
            if not passed:
                all_checks_pass = False
                failed_checks.append(check_name)
            if "Requires sudo" in status:
                requires_sudo = True
            print(f"{check_name:<45} {status:<55} {result:<20}")
        except Exception as e:
            print(f"{check_name:<45} {'Error: ' + str(e):<55} {'FAIL':<20}")
            all_checks_pass = False
            failed_checks.append(check_name)
    
    print("=" * 120)
    
    if all_checks_pass:
        print("\n✓ SERVICE HEALTH: All checks passed")
    else:
        print("\n✗ SERVICE HEALTH: One or more checks failed")
    
    if requires_sudo:
        print("\nNOTE: Some service health checks require elevated permissions.")
        print("      Run with 'sudo' for complete PTP lock verification:")
        print("      sudo python3 server_version_check.py manifest.csv")
    
    # Output failures list for parsing by other scripts
    if failed_checks:
        print(f"\nSERVICE_FAILURES: {', '.join(failed_checks)}")
    
    return all_checks_pass


def auto_detect_system_type(system_versions: Optional[Dict[str, Optional[str]]] = None) -> str:
    """Auto-detect system type based on hardware characteristics.
    
    Detection logic:
    - DU (Distributed Unit): Has GPU drivers (nvidia-smi works), kernel contains "nvidia-64k"
    - RU (Radio Unit): No GPU drivers, kernel contains "nvidia-lowlatency"
    
    Args:
        system_versions: Pre-collected system versions. If None, will collect them.
    
    Returns:
        System type string ("DU" or "RU")
    """
    # Collect versions if not provided
    if system_versions is None:
        system_versions = collect_system_versions()
    
    # Check kernel version first
    kernel = system_versions.get("Kernel")
    if kernel:
        if "nvidia-64k" in kernel or "64k" in kernel:
            return "DU"
        elif "nvidia-lowlatency" in kernel or "lowlatency" in kernel:
            return "RU"
    
    # Check for GPU presence as fallback
    gpu_driver = system_versions.get("GPU Driver")
    if gpu_driver:
        return "DU"
    
    # Default to DU if unclear
    print("Warning: Could not definitively determine system type, defaulting to DU", file=sys.stderr)
    return "DU"


def generate_manifest(output_file: str, system_type: Optional[str] = None) -> bool:
    """Generate a manifest CSV file from the current system.
    
    Args:
        output_file: Path to the output CSV file
        system_type: System type (DU or RU). If None, auto-detect.
        
    Returns:
        True if manifest was generated successfully, False otherwise
    """
    # Collect all component versions
    system_versions = collect_system_versions()
    
    # Auto-detect system type if not provided
    if system_type is None:
        system_type = auto_detect_system_type(system_versions)
    
    print(f"\nGenerating manifest for system type: {system_type}")
    print("=" * 120)
    
    components = []
    # Iterate in the order defined in VERSION_COMMANDS to maintain consistent ordering
    for component in VERSION_COMMANDS.keys():
        version = system_versions[component]
        if version:
            # Determine if component is optional
            # Container Toolkit is optional, all others are required
            optional = 'y' if component == 'Container Toolkit' else 'n'
            
            components.append({
                'system_type': system_type,
                'component': component,
                'version': version,
                'optional': optional
            })
            print(f"  ✓ {component:<22} {version}")
        else:
            print(f"  ✗ {component:<22} (not detected)")
    
    print("=" * 120)
    
    if not components:
        print("\nError: No components detected. Cannot generate manifest.")
        return False
    
    # Write to CSV file
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['system_type', 'component', 'version', 'optional'], lineterminator='\n')
            writer.writeheader()
            for component in components:
                writer.writerow(component)
        
        print(f"\n✓ Manifest generated successfully: {output_file}")
        print(f"  Total components: {len(components)}")
        return True
    except Exception as e:
        print(f"\nError writing manifest file: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify server versions against a manifest file or generate a new manifest'
    )
    parser.add_argument(
        'manifest',
        nargs='?',
        help='Path to the CSV manifest file (for verification mode)'
    )
    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List all available components that can be checked'
    )
    parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Generate a manifest CSV file from the current system (auto-detects system type)'
    )
    
    args = parser.parse_args()
    
    # Handle list option
    if args.list:
        list_components()
        sys.exit(0)
    
    # Handle manifest generation
    if args.output:
        success = generate_manifest(args.output)
        sys.exit(0 if success else 1)
    
    # Manifest file is required if not listing or generating
    if not args.manifest:
        parser.error('manifest file is required unless using -l/--list or -o/--output')
    
    version_check_passed = check_versions(args.manifest)
    service_health_passed = check_service_health()
    
    # Overall result
    if version_check_passed and service_health_passed:
        print("\n" + "=" * 120)
        print("✓ OVERALL: All version and service health checks passed")
        print("=" * 120)
        sys.exit(0)
    else:
        print("\n" + "=" * 120)
        print("✗ OVERALL: One or more checks failed")
        print("=" * 120)
        sys.exit(1)


if __name__ == '__main__':
    main()

