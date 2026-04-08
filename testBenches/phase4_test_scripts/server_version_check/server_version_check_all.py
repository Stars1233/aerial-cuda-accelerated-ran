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

"""Run server version checks across all nodes in a cluster."""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional


def parse_test_nodes(nodes_file: str) -> List[Dict[str, str]]:
    """Parse test nodes file and extract CG1+R750 paired aerial hostnames.
    
    Args:
        nodes_file: Path to cicd_test_nodes.py
        
    Returns:
        List of dicts with hostname and type (DU/RU)
    """
    # Load the nodes file dynamically
    spec = importlib.util.spec_from_file_location("cicd_test_nodes", nodes_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load nodes file: {nodes_file}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    nodes = []
    test_nodes = getattr(module, 'TEST_NODES', {})
    
    for rack_name, rack_data in test_nodes.items():
        if not isinstance(rack_data, dict):
            continue
        
        # Check if this rack has CG1 nodes
        cuphy_host = rack_data.get('cuphy-host')
        rue_host = rack_data.get('rue-host')
        
        if not isinstance(cuphy_host, dict) or not isinstance(rue_host, dict):
            continue
        
        # Check if cuphy-host is a CG1 node (has host_type: "_CG1")
        host_type = cuphy_host.get('host_type', '')
        if host_type != '_CG1':
            continue
        
        # Extract both hostnames
        smc_hostname = cuphy_host.get('hostname', '')
        r750_hostname = rue_host.get('hostname', '')
        
        # Only add if BOTH are present and match expected patterns
        has_smc = smc_hostname and 'aerial-smc-' in smc_hostname
        has_r750 = r750_hostname and 'aerial-r750-' in r750_hostname
        
        # Skip this rack if it's not a complete CG1+R750 pair
        if not (has_smc and has_r750):
            continue
        
        # Add both nodes from the complete pair
        nodes.append({'hostname': smc_hostname, 'type': 'DU'})
        nodes.append({'hostname': r750_hostname, 'type': 'RU'})
    
    return sorted(nodes, key=lambda x: x['hostname'])


def run_check_on_node(
    hostname: str,
    username: str,
    nfs_path: str,
    manifest: str,
    test_mode: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run version check on a remote node via SSH.
    
    Args:
        hostname: Target hostname
        username: SSH username
        nfs_path: NFS mount path on remote node
        manifest: Manifest filename (assumed to be in same directory as this script)
        test_mode: If True, print command without executing
        verbose: If True, print full output from node
        
    Returns:
        Dictionary with check results
    """
    # Get password from environment variable
    password = os.environ.get('SSHPASS', '')
    if not password:
        raise ValueError("SSHPASS environment variable not set")
    
    # Determine script path relative to nfs_path (same directory as this script)
    current_script = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_script)
    script_rel_path = os.path.relpath(script_dir, os.path.abspath(nfs_path))
    script_path = f"{nfs_path}/{script_rel_path}/server_version_check.py"
    manifest_path = f"{nfs_path}/{script_rel_path}/{manifest}"
    
    # Build SSH command with sshpass -e to read password from SSHPASS env var
    # Use sudo -S to read password from stdin
    # Run entire command (including cd) under sudo
    ssh_cmd = (
        f"sshpass -e ssh "
        f"-o ConnectTimeout=10 "
        f"-o StrictHostKeyChecking=no "
        f"{username}@{hostname} "
        f"'echo \"{password}\" | sudo -S sh -c \"cd {nfs_path} && python3 {script_path} {manifest_path}\" 2>&1'"
    )
    
    # Test mode: just print the command and return mock result
    if test_mode:
        print(f"\nSSH Command for {hostname}:")
        print(f"  {ssh_cmd}")
        return {
            'hostname': hostname,
            'type': 'TEST',
            'reachable': True,
            'version_check': 'TEST',
            'version_passed': 0,
            'version_total': 0,
            'service_health': 'TEST',
            'service_passed': 0,
            'service_total': 0,
            'overall': 'TEST',
            'error': None
        }
    
    result = {
        'hostname': hostname,
        'type': 'UNKNOWN',
        'reachable': True,
        'version_check': 'UNKNOWN',
        'version_passed': 0,
        'version_total': 0,
        'service_health': 'UNKNOWN',
        'service_passed': 0,
        'service_total': 0,
        'overall': 'UNKNOWN',
        'error': None,
        'version_failures': [],
        'service_failures': []
    }
    
    try:
        proc = subprocess.run(
            ssh_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=60
        )
        
        output = proc.stdout + proc.stderr
        
        # Verbose mode: Print complete output for troubleshooting
        if verbose:
            print(f"\n{'='*80}")
            print(f"VERBOSE OUTPUT for {hostname}")
            print(f"{'='*80}")
            print(f"Return code: {proc.returncode}")
            print(f"Full output:\n{output}")
            print(f"{'='*80}\n")
        
        # Check for SSH and execution failures
        if proc.returncode != 0:
            if 'Permission denied' in output and 'password for' not in output:
                # SSH auth failed (not sudo)
                result['reachable'] = False
                result['error'] = 'SSH authentication failed'
                result['overall'] = 'ERROR'
                return result
            elif 'Connection refused' in output or 'Connection timed out' in output or 'No route to host' in output:
                result['reachable'] = False
                result['error'] = 'SSH connection failed (unreachable)'
                result['overall'] = 'ERROR'
                return result
            elif "can't cd to" in output or "cd:" in output:
                result['reachable'] = False
                result['error'] = f'NFS path not accessible on node'
                result['overall'] = 'ERROR'
                return result
            elif len(output) < 100 and 'Detected System Type' not in output:
                # Script didn't run properly
                result['reachable'] = False
                result['error'] = f'Script execution failed (see -v output)'
                result['overall'] = 'ERROR'
                return result
        
        # Parse output
        # Extract system type
        type_match = re.search(r'Detected System Type:\s+(\w+)', output)
        if type_match:
            result['type'] = type_match.group(1)
        
        # Count version check results
        version_section = re.search(
            r'Component.*?Status.*?=+\s+(.*?)\s+=+',
            output,
            re.DOTALL
        )
        if version_section:
            lines = version_section.group(1).strip().split('\n')
            result['version_total'] = len([l for l in lines if l.strip()])
            result['version_passed'] = len([l for l in lines if 'PASS' in l])
        
        # Extract version check overall result
        if '✓ OVERALL: PASS - All required versions match' in output:
            result['version_check'] = 'PASS'
        elif '✗ OVERALL: FAIL' in output:
            result['version_check'] = 'FAIL'
        
        # Count service health results
        service_section = re.search(
            r'SERVICE HEALTH CHECKS.*?Service Check.*?=+\s+(.*?)\s+=+',
            output,
            re.DOTALL
        )
        if service_section:
            lines = service_section.group(1).strip().split('\n')
            result['service_total'] = len([l for l in lines if l.strip()])
            result['service_passed'] = len([l for l in lines if 'PASS' in l])
        
        # Extract service health overall result
        if '✓ SERVICE HEALTH: All checks passed' in output:
            result['service_health'] = 'PASS'
        elif '✗ SERVICE HEALTH: One or more checks failed' in output:
            result['service_health'] = 'FAIL'
        
        # Parse failure details
        version_fail_match = re.search(r'VERSION_FAILURES:\s+(.+)', output)
        if version_fail_match:
            result['version_failures'] = [f.strip() for f in version_fail_match.group(1).split(',')]
        
        service_fail_match = re.search(r'SERVICE_FAILURES:\s+(.+)', output)
        if service_fail_match:
            result['service_failures'] = [f.strip() for f in service_fail_match.group(1).split(',')]
        
        # Determine overall status
        if result['version_check'] == 'PASS' and result['service_health'] == 'PASS':
            result['overall'] = 'PASS'
        else:
            result['overall'] = 'FAIL'
            
            # Build error message from failures
            errors = []
            if result['version_failures']:
                errors.extend(result['version_failures'])
            if result['service_failures']:
                errors.extend(result['service_failures'])
            result['error'] = ', '.join(errors) if errors else 'Check failed'
        
    except subprocess.TimeoutExpired:
        result['reachable'] = False
        result['error'] = 'SSH timeout'
        result['overall'] = 'ERROR'
    except Exception as e:
        result['reachable'] = False
        result['error'] = str(e)
        result['overall'] = 'ERROR'
    
    return result


def display_summary(results: List[Dict[str, Any]]):
    """Display summary table of all results.
    
    Args:
        results: List of result dictionaries from run_check_on_node
    """
    print("\n" + "=" * 140)
    print("CLUSTER VERSION CHECK SUMMARY")
    print("=" * 140)
    print(f"{'Node':<35} {'Type':<8} {'Version Check':<20} {'Service Health':<20} {'Overall':<15} {'Failures':<40}")
    print("=" * 140)
    
    for result in results:
        hostname_short = result['hostname'].replace('.nvidia.com', '')
        
        if not result['reachable']:
            version_str = "UNREACHABLE"
            service_str = "UNREACHABLE"
            overall_str = "✗ ERROR"
            failures_str = result['error'] or "SSH failed"
        else:
            # Format version check
            if result['version_check'] == 'PASS':
                version_str = f"PASS ({result['version_passed']}/{result['version_total']})"
            elif result['version_check'] == 'FAIL':
                version_str = f"FAIL ({result['version_passed']}/{result['version_total']})"
            else:
                version_str = result['version_check']
            
            # Format service health
            if result['service_health'] == 'PASS':
                service_str = f"PASS ({result['service_passed']}/{result['service_total']})"
            elif result['service_health'] == 'FAIL':
                service_str = f"FAIL ({result['service_passed']}/{result['service_total']})"
            else:
                service_str = result['service_health']
            
            # Format overall
            overall_str = "✓ PASS" if result['overall'] == 'PASS' else "✗ FAIL"
            
            # Format failures
            all_failures = result['version_failures'] + result['service_failures']
            if all_failures:
                failures_str = ', '.join(all_failures[:3])  # Limit to first 3
                if len(all_failures) > 3:
                    failures_str += f" +{len(all_failures)-3} more"
            else:
                failures_str = ""
        
        print(f"{hostname_short:<35} {result['type']:<8} {version_str:<20} {service_str:<20} {overall_str:<15} {failures_str:<40}")
    
    print("=" * 140)
    
    # Summary statistics
    total = len(results)
    passed = len([r for r in results if r['overall'] == 'PASS'])
    failed = len([r for r in results if r['overall'] == 'FAIL'])
    unreachable = len([r for r in results if not r['reachable']])
    
    print(f"\nSummary:")
    print(f"  Total nodes: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Unreachable: {unreachable}")
    
    # List failed nodes with reasons
    failed_nodes = [r for r in results if r['overall'] == 'FAIL' and r['reachable']]
    if failed_nodes:
        print(f"\nFailed nodes:")
        for node in failed_nodes:
            hostname_short = node['hostname'].replace('.nvidia.com', '')
            print(f"  - {hostname_short}: {node['error']}")
    
    # List unreachable nodes
    unreachable_nodes = [r for r in results if not r['reachable']]
    if unreachable_nodes:
        print(f"\nUnreachable/SSH failed nodes:")
        for node in unreachable_nodes:
            hostname_short = node['hostname'].replace('.nvidia.com', '')
            print(f"  - {hostname_short}: {node['error']}")
    
    print(f"\nRun 'python3 testBenches/phase4_test_scripts/server_version_check/server_version_check.py <manifest>' on failed nodes for details.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run server version checks across all cluster nodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  export SSHPASS=$(cat ~/aerial_pw)
  python3 server_version_check_all.py \\
      /home/user/nfs/gitlab/cicd-scripts/cicd_test_nodes.py \\
      manifest_cg1_r750_25.3.csv \\
      /home/user/nfs/cuBB_0102 \\
      username

Note: Password must be set in SSHPASS environment variable for security.
      It is assumed the password is stored in ~/aerial_pw (chmod 600).
      This prevents the password from appearing in process lists or logs.
        """
    )
    
    parser.add_argument(
        'nodes_file',
        help='Path to cicd_test_nodes.py file'
    )
    parser.add_argument(
        'manifest',
        help='Manifest CSV filename (in same directory as this script)'
    )
    parser.add_argument(
        'nfs_path',
        help='Shared NFS path accessible on all nodes'
    )
    parser.add_argument(
        'username',
        help='SSH username for connecting to nodes'
    )
    parser.add_argument(
        '--filter',
        help='Optional hostname filter pattern (e.g., "smc-15", "r750-1[56]")',
        default=None
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Test mode: display SSH commands without executing them'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose mode: display full output from each node check'
    )
    
    args = parser.parse_args()
    
    # Check if SSHPASS environment variable is set
    if 'SSHPASS' not in os.environ:
        print("Error: SSHPASS environment variable not set")
        print("Set it with: export SSHPASS=$(cat ~/aerial_pw)")
        sys.exit(1)
    
    # Check if sshpass is available
    sshpass_check = subprocess.run(
        'which sshpass',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if sshpass_check.returncode != 0:
        print("Error: 'sshpass' is not installed. Install it with: sudo apt install sshpass")
        sys.exit(1)
    
    # Parse nodes
    try:
        print(f"Parsing nodes from: {args.nodes_file}")
        nodes = parse_test_nodes(args.nodes_file)
        print(f"Found {len(nodes)} nodes")
        
        # Apply filter if provided
        if args.filter:
            original_count = len(nodes)
            nodes = [n for n in nodes if re.search(args.filter, n['hostname'])]
            print(f"Filter '{args.filter}' matched {len(nodes)}/{original_count} nodes")
        
        if not nodes:
            print("No nodes to check")
            sys.exit(0)
        
    except Exception as e:
        print(f"Error parsing nodes file: {e}")
        sys.exit(1)
    
    # Run checks on each node
    if args.test:
        print(f"\nTEST MODE: Displaying SSH commands for {len(nodes)} nodes (not executing)\n")
    else:
        print(f"\nRunning version checks on {len(nodes)} nodes...")
        print("(This may take several minutes)\n")
    
    results = []
    for i, node in enumerate(nodes, 1):
        hostname = node['hostname']
        if not args.test:
            print(f"[{i}/{len(nodes)}] Checking {hostname}...", flush=True)
        
        result = run_check_on_node(
            hostname,
            args.username,
            args.nfs_path,
            args.manifest,
            test_mode=args.test,
            verbose=args.verbose
        )
        results.append(result)
    
    # Display summary
    if not args.test:
        display_summary(results)
        
        # Exit with error if any checks failed
        failed = len([r for r in results if r['overall'] != 'PASS'])
        sys.exit(0 if failed == 0 else 1)
    else:
        print(f"\nTEST MODE: {len(results)} SSH commands generated (not executed)")
        sys.exit(0)


if __name__ == '__main__':
    main()

