#!/usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import getpass
import json
import logging
import os
from packaging.version import parse, Version, InvalidVersion
import platform
import psutil
import paramiko
import re
import shlex
import subprocess
import time

SUPPORTED_GPU_DRV_VER = "570.104"
SUPPORTED_CUDA_VER = "12.8"
SUPPORTED_OFED_VER = "24.04-0.6.6"
SUPPORTED_NIC_FW_VER = "32.41.1000"
SUPPORTED_NVIDIA_CONTAINER_TOOLKIT_VER = "1.17.4"

KEY_LJUST_VALUE = 35
EXEC_PATH_LJUST_VALUE = 12
DELIMETER_CENTER_VALUE = 45
NOT_FOUND_STRING = 'N/A'

#--- Helper Class
# Regex to match a typical Bash prompt line like: "user@host:~$ " or "root@host:~# "
PROMPT_REGEX = re.compile(r'^[^@]+@[^:]+:.*[#$]\s*$')

class SSHClient:
    """
    SSHClient class to manage SSH connections and execute multiple commands on the host
    via a persistent interactive shell session.
    Prompts for password interactively (no need to pass it as an argument if no private key).
    """

    def __init__(self, host, port, username, private_key_path=None):
        """
        Initializes the SSHClient with connection details.

        Args:
            host (str): SSH destination host (IP or domain name)
            port (int): SSH port (e.g., 22)
            username (str): SSH username
            private_key_path (str, optional): Path to the private key file (for public key auth)
        """
        self.host = host
        self.port = port
        self.username = username
        self.private_key_path = private_key_path

        # Password is only needed if no private key is provided
        self.password = None

        self.sudo_cached = False
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.shell = None  # Holds our persistent shell session

        self.ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')  # Match pattern for ANSI escape sequence

    def connect(self):
        """
        Establishes an SSH connection using either:
          - Public key authentication (if private_key_path is set)
          - Password authentication (prompted via getpass)
        Then opens a persistent interactive shell and tries to cache sudo password immediately.
        """
        try:
            # If we have a private key, use key-based auth
            if self.private_key_path:
                print(f"[+] Connecting to {self.host} with public key auth: {self.private_key_path}")
                private_key = paramiko.RSAKey.from_private_key_file(self.private_key_path)
                self.client.connect(self.host, self.port, self.username, pkey=private_key)
            else:
                # Prompt for password if no key is provided
                print(f"[+] Connecting to {self.host} with password auth.")
                self.password = getpass.getpass(f"Password for {self.username}@{self.host}: ")
                self.client.connect(self.host, self.port, self.username, self.password)

            # Open an interactive shell session
            self.shell = self.client.invoke_shell()
            time.sleep(0.5)  # Give shell time to initialize
            self._drain_shell_buffer()

            # If we have a password, attempt to cache sudo right away
            if self.password:
                self._cache_sudo_password()

        except Exception as e:
            print(f"[!] Connection error: {e}")

    def _cache_sudo_password(self):
        """
        Runs 'sudo -v' to cache the password so subsequent sudo commands don't prompt again.
        """
        if self.sudo_cached or not self.shell or not self.password:
            return

        try:
            print("[+] Caching sudo password...")
            self.shell.send("sudo -v\n")
            time.sleep(0.2)
            self.shell.send(self.password + "\n")
            time.sleep(0.5)

            self._drain_shell_buffer()
            self.sudo_cached = True
            print("[+] Sudo password cached successfully.")
        except Exception as e:
            print(f"[!] Failed to cache sudo password: {e}")

    def _drain_shell_buffer(self):
        """Reads and discards any pending data in the shell's buffer."""
        while self.shell and self.shell.recv_ready():
            self.shell.recv(1024)

    def close(self):
        """Closes the persistent shell and the SSH connection."""
        if self.shell:
            self.shell.close()
        self.client.close()
        print("[+] SSH connection closed.")

    def execute(self, command_list, timeout=10):
        """
        Executes a command on the remote host via the persistent interactive shell.

        We use a sentinel marker (e.g. '__CMD_DONE__') to detect command completion,
        then remove lines containing both the echoed command, the marker, 
        and also remove any line that appears to be a shell prompt.

        Args:
            command_list (list): The command to execute, e.g. ['ls', '-l', '/root'].
            sudo (bool): Whether to prepend 'sudo' to the command.
            timeout (int): Max time (seconds) to wait for command completion.

        Returns:
            str: The captured stdout (minus the echoed command, marker, and prompt lines).
        """
        if not self.shell:
            print("[!] Shell session not active; reconnecting...")
            self.connect()
            if not self.shell:
                return ""

        base_command = shlex.join(command_list)

        self._drain_shell_buffer()

        try:
            self.shell.send(base_command + "\n")
        except Exception as e:
            print(f"[!] Error sending command: {e}")
            return ""

        output = ""
        start_time = time.time()

        while True:
            if (time.time() - start_time) > timeout:
                print(f"[!] Timeout waiting for command: {base_command}")
                break

            if self.shell.recv_ready():
                chunk = self.shell.recv(4096).decode(errors="replace")
                output += chunk

            else:
                time.sleep(0.01)

            if PROMPT_REGEX.match(output):
                break

        # Split the entire output by lines
        lines = output.splitlines(True)

        final_lines = []
        for line in lines:
            # Remove lines containing the command
            if base_command in line:
                continue

            # Remove lines that look like "user@host:~$" or "root@host:~#"
            if PROMPT_REGEX.match(line.strip()):
                continue

            final_lines.append(line)

        cleaned_output = "".join(final_lines)
        cleaned_output = self.ansi_escape.sub('', cleaned_output)  # remove ansi escape sequence
        cleaned_output = cleaned_output.strip('\n\r')
        # print(r"<<{}>>".format(cleaned_output))
        return cleaned_output


class KubernetesClient:
    def __init__(self, cli, aerial_pod_name=None, aerial_namespace=None):
        """
        Initialize the Kubernetes/OpenShift Manager.

        :param cli: "oc" (OpenShift) or "kubectl" (Kubernetes)
        :param aerial_pod_name: Aerial pod name (optional)
        :param aerial_: Aerial namespace (optional)
        """
        self.cli = cli
        self.aerial_pod_name = aerial_pod_name
        self.aerial_namespace = aerial_namespace
        self.target_pod = None
        self.target_container = None


    def run_command(self, command):
        """
        Execute an OpenShift (`oc`) or Kubernetes (`kubectl`) command and return the output.

        :param command: The command string to execute
        :return: The output of the command, or None if it fails
        """
        full_command = f"{self.cli} {command}"
        try:
            result = subprocess.run(full_command, shell=True, check=True, text=True, capture_output=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            print(f"Error output: {e.stderr}")
            return None
        

    def get_pod_from_daemonset(self, daemonset_name, namespace):
        """
        Retrieve the pod name corresponding to a given daemonset name.

        :param namespace: The namespace where the daemonset is deployed
        :param daemonset_name: The name of the daemonset
        :return: The pod name if found, otherwise None
        """
        # print(f"Fetching pod for daemonset '{daemonset_name}' in namespace '{namespace}' using '{self.cli}'...")

        cmd = f"get pods -n {namespace} -o json"
        output = self.run_command(cmd)

        if not output:
            print("Failed to retrieve pod list.")
            return None

        try:
            pods = json.loads(output)
            for pod in pods.get("items", []):
                pod_name = pod["metadata"]["name"]

                # Check if pod name starts with daemonset name
                if pod_name.startswith(daemonset_name):
                    # print(f"Found pod '{pod_name}' for daemonset '{daemonset_name}'.")
                    return pod_name
        except json.JSONDecodeError:
            print("Error parsing JSON response.")

        print(f"No pod found for daemonset '{daemonset_name}'.")
        return None
    

    def set_target_pod(self, daemonset_name, namespace="default", container=None):
        if "debug" in daemonset_name:
            self.target_pod = daemonset_name
        else:
            self.target_pod = self.get_pod_from_daemonset(daemonset_name, namespace)
            self.target_container = container
        self.target_namespace = namespace


    def execute(self, command_list):
        """
        Run a command inside a specific pod.

        :param command: The command to execute inside the pod
        """
        base_command = shlex.join(command_list).replace("'", '')
        if self.target_pod == "debug":
            k8s_command = f"oc debug node/$(oc get node -o json |jq -r '.items[0].metadata.name') -- {base_command}"
        elif self.target_pod == "debug_root":
            k8s_command = f"oc debug node/$(oc get node -o json |jq -r '.items[0].metadata.name') -- chroot /host {base_command}"
        else:
            ctr = ""
            if self.target_container:
                ctr = f"-c {self.target_container}"
            k8s_command = f"{self.cli} exec {self.target_pod} -n {self.target_namespace} {ctr} -- {base_command}"
        try:
            result = subprocess.run(k8s_command, shell=True, check=True, text=True, capture_output=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            print(f"Error output: {e.stderr}")
            return None


    def logs(self, options):
        """
        Run a command inside a specific pod.

        :param command: The command to execute inside the pod
        """
        ctr = ""
        if self.target_container:
            ctr = f"-c {self.target_container}"
        k8s_command = f"{self.cli} logs {self.target_pod} -n {self.target_namespace} {ctr} {options}"
        try:
            result = subprocess.run(k8s_command, shell=True, check=True, text=True, capture_output=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            print(f"Error output: {e.stderr}")
            return None
        

#--- Helper methods
def print_delimiter(name):
    print('-----' + name.ljust(DELIMETER_CENTER_VALUE, '-'))


def is_pep440_compliant(version_str):
    try:
        Version(version_str)
        return True
    except InvalidVersion:
        return False


def extract_ofed_version(version_str: str) -> str:
    """
    Extracts the numeric part of the string after a prefix
    matching 'OFED[a-zA-Z-]+' and a dash.
    
    Example:
      "OFED-internal-24.04-0.6.6" -> "24.04-0.6.6"
      
    If the string does not match the pattern, returns "".
    """
    # Pattern breakdown:
    #  1) ^(OFED[a-zA-Z-]+)-  : 
    #       - Must start with 'OFED'
    #       - Followed by one or more alphabets or hyphens
    #       - Followed by a dash
    #
    #  2) (\d+(?:\.\d+)*-\d+(?:\.\d+)*)  :
    #       - One or more digits
    #       - Optionally followed by dot + digits (zero or more times)
    #       - A dash
    #       - Same digit-dot pattern again
    #
    #  3) $ : End of string
    
    pattern = r'^(OFED[a-zA-Z-]+)-(\d+(?:\.\d+)*-\d+(?:\.\d+)*)$'
    match = re.match(pattern, version_str)
    if not match:
        # Return an empty string if it doesn't match
        return ""
    
    # group(2) corresponds to the numeric part: e.g. "24.04-0.6.6"
    return match.group(2)


# def parse_custom_version(version_str: str) -> tuple[int, ...]:
def parse_custom_version(version_str):
    """
    Splits a custom version string (e.g., '24.04-0.6.6') by hyphens
    and dots, and returns a tuple of integers for comparison.
    Example:
      '24.04-0.6.6' -> (24, 4, 0, 6, 6)
    """
    # Split by both hyphens and dots
    parts = re.split(r'[.-]', version_str)
    # Remove empty strings that might appear after splitting
    parts = [p for p in parts if p]
    # Convert each part to an integer
    return tuple(int(p) for p in parts)


def validate_version_parameter(value, requirement):
    is_compatible = False
    if is_pep440_compliant(value) and is_pep440_compliant(requirement):  # PEP 440 compliant version format
        if parse(value) >= parse(requirement):
            is_compatible = True

    elif extract_ofed_version(value) != "":  # OFED version format
        version = parse_custom_version(extract_ofed_version(value))
        if version >= parse_custom_version(requirement):
            is_compatible = True

    else:
        if type(requirement) is not list:
            requirement = [requirement]
        for i in range(len(requirement)):
            if requirement[i] in value:
                is_compatible = True
                break

    return is_compatible


def print_config(key, value, requirement=None, must=True):
    if requirement:
        is_compatible = validate_version_parameter(value, requirement)
        if not is_compatible:
            if len(requirement) <= 1:
                requirement = requirement[0]
            if must: 
                value = '\033[41m' + value + '\033[0m' + '\033[31m' + f'  ("{requirement}" is required)' + '\033[0m'
            else:
                value = '\033[43m' + value + '\033[0m' + '\033[33m' + f'  ("{requirement}" is recommended)' + '\033[0m'

    global KEY_LJUST_VALUE
    print(key.ljust(KEY_LJUST_VALUE) + ': ' + value)


def execute(command, sudo=False, client=None):
    if sudo: command = ['sudo', '-E'] + command
    if client:
        result = client.execute(command)
    else:
        if sudo: command = ['sudo', '-E'] + command
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True).stdout.strip()
        except FileNotFoundError:
            result = ""
    # print('"', result, '"')
    return result


def find_pattern_in_string(string, pattern, name, requirement=None, must=True, dump=True):
    result = re.search(pattern, string)
    if result: value = result.group(0)
    else: value = NOT_FOUND_STRING
    if dump: print_config(name, value, requirement=requirement, must=must)


def find_value_after_pattern_in_string(string, pattern, name, requirement=None, must=True, dump=True):
    key = re.search(pattern, string)
    if key:
        end = key.end()
        try:
            value = string[end:string.index('\n', end)].strip()
        except ValueError:
            value = string[end:].strip()
        if dump: print_config(name, value, requirement=requirement, must=must)
        return value
    else:
        if dump: print_config(name, NOT_FOUND_STRING, requirement=requirement, must=must)
        return None


def print_file_contents(filepath, name, client=None):
    result = execute(["cat", filepath], client=client)
    print_config(name, result)


def get_executable_path_dirname(executable, client=None):
    return os.path.dirname(execute(['which', executable], client=client))


def dump_package_if_installed(command, regex, version_after_regex, name, client=None, requirement=None, must=False):
    try:
        result = execute(command, client=client)
        dirname = get_executable_path_dirname(name, client=client)
        if version_after_regex:
            find_value_after_pattern_in_string(result, regex, name.ljust(EXEC_PATH_LJUST_VALUE) + dirname, requirement=requirement, must=must)
        else:
            find_pattern_in_string(result, regex, name.ljust(EXEC_PATH_LJUST_VALUE) + dirname, requirement=requirement, must=must)
    except FileNotFoundError:
        print_config(name, NOT_FOUND_STRING)


#--- System checks on the host
def dump_general_info(client=None):
    if type(client) == KubernetesClient:
        client.set_target_pod("debug")
    print_delimiter('General')
    result = execute(['cat', '/etc/hostname'], client=client)
    print_config('Hostname', result)
    result = execute(['hostname', '-I'], client=client)
    print_config('IP address', result)
    result = execute(['cat', '/etc/os-release'], client=client)
    find_value_after_pattern_in_string(result, 'PRETTY_NAME=', 'Linux distro')
    result = execute(['uname', '-r'], client=client)
    print_config('Linux kernel version', result)


def dump_system_info(client=None):
    print_delimiter('System')
    if type(client) == KubernetesClient: return  # No default pod that has ipmitool in Kubernetes/ OCP cluster
    arch = platform.machine()

    if (arch == 'aarch64'):
        print(execute(['ipmitool', 'fru', 'print'], sudo=True, client=client))
    else:
        result=execute(['dmidecode', '-t 1'], client=client)
        find_value_after_pattern_in_string(result, 'Manufacturer:\s+', 'Manufacturer')
        find_value_after_pattern_in_string(result, 'Product Name:\s+', 'Product Name')
        result=execute(['dmidecode', '-t 2'], client=client)
        find_value_after_pattern_in_string(result, 'Manufacturer:\s+', 'Base Board Manufacturer')
        find_value_after_pattern_in_string(result, 'Product Name:\s+', 'Base Board Product Name')
        result=execute(['dmidecode', '-t 3'], client=client)
        find_value_after_pattern_in_string(result, 'Manufacturer:\s+', 'Chassis Manufacturer')
        find_value_after_pattern_in_string(result, 'Type:\s+', 'Chassis Type')
        find_value_after_pattern_in_string(result, 'Height:\s+', 'Chassis Height')
        result=execute(['dmidecode', '-t 4'], client=client)
        find_value_after_pattern_in_string(result, 'Version:\s+', 'Processor')
        find_value_after_pattern_in_string(result, 'Max Speed:\s+', 'Max Speed')
        find_value_after_pattern_in_string(result, 'Current Speed:\s+', 'Current Speed')


def dump_kernel_cmdline(client=None):
    print_delimiter('Kernel Command Line')

    if type(client) == KubernetesClient and client.cli == "oc":
        client.set_target_pod("node-exporter", "openshift-monitoring")

    result = execute(['cat', '/proc/cmdline'], client=client)
    find_pattern_in_string(result, 'audit=(0|1|off|on)', 'Audit subsystem')
    find_pattern_in_string(result, 'clocksource=\S+', 'Clock source')
    find_pattern_in_string(result, 'hugepages=\d+', 'HugePage count', requirement='hugepages')
    find_pattern_in_string(result, 'hugepagesz=\d+(G|M)', 'HugePage size', requirement='hugepagesz')
    find_pattern_in_string(result, 'idle=[a-z]+', 'CPU idle time management', requirement='idle=poll', must=False)
    find_pattern_in_string(result, 'intel_idle.max_cstate=\d', 'Max Intel C-state')
    find_pattern_in_string(result, 'intel_iommu=[a-z\_]+', 'Intel IOMMU')
    find_pattern_in_string(result, 'iommu=[a-z]+', 'IOMMU')
    find_pattern_in_string(result, 'isolcpus=[a-z0-9\-,_]+', 'Isolated CPUs', requirement='isolcpus')
    find_pattern_in_string(result, 'mce=[a-z\_]+', 'Corrected errors')
    find_pattern_in_string(result, 'nohz_full=[0-9\-,]+', 'Adaptive-tick CPUs', requirement='nohz_full')
    find_pattern_in_string(result, 'nosoftlockup', 'Soft-lockup detector disable', requirement='nosoftlockup')
    find_pattern_in_string(result, 'processor.max_cstate=\d', 'Max processor C-state', requirement='processor.max_cstate=0')
    find_pattern_in_string(result, 'rcu_nocb_poll', 'RCU callback polling', requirement='rcu_nocb_poll')
    find_pattern_in_string(result, 'rcu_nocbs=[0-9\-,]+', 'No-RCU-callback CPUs', requirement='rcu_nocbs')
    find_pattern_in_string(result, 'tsc=[a-z]+', 'TSC stability checks', requirement='tsc=reliable')
    find_pattern_in_string(result, 'irqaffinity=[a-z0-9]+', 'IRQ affinity', requirement='irqaffinity')
    find_pattern_in_string(result, 'acpi_power_meter\.force_cap_on=[a-z]+', 'ACPI power meter cap forcely on', requirement='acpi_power_meter.force_cap_on=y')
    find_pattern_in_string(result, 'numa_balancing=[a-z]+', 'NUMA balancing', requirement='numa_balancing=disable')
    find_pattern_in_string(result, 'init_on_alloc=[a-z0-9]+', 'Mem init on alloc', requirement='init_on_alloc=0')
    find_pattern_in_string(result, 'preempt=[a-z0-9]+', 'Preempt')
    find_pattern_in_string(result, 'psi=[a-z0-9]+', 'Pressure Stall Information', requirement='psi=0', must=False)


def dump_cpu_info(client=None):
    print_delimiter('CPU')

    if type(client) == KubernetesClient and client.cli == "oc":
        client.set_target_pod("node-exporter", "openshift-monitoring")

    result = execute(['lscpu'], client=client)
    find_value_after_pattern_in_string(result, 'CPU\(s\):\s+', 'CPU cores')
    find_value_after_pattern_in_string(result, 'Thread\(s\) per core:\s+', 'Thread(s) per CPU core')
    if 'x86_64' in result:
        find_value_after_pattern_in_string(result, 'CPU MHz:\s+', 'CPU MHz:')
    elif 'aarch64' in result:
        find_value_after_pattern_in_string(result, 'CPU max MHz:\s+', 'CPU max MHz:')
    find_value_after_pattern_in_string(result, 'Socket\(s\):\s+', 'CPU sockets')


def dump_memory_info(client=None):
    print_delimiter('Memory')

    if type(client) == KubernetesClient and client.cli == "oc":
        client.set_target_pod("node-exporter", "openshift-monitoring")

    result = execute(['cat', '/proc/meminfo'], client=client)
    find_value_after_pattern_in_string(result, 'HugePages_Total:\s+', 'HugePage count')
    find_value_after_pattern_in_string(result, 'HugePages_Free:\s+', 'Free HugePages')
    find_value_after_pattern_in_string(result, 'Hugepagesize:\s+', 'HugePage size')
    result = execute(['df', '-h', '|', 'grep', 'tmpfs', '|', 'grep', '/dev/shm'], client=client)
    find_pattern_in_string(result, '\d+[\.\d+](G|M)?', 'Shared memory size')


def dump_nic_info(client=None):
    print_delimiter('Mellanox NICs')

    if type(client) == KubernetesClient: return

    try:
        result = execute(['mlxfwmanager'], sudo=True, client=client)
        nic_idx = 1
        nics = re.finditer('Device \#\d+', result)
        for nic in nics:
            print('NIC' + str(nic_idx))
            nic_info = result[nic.end():]
            nic_bdf = find_value_after_pattern_in_string(nic_info, 'PCI Device Name:\s+', 'NIC', dump=False)

            find_value_after_pattern_in_string(nic_info, 'Device Type:\s+', '  NIC product name')
            find_value_after_pattern_in_string(nic_info, 'Part Number:\s+', '  NIC part number')
            find_value_after_pattern_in_string(nic_info, 'PCI Device Name:\s+', '  NIC PCIe bus id')
            find_pattern_in_string(nic_info, '\d+\.\d+\.\d+', '  NIC FW version', requirement=SUPPORTED_NIC_FW_VER, must=True)

            mlxconfig_result = execute(['mlxconfig', '-d', nic_bdf, 'q'], sudo=True, client=client)
            find_value_after_pattern_in_string(mlxconfig_result, 'INTERNAL_CPU_MODEL\s+', '  INTERNAL_CPU_MODEL', requirement="EMBEDDED_CPU(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'INTERNAL_CPU_PAGE_SUPPLIER\s+', '  INTERNAL_CPU_PAGE_SUPPLIER', requirement="EXT_HOST_PF(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'INTERNAL_CPU_ESWITCH_MANAGER\s+', '  INTERNAL_CPU_ESWITCH_MANAGER', requirement="EXT_HOST_PF(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'INTERNAL_CPU_IB_VPORT0\s+', '  INTERNAL_CPU_IB_VPORT0', requirement="EXT_HOST_PF(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'INTERNAL_CPU_OFFLOAD_ENGINE\s+', '  INTERNAL_CPU_OFFLOAD_ENGINE', requirement="DISABLED(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'FLEX_PARSER_PROFILE_ENABLE\s+', '  FLEX_PARSER_PROFILE_ENABLE', requirement="4", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'PROG_PARSE_GRAPH\s+', '  PROG_PARSE_GRAPH', requirement="True(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'ACCURATE_TX_SCHEDULER\s+', '  ACCURATE_TX_SCHEDULER', requirement="True(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'CQE_COMPRESSION\s+', '  CQE_COMPRESSION', requirement="AGGRESSIVE(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'REAL_TIME_CLOCK_ENABLE\s+', '  REAL_TIME_CLOCK_ENABLE', requirement="True(1)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'LINK_TYPE_P1\s+', '  LINK_TYPE_P1', requirement="ETH(2)", must=True)
            find_value_after_pattern_in_string(mlxconfig_result, 'LINK_TYPE_P2\s+', '  LINK_TYPE_P2', requirement="ETH(2)", must=True)

            nic_idx += 1
    except FileNotFoundError:
        return


def dump_net_interface_info(client=None):
    print_delimiter('Mellanox NIC Interfaces')

    if type(client) == KubernetesClient:
        client.set_target_pod("mofed", "nvidia-network-operator", "mofed-container")

    try:
        net_idx = 0
        ibdev2netdev = execute(['ibdev2netdev', '-v'], client=client)
        ports = re.findall('[0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+', ibdev2netdev)
        ibdevs = re.findall('mlx5_\d+', ibdev2netdev)
        interfaces = re.findall('==> [a-zA-Z0-9]+', ibdev2netdev)

        for port in ports:
            print('Interface' + str(net_idx))
            ifc = interfaces[net_idx].split()[1]
            port_bdf = ports[net_idx]
            ibdev = ibdevs[net_idx]
            print_config('  Name', ifc)
            print_config('  Network adapter', ibdev)
            print_config('  PCIe bus id', port_bdf)
            # result = execute(['ifconfig', ifc], client=client)
            print_file_contents('/sys/class/net/' + ifc + '/address', '  Ethernet address', client)
            print_file_contents('/sys/class/net/' + ifc + '/operstate', '  Operstate', client)
            print_file_contents('/sys/class/net/' + ifc + '/mtu', '  MTU', client)
            result = execute(['ethtool', '-a', ifc], client=client)
            find_value_after_pattern_in_string(result, 'RX:\s+', '  RX flow control', requirement='off', must=True)
            find_value_after_pattern_in_string(result, 'TX:\s+', '  TX flow control', requirement='off', must=True)
            result = execute(['ethtool', '-T', ifc], client=client)
            find_value_after_pattern_in_string(result, 'PTP Hardware Clock:\s+', '  PTP hardware clock')
            result = execute(['mlnx_qos', '-i', ifc], client=client)
            find_value_after_pattern_in_string(result, 'Priority trust state:\s+', '  QoS Priority trust state')
            result = execute(['lspci', '-s', port_bdf, '-vvv'], client=client)
            find_value_after_pattern_in_string(result, 'MaxReadReq\s+', '  PCIe MRRS')
            result = execute(['ethtool', '--show-priv-flags', ifc], client=client)
            find_value_after_pattern_in_string(result, 'tx_port_ts\s+: ', 'High-quality Tx timestamp', requirement='on', must=True)
            net_idx += 1
    except FileNotFoundError:
        return


def dump_linux_ptp_deamons_output(client=None):
    print_delimiter('Linux PTP')

    if type(client) == KubernetesClient and client.cli == "oc":
        client.set_target_pod("linuxptp-daemon", "openshift-ptp", "linuxptp-daemon-container")
        print(client.logs("| grep 'ptp4l\[' | tail"))
        client.set_target_pod("linuxptp-daemon", "openshift-ptp", "linuxptp-daemon-container")
        print(client.logs("| grep 'phc2sys\[' | tail"))
    else:
        print(execute(['systemctl', 'status', 'ptp4l.service', '--no-pager', '--full'], sudo=True, client=client))
        print(execute(['systemctl', 'status', 'phc2sys.service', '--no-pager', '--full'], sudo=True, client=client))
        print_delimiter('NTP')
        find_value_after_pattern_in_string(execute(['timedatectl'], sudo=True, client=client), 'NTP service: ', 'NTP', requirement='inactive', must=True)


def dump_required_host_packages_info(client=None):
    print_delimiter('Software Packages')
    dump_package_if_installed(['docker', '--version'], '\d+(\.\d+)+', False, 'docker', client=client)
    find_value_after_pattern_in_string(execute(['nvidia-ctk', '--version'], client=client), 'NVIDIA Container Toolkit CLI version ', 'NVIDIA Container Toolkit', requirement=SUPPORTED_NVIDIA_CONTAINER_TOOLKIT_VER, must=True)
    dump_package_if_installed(['ofed_info', '-s'], '[a-zA-Z\d\.-]+', False, 'OFED version', client=client, requirement=SUPPORTED_OFED_VER, must=True)
    dump_package_if_installed(['dpkg', '--no-pager', '-l', 'linuxptp'], '\d+(\.\d+)*-\d+(\.\d+)*', False, 'ptp4l', client=client)


def dump_required_pod_info(client):
    print_delimiter('Software Packages')

    client.set_target_pod("mofed", "nvidia-network-operator")
    dump_package_if_installed(['ofed_info', '-s'], '[a-zA-Z\d\.-]+', False, 'ofed_info', client=client, requirement=SUPPORTED_OFED_VER, must=True)

    client.set_target_pod("linuxptp-daemon", "openshift-ptp", "linuxptp-daemon-container")
    dump_package_if_installed(['ptp4l', '-v'], '.*', False, 'ptp4l', client=client)


def dump_kernel_modules(client=None):
    print_delimiter('Loaded Kernel Modules')

    if type(client) == KubernetesClient and client.cli == "oc":
        client.set_target_pod("debug_root")

    result = execute(['lsmod'], client=client)
    find_pattern_in_string(result, 'gdrdrv', 'GDRCopy')
    find_pattern_in_string(result, '(nv_peer_mem|nvidia_peermem)', 'GPUDirect RDMA')
    find_pattern_in_string(result, 'nvidia', 'Nvidia')


def dump_non_persistent_settings(client=None):
    print_delimiter('Non-persistent settings')
    
    if type(client) == KubernetesClient and client.cli == "oc":
        client.set_target_pod("debug")

    result = execute(['sysctl', '-a'], client=client)
    find_pattern_in_string(result, 'vm\.swappiness = \d+', 'VM swappiness')
    find_pattern_in_string(result, 'vm\.zone_reclaim_mode = \d+', 'VM zone reclaim mode')


def dump_docker_info(client=None):
    try:
        print_delimiter('Docker images')
        print(execute(['docker', 'image', 'ls'], client=client))
        print_delimiter('Docker containers')
        print(execute(['docker', 'ps'], client=client))
    except FileNotFoundError:
        return


def dump_kernel_params(client=None):
    print_delimiter('Kernel Parameters')

    if type(client) == KubernetesClient and client.cli == "oc":
        client.set_target_pod("node-exporter", "openshift-monitoring")
        
    result = execute(['cat', '/proc/sys/kernel/sched_rt_runtime_us'], client=client)
    print_config('Real-time throttling', result, requirement='-1')
    result = execute(['cat', '/sys/kernel/mm/transparent_hugepage/enabled'], client=client)
    find_pattern_in_string(result, '\[[a-z]+\]', 'Transparent hugepage', requirement=['[madvise]', '[never]'])


MISFIT_ALLOWLIST = [
    'cpuhp\/*',
    'cuphycontroller',
    'idle_inject\/*',
    'irq\/*',
    'kworker\/*',
    'migration\/*',
    'ru-emulator',
    'test_mac'
]

ISOLCPUS = range(2,22)

def intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def dump_check_affinity(client=None):
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    logging.warning('Start scanning for processes running on Aerial-reserved CPU cores.  Hit Control-C to quit.')
    while True:
        misfits = []
        for proc in psutil.process_iter():
            cores = set()
            cores.add(proc.cpu_num())
            for t in proc.threads():
                pt = psutil.Process(pid=t.id)

                cores.add(pt.cpu_num())

            cores_isect = intersection(cores,ISOLCPUS)
            if len(cores_isect) > 0:
                is_allowed_misfit = False
                for allow in MISFIT_ALLOWLIST:
                    m = re.search(allow,proc.name())
                    if m is not None:
                        is_allowed_misfit = True
                        break # out of inner for loop

                if is_allowed_misfit == False:
                    misfits.append((proc.name(), cores))

        for misfit in misfits:
            logging.warning(f"Found name: {misfit[0]} cores: {misfit[1]}")


#--- System checks in the container
def dump_gpu_info(client=None):
    print_delimiter('Nvidia GPUs')

    if type(client) == KubernetesClient:
        client.set_target_pod("nvidia-driver-daemonset", "nvidia-gpu-operator")

    try:
        result = execute(['nvidia-smi', '-q'], client=client)
    except FileNotFoundError:
        return

    find_value_after_pattern_in_string(result, 'Driver Version\s+:\s', 'GPU driver version', requirement=SUPPORTED_GPU_DRV_VER, must=True)
    find_value_after_pattern_in_string(result, 'CUDA Version\s+:\s', 'CUDA version', requirement=SUPPORTED_CUDA_VER, must=True)

    gpu_idx = 0
    gpus = re.finditer('GPU [0-9A-Fa-f]+:[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+', result)
    for gpu in gpus:
        print('GPU' + str(gpu_idx))
        gpu_info = result[gpu.end():]
        find_value_after_pattern_in_string(gpu_info, 'Product Name\s+:\s', '  GPU product name')
        find_value_after_pattern_in_string(gpu_info, 'Persistence Mode\s+:\s', '  GPU persistence mode', requirement="Enabled", must=True)
        find_value_after_pattern_in_string(gpu_info, 'GPU Current Temp\s+:\s', '  Current GPU temperature')
        max_gpu_clock = find_value_after_pattern_in_string(gpu_info, 'Max Clocks\s+Graphics\s+:\s', '  Max GPU clock frequency')
        find_value_after_pattern_in_string(gpu_info, 'Clocks\s+Graphics\s+:\s', '  GPU clock frequency', requirement=max_gpu_clock, must=True)
        find_value_after_pattern_in_string(gpu_info, 'Bus Id\s+:\s', '  GPU PCIe bus id')
        gpu_idx += 1

    print_delimiter('GPUDirect topology')
    print(execute(['nvidia-smi', 'topo', '-m'], client=client))

  
def dump_envvars(client=None):
    print_delimiter('Environment variables')
    result = execute(['env'], client=client)
    find_value_after_pattern_in_string(result, 'CUDA_DEVICE_MAX_CONNECTIONS=', 'CUDA_DEVICE_MAX_CONNECTIONS')
    find_value_after_pattern_in_string(result, 'cuBB_SDK=', 'cuBB_SDK')


def dump_required_container_packages_info(client=None):
    print_delimiter('Software Packages in the Container')

    if type(client) == KubernetesClient and client.aerial_pod_name:
        client.set_target_pod(client.aerial_pod_name, client.aerial_namespace)
    else: return

    dump_package_if_installed(['cmake', '--version'], 'cmake version\s+', True, 'cmake')
    dump_package_if_installed(['gcc', '--version'], '\d+(\.\d+)+', False, 'gcc')
    dump_package_if_installed(['meson', '--version'], '^', True, 'meson')
    dump_package_if_installed(['ninja', '--version'], '^', True, 'ninja')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cuBB_system_info', description='Dump system information for running cuBB')
    parser.add_argument('-a', '--check_affinity', help='Check for processes with incorrect CPU affinity', action='store_true')
    parser.add_argument('-b', '--boot', help='Kernel boot parameters', action='store_true')
    parser.add_argument('-c', '--cpu', help='CPU', action='store_true')
    parser.add_argument('-d', '--docker', help='Docker images and containers', action='store_true')
    parser.add_argument('-e', '--envvar', help='Environment variables', action='store_true')
    parser.add_argument('-g', '--gpu', help='GPU', action='store_true')
    parser.add_argument('-i', '--interface', help='Net interface', action='store_true')
    parser.add_argument('-l', '--lkm', help='Loaded Kernel Modules (LKMs)', action='store_true')
    parser.add_argument('-m', '--memory', help='Memory', action='store_true')
    parser.add_argument('-n', '--nic', help='NICs', action='store_true')
    parser.add_argument('-p', '--packages', help='Software packages', action='store_true')
    parser.add_argument('--ptp', help='linuxptp deamons', action='store_true')
    parser.add_argument('-s', '--sysctl', help='Non-persistent system settings', action='store_true')
    parser.add_argument('--sys', help='System Info', action='store_true')
    parser.add_argument('-k', '--kernel', help='Kernel param', action='store_true')
    parser.add_argument("--host", help="SSH host (IP or domain)")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--username", help="SSH username")
    parser.add_argument("--password", help="SSH password (for password authentication)")
    parser.add_argument("--private-key", dest="private_key_path", help="Path to private key file (for public key authentication)")
    parser.add_argument("--cli", choices=["oc", "kubectl"], default=None, help="Specify whether to use 'oc' or 'kubectl'. Default is 'oc'.")
    parser.add_argument("--aerial-pod", help="Aerial Pod name", default=None)
    parser.add_argument("--aerial-namespace", help="Aerial namespace", default=None)

    args = parser.parse_args()

    check_affinity = False
    boot = True
    cpu = True
    docker = False
    envvar = True
    gpu = True
    interface = True
    lkm = True
    memory = True
    nic = True
    packages = True
    ptp = True
    sysctl = True
    sys = True
    kernel = True

    #if any([args.check_affinity, args.boot, args.cpu, args.docker, args.envvar, args.gpu, args.interface, args.lkm, args.memory, args.nic, args.packages, args.ptp, args.sysctl]):
    #    (check_affinity, boot, cpu, docker, envvar, gpu, interface, lkm, memory, nic, packages, ptp, sysctl) = (args.boot, args.cpu, args.docker, args.envvar, args.gpu, args.interface, args.lkm, args.memory, args.nic, args.packages, args.ptp, args.sysctl)
    if any([args.check_affinity, args.boot, args.cpu, args.docker, args.envvar, args.gpu, args.interface, args.lkm, args.memory, args.nic, args.packages, args.ptp, args.sysctl, args.sys, args.kernel]):
        (check_affinity, boot, cpu, docker, envvar, gpu, interface, lkm, memory, nic, packages, ptp, sysctl, sys, kernel) = (args.check_affinity, args.boot, args.cpu, args.docker, args.envvar, args.gpu, args.interface, args.lkm, args.memory, args.nic, args.packages, args.ptp, args.sysctl, args.sys, args.kernel)

    ssh_client = None
    if args.host and args.username:
        # Initialize SSH client with arguments (some may have been prompted)
        ssh_client = SSHClient(
            host=args.host,
            port=args.port,
            username=args.username,
            private_key_path=args.private_key_path
        )

        # Connect
        ssh_client.connect()
    
    k8s_client = None
    if args.cli:
        k8s_client = KubernetesClient(args.cli, args.aerial_pod, args.aerial_namespace)

    if ssh_client:
        client = ssh_client  # SSH is primary
    elif k8s_client:
        client = k8s_client  # K8s is secondary
    else:
        client = None

    dump_general_info(client)
    if sys:             dump_system_info(client)
    if check_affinity:  dump_check_affinity()
    if boot:            dump_kernel_cmdline(client)
    if cpu:             dump_cpu_info(client)
    if envvar:
        if k8s_client and args.aerial_pod: 
                        dump_envvars(k8s_client)
        else:           dump_envvars()
    if memory:          dump_memory_info(client)
    if gpu:             dump_gpu_info(k8s_client)
    if lkm:             dump_kernel_modules(client)
    if sysctl:          dump_non_persistent_settings(client)
    if docker and not k8s_client: 
                        dump_docker_info(client)
    if kernel:          dump_kernel_params(client)
    if packages and ssh_client: 
                        dump_required_host_packages_info(ssh_client)
    if packages and k8s_client: 
                        dump_required_pod_info(k8s_client)
    if packages:        dump_required_container_packages_info(k8s_client)
    if ptp:
        if k8s_client and args.cli == "oc": 
                        dump_linux_ptp_deamons_output(k8s_client)
        else:           dump_linux_ptp_deamons_output(client)
    if interface:
        if k8s_client:  dump_net_interface_info(k8s_client)
        else:           dump_net_interface_info(client)
    if nic:             dump_nic_info(client)