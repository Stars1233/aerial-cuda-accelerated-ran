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

import os
import argparse
import yaml
import sys
import re

#Maps input parameter from physical core yaml to output parameter
# in cuphycontroller/l2adapter/testmac/ru_emulator yaml
#
#Lists indicate nested structure of output parameter
config_mappings = {
    "cuphycontroller": {
        "all_low_priority_threads": "low_priority_core",
        "dpdk_thread": ["cuphydriver_config", "dpdk_thread"],
        "fh_stats_dump_thread": ["cuphydriver_config", "fh_stats_dump_cpu_core"],
        "h2d_copy_thread": ["cuphydriver_config", "h2d_copy_thread_cpu_affinity"],
        "ul_worker_threads": ["cuphydriver_config", "workers_ul"],
        "dl_worker_threads": ["cuphydriver_config", "workers_dl"],
        "dl_debug_thread": ["cuphydriver_config", "debug_worker"],
    },
    "l2adapter": {
        "l2a_timer_thread": ["timer_thread_config", "cpu_affinity"],
        "l2a_message_processing_thread": ["message_thread_config", "cpu_affinity"],
        "l2a_pcap_capture_thread": (
            ["transport", "app_config", "pcap_shm_caching_cpu_core"],
            ["transport", "app_config", "pcap_file_saving_cpu_core"],
        ),
    },
    "testmac": {
        "tbonly_tm_all_low_priority_threads": "low_priority_core",
        "tbonly_tm_schedule_thread": ["sched_thread_config", "cpu_affinity"],
        "tbonly_tm_receive_thread": ["recv_thread_config", "cpu_affinity"],
        "tbonly_tm_builder_thread": ["builder_thread_config", "cpu_affinity"],
        "tbonly_worker_threads": ["worker_cores"],
    },
    "testmac1": {
        "tbonly_tm1_all_low_priority_threads": "low_priority_core",
        "tbonly_tm1_schedule_thread": ["sched_thread_config", "cpu_affinity"],
        "tbonly_tm1_receive_thread": ["recv_thread_config", "cpu_affinity"],
        "tbonly_tm1_builder_thread": ["builder_thread_config", "cpu_affinity"],
        "tbonly_tm1_worker_threads": ["worker_cores"],
    },
    "multi_nvipc": {
        "l2a_pcap_capture_thread": (
            ["transport[0]", "app_config", "pcap_shm_caching_cpu_core"],
            ["transport[0]", "app_config", "pcap_file_saving_cpu_core"],
        ),
        "l2a_pcap_capture_thread_1": (
            ["transport[1]", "app_config", "pcap_shm_caching_cpu_core"],
            ["transport[1]", "app_config", "pcap_file_saving_cpu_core"],
        ),
    },
    "ru_emulator": {
        "tbonly_re_ul_threads": ["ru_emulator", "ul_core_list"],
        "tbonly_re_ul_srs_threads": ["ru_emulator", "ul_srs_core_list"],
        "tbonly_re_dl_threads": ["ru_emulator", "dl_core_list"],
        "tbonly_re_all_low_priority_threads": ["ru_emulator", "low_priority_core"],
        "tbonly_re_dpdk_thread": ["ru_emulator", "aerial_fh_dpdk_thread"],
        "tbonly_re_pdump_client_thread": ["ru_emulator", "aerial_fh_pdump_client_thread"],
    },
}

def conditional_print(message, quiet=False):
    if not quiet:
        print(message)

def read_yaml(yaml_file):
    #Read in input yaml
    with open(yaml_file) as fid:
        try:
            yaml_data = yaml.safe_load(fid)
            return yaml_data
        except yaml.YAMLError as exc:
            print(exc)
            return None

#Checks if a set of nested keys exist in the input dictionary
def nested_key_exists(input_dict, key_list):
    dict_ptr = input_dict
    for key in key_list:
        # Check if key contains array indexing notation like 'transport[0]'
        match = re.match(r'^(.+)\[(\d+)\]$', key)
        if match:
            # Extract the key name and index
            key_name = match.group(1)
            index = int(match.group(2))
            dict_ptr = dict_ptr.get(key_name)
            if dict_ptr is None or not isinstance(dict_ptr, list) or index >= len(dict_ptr):
                return False
            dict_ptr = dict_ptr[index]
        else:
            dict_ptr = dict_ptr.get(key)
            if dict_ptr is None:
                return False
    return True

def nested_set(input_dict, key_list, value):
    dict_ptr = input_dict

    # Navigate through all keys except the last one
    for key in key_list[:-1]:
        # Check if key contains array indexing notation like 'transport[0]'
        match = re.match(r'^(.+)\[(\d+)\]$', key)
        if match:
            # Extract the key name and index
            key_name = match.group(1)
            index = int(match.group(2))
            dict_ptr = dict_ptr.get(key_name)
            if dict_ptr is None or not isinstance(dict_ptr, list) or index >= len(dict_ptr):
                return False
            dict_ptr = dict_ptr[index]
        else:
            dict_ptr = dict_ptr.get(key)
            if dict_ptr is None:
                return False

    # Handle the last key (where we set the value)
    last_key = key_list[-1]
    match = re.match(r'^(.+)\[(\d+)\]$', last_key)
    if match:
        # Extract the key name and index
        key_name = match.group(1)
        index = int(match.group(2))
        list_obj = dict_ptr.get(key_name)
        if list_obj is None or not isinstance(list_obj, list) or index >= len(list_obj):
            return False
        list_obj[index] = value
    else:
        dict_ptr[last_key] = value

    return True

def overwrite_yaml(yaml_file, overrides, input_data, quiet):
    # Read input yaml
    conditional_print(f"Reading {yaml_file}...", quiet)
    output_data = read_yaml(yaml_file)
    if not output_data:
        print(f"ERROR :: Failed to read YAML file={yaml_file}")
        return False

    #Override keys in existing data
    conditional_print(f"Setting values in output file {yaml_file}", quiet)
    for key, override_keys in overrides.items():
        if key not in input_data:
            print(f"WARNING :: Could not find {key} in input yaml. Will not set {override_keys} in output file {yaml_file}.")
            continue

        if isinstance(override_keys, str):
            override_keys = [override_keys]

        if isinstance(override_keys, list):
            success = nested_set(output_data, override_keys, input_data[key])
            if success:
                conditional_print(f"Setting {override_keys} to {input_data[key]}", quiet)
            else:
                print(f"WARNING :: Failed to set {override_keys} in output yaml {yaml_file}")
        elif isinstance(override_keys, tuple):
            for sub_keys in override_keys:
                success = nested_set(output_data, sub_keys, input_data[key])
                if success:
                    conditional_print(f"Setting {sub_keys} to {input_data[key]}", quiet)
                else:
                    print(f"WARNING :: Failed to set {sub_keys} in output yaml {yaml_file}")

    #Write output yaml
    conditional_print(f"Writing {yaml_file}...", quiet)
    with open(yaml_file,'w') as fid:
        try:
            yaml.dump(output_data, fid, explicit_start=True, explicit_end=True, default_flow_style=None, width=float("inf"), sort_keys=False, default_style='')
        except yaml.YAMLError as exc:
            print(exc)
            return False

    return True

def main(args):
    global config_mappings

    conditional_print(f"Reading {args.input_yaml}...", args.quiet)
    input_data = read_yaml(args.input_yaml)
    if not input_data:
        print(f"ERROR :: Failed to read input_yaml={args.input_yaml}")
        return 1

    for yaml_file, config_key in zip(
        [args.cuphycontroller_yaml, args.l2adapter_yaml, args.testmac_yaml, args.testmac1_yaml, args.multi_nvipc_yaml, args.ru_emulator_yaml],
        ["cuphycontroller", "l2adapter", "testmac", "testmac1", "multi_nvipc", "ru_emulator"],
    ):
        if yaml_file:
            success = overwrite_yaml(yaml_file, config_mappings[config_key], input_data, args.quiet)
            if not success:
                print(f"Failed to write {config_key} in file ({yaml_file})")
                return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Using input L1 logical core yaml, produces a physical core yaml based on system configuration.
        """
    )
    parser.add_argument(
        "input_yaml", help="Input physical core yaml"
    )
    parser.add_argument(
        "-c", "--cuphycontroller_yaml", help="Cuphycontroller yaml to edit"
    )
    parser.add_argument(
        "-l", "--l2adapter_yaml", help="L2adapter yaml to edit"
    )
    parser.add_argument(
        "-t", "--testmac_yaml", help="TestMAC yaml to edit"
    )
    parser.add_argument(
        "-1", "--testmac1_yaml", help="The second TestMAC yaml to edit"
    )
    parser.add_argument(
        "-m", "--multi_nvipc_yaml", help="Multi_NVIPC yaml to edit"
    )
    parser.add_argument(
        "-r", "--ru_emulator_yaml", help="Ru_emulator yaml to edit"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Silence all print logs except for warnings and errors"
    )
    args = parser.parse_args()

    main(args)