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

def conditional_print(message, quiet=False):
    if not quiet:
        print(message)

#Parse a string list of integer values (all ranges inclusive)
#Example "1-4,7,9-11" --> [1,2,3,4,7,9,10,11]
def parse_cpu_list(s):
    core_list = []
    s = s.strip()
    if not s:  # check and return if the string is empty
        return core_list

    for s_comma in s.split(','):
        s_hyphon = s_comma.split('-')
        if len(s_hyphon) == 1:
            core_list.append(int(s_comma))
        else:
            core_list.extend(range(int(s_hyphon[0]),int(s_hyphon[1])+1))
    core_list.sort()
    return core_list

#Opens a cpu specification file and parses to list of integers (see parse_cpu_list)
def get_cores_from_file(filename):
    core_list = []
    with open(filename) as f:
        cores_string = f.read()
        #conditional_print(f"Core string: {cores_string}", args.quiet)
        core_list = parse_cpu_list(cores_string)
        #conditional_print(f"Core list: {core_list}", args.quiet)
        #conditional_print(f"-----------------------------------", args.quiet)

    return core_list

def parse_online_cpu_list():
    """Parse sysfs file /sys/devices/system/cpu/online information into list of integers"""
    cpu_str = open('/sys/devices/system/cpu/online').read().strip()
    cpus = []
    for part in cpu_str.strip().split(','):
        if '-' in part:
            start, end = map(int, part.split('-', 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(cpus)

def get_ht_core_mapping(core_list=None):
    if core_list is None:
        core_list = [k for k in parse_online_cpu_list()]

    #Create dictionary that maps core number to sibling list
    ht_core_mapping = {}
    for core in core_list:
        sibling_cpus = get_cores_from_file(f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list')
        ht_core_mapping[core] = sibling_cpus

    return ht_core_mapping


def get_isolated_cpus_for_rhocp():
    """Get isolated CPUs for RHCOS/RHOCP platforms using process affinity.
    
    On RHCOS/RHOCP platforms where isolated cores may not be available via /sys,
    Read the current process affinity as a fallback. Because cores are explicitly 
    reserved for cuphycontroller and testmac containers when aerial deploys them.
    
    Returns:
        List of CPU IDs from process affinity, sorted. 
        Empty list if not on RHCOS platform or if reading affinity fails.
    """
    isolated_cpus = []
    
    # Check if running on RHCOS platform
    try:
        with open("/proc/cmdline", "r", encoding="utf-8") as _f:
            _is_rhcos = ("rhcos" in _f.read().lower())
    except Exception:
        _is_rhcos = False
    
    if _is_rhcos:
        conditional_print("isolated_cpus empty and it is rhocp platform, using process affinity instead", args.quiet)
        try:
            isolated_cpus = sorted(os.sched_getaffinity(0))
        except Exception as e:
            conditional_print("failed to get current process affinity on rhocp platform: %s"%e, args.quiet)
            isolated_cpus = []
    
    return isolated_cpus


def get_available_physical_cores(numa_node):
    conditional_print("Finding available cores for numa_node %s"%(numa_node), args.quiet)

    #Determine what cores are available to be used for the affinity of this python script
    all_cores = [k for k in parse_online_cpu_list()]
    all_cores.sort()
    conditional_print("all_cores = %s"%all_cores, args.quiet)

    #Compare this with list available from sysfs (this should be a subset)
    sysfs_cpus = [int(k.path.split('/sys/devices/system/cpu/cpu')[1]) for k in os.scandir('/sys/devices/system/cpu/')
                if k.is_dir() and
                   len(k.path.split('/sys/devices/system/cpu/cpu')) == 2 and
                   k.path.split('/sys/devices/system/cpu/cpu')[1].isnumeric()]
    sysfs_cpus.sort()
    all_cores_verified = all([aa in sysfs_cpus for aa in all_cores])
    if not all_cores_verified:
        print("WARNING :: mismatch in sysfs cores - expectation is we should see a subset of sysfs cpus, all_cores=%s, sysfs_cpus=%s"%(all_cores,sysfs_cpus))

    ht_core_mapping = get_ht_core_mapping(all_cores)
    conditional_print("ht_core_mapping=%s"%ht_core_mapping, args.quiet)

    #Determine if HT is enabled
    ht_enabled = all([len(ht_core_mapping[key])>1 for key in ht_core_mapping.keys()])
    if ht_enabled:
        conditional_print(f'Detected HT Enabled', args.quiet)
    else:
        conditional_print(f'Detected HT Disabled', args.quiet)

    # Determine if system has multiple NUMA nodes
    all_numa_nodes = [int(k.path.split('/sys/devices/system/node/node')[1]) for k in os.scandir('/sys/devices/system/node')
                      if k.is_dir() and
                      len(k.path.split('/sys/devices/system/node/node')) == 2
                      and k.path.split('/sys/devices/system/node/node')[1].isnumeric()]
    all_numa_nodes.sort()
    multiple_numa_enabled = len(all_numa_nodes) > 1
    if multiple_numa_enabled:
        conditional_print(f'Detected Multiple NUMA Nodes: {all_numa_nodes}', args.quiet)
    else:
        conditional_print(f'Detected Single NUMA Node.', args.quiet)
    conditional_print("all_numa_nodes = %s"%all_numa_nodes, args.quiet)

    #Filters those cores down to only those on specified numa
    numa_cores = [k for k in all_cores if os.path.isdir(f'/sys/devices/system/node/node{numa_node}/cpu{k}')]
    numa_cores.sort()
    conditional_print("numa_cores (numa=%i) = %s"%(numa_node, numa_cores), args.quiet)

    #Determines what cores are isolated
    isolated_cpus = get_cores_from_file('/sys/devices/system/cpu/isolated')
    conditional_print("isolated_cpus = %s"%isolated_cpus, args.quiet)

    # when isolated core read is empty and it is rhocp (core OS), use current process (cuphycontroller/testmac) affinity instead
    # because we reserve cores explicitly for cuphycontroller and testmac containers when deployed in cicd.
    if not isolated_cpus:
        isolated_cpus = get_isolated_cpus_for_rhocp()

    #Filter based on cores that have both themselves and sibling cores isolated
    primary_core_list = list(set([ht_core_mapping[key][0] for key in ht_core_mapping.keys()]))
    numa_primary_core_list = [aa for aa in primary_core_list if aa in numa_cores]
    isolated_numa_primary_core_list = [aa for aa in numa_primary_core_list if all([bb in isolated_cpus for bb in ht_core_mapping[aa]])]
    isolated_numa_primary_core_list.sort()
    conditional_print("primary_core_list=%s"%primary_core_list, args.quiet)
    conditional_print("numa_primary_core_list=%s"%numa_primary_core_list, args.quiet)
    conditional_print("isolated_numa_primary_core_list=%s"%isolated_numa_primary_core_list, args.quiet)

    return isolated_numa_primary_core_list

#All of the logical cores are defined in yaml as
# <logical core>.<logical HT sibling>.<numa>
# The specification of the last two components are optional, which means that the
# yaml core information can be read in as an int, float, or string
#
#So options are:
#<logical core> -> int
#<logical core>.<logical HT sibling> -> float
#<logical core>.<logical HT sibling>.<numa> -> str
# -1 -> int (no core specified - this is a typical way to disable threads)
def yamlcore2logicaltuple(yamlcore):
    # conditional_print("yamlcore=%s (%s)"%(yamlcore,type(yamlcore)), args.quiet)
    
    if(type(yamlcore) == str):
        #  str example: 0.1.2 would mean physical core 0, sibling 1, numa 2
        ss = yamlcore.split(".")
        return (int(ss[0]),int(ss[1]),int(ss[2]))
    elif(type(yamlcore) == int or (type(yamlcore) == float and yamlcore%1.0 == 0.0)):
        #  int example: 5 would mean physical core 5, sibling 0, numa 0
        #  float example: 5.0 would mean physical core 5, sibling 0, numa 0
        return (int(yamlcore),0,0)
    else:
        #  float example: 5.1 would mean physical core 5, sibling 1, numa 0
        return (int(yamlcore),1,0)

def logicaltuple2yamlcore(logicaltuple):
    return "%s.%s.%s"%(logicaltuple[0],logicaltuple[1],logicaltuple[0])

def read_full_core_config(input_yaml):
    #Read in input yaml
    conditional_print("Reading %s..."%input_yaml, args.quiet)
    with open(input_yaml) as fid:
        try:
            yaml_data = yaml.safe_load(fid)
        except yaml.YAMLError as exc:
            print(exc)
            return False

    # Check if this is a LOOPBACK configuration
    is_loopback = 'LOOPBACK' in os.path.basename(input_yaml).upper()
    conditional_print("is_loopback=%s"%is_loopback, args.quiet)

    #Determine which keys are for each category
    testmac_keys = [aa for aa in yaml_data.keys() if aa.lower().startswith("tbonly_tm_")]
    ru_emulator_keys = [aa for aa in yaml_data.keys() if (aa.lower().startswith("tbonly_re_"))]
    l1_keys = [aa for aa in yaml_data.keys() if aa not in testmac_keys and aa not in ru_emulator_keys]
    all_keys = l1_keys+testmac_keys+ru_emulator_keys
    conditional_print("l1_keys=%s"%l1_keys, args.quiet)
    conditional_print("testmac_keys=%s"%testmac_keys, args.quiet)
    conditional_print("ru_emulator_keys=%s"%ru_emulator_keys, args.quiet)

    #Determine the unique set of logical cores, ignoring sibling information
    #Each input entry here is (logical core,logical HT sibling)
    #We are just plucking off the "logical core" part
    du_logical_core_map = {}
    ru_logical_core_map = {}
    for key in all_keys:
        val = yaml_data[key]
        if type(val) == list:
            core_list = val
        else:
            core_list = [val]

        # For LOOPBACK configs, put all keys in DU map (so RU shares cores with DU)
        # For non-LOOPBACK configs, separate DU and RU
        if is_loopback:
            # Put everything in DU map for loopback
            du_logical_core_map[key] = []
            for core in core_list:
                ltuple = yamlcore2logicaltuple(core)
                du_logical_core_map[key].append(ltuple)
        else:
            #Append to DU if L1 or testmac key
            if key in l1_keys+testmac_keys:
                du_logical_core_map[key] = []
                for core in core_list:
                    ltuple = yamlcore2logicaltuple(core)
                    du_logical_core_map[key].append(ltuple)

            #Append to RU if ru_emulator key
            if key in ru_emulator_keys:
                ru_logical_core_map[key] = []
                for core in core_list:
                    ltuple = yamlcore2logicaltuple(core)
                    ru_logical_core_map[key].append(ltuple)
    return du_logical_core_map,ru_logical_core_map,yaml_data

def allocate_physical_cores(input_yaml,output_yaml,map_ru_emulator=False):

    #input_yaml - yaml to read for logical cores
    #output_yaml - yaml to write physical cores
    #map_ru_emulator - By default (false) outputs file applicable to testmac+cuphycontroller.  Setting to (true) output file applicable to ru_emulator

    #Create dictionary that maps thread name to list of (logical core, sibling core, numa) tuples
    du_logical_core_map,ru_logical_core_map,yaml_data = read_full_core_config(input_yaml)
    conditional_print("du_logical_core_map=%s"%du_logical_core_map, args.quiet)
    conditional_print("ru_logical_core_map=%s"%ru_logical_core_map, args.quiet)

    conditional_print("yaml_data=%s"%yaml_data, args.quiet)

    # Check if this is a LOOPBACK configuration
    is_loopback = 'LOOPBACK' in os.path.basename(input_yaml).upper()
    conditional_print("is_loopback=%s"%is_loopback, args.quiet)

    #Determine number of numas
    du_numas = list(set([aa[2] for key in du_logical_core_map.keys() for aa in du_logical_core_map[key]]))
    ru_numas = list(set([aa[2] for key in ru_logical_core_map.keys() for aa in ru_logical_core_map[key]]))
    conditional_print("du_numas=%s"%du_numas, args.quiet)
    conditional_print("ru_numas=%s"%ru_numas, args.quiet)

    #Determine which keys we are keeping, which keys we are removing
    if(not map_ru_emulator or is_loopback):
        logical_core_map = du_logical_core_map
        numas = du_numas
    else:
        logical_core_map = ru_logical_core_map
        numas = ru_numas
    numas.sort()

    conditional_print("logical_core_map=%s"%logical_core_map, args.quiet)
    conditional_print("numas=%s"%numas, args.quiet)

    #Create map of all available cores for each numa
    #Also make sure we have enough cores on each numa
    logical_tuple_list = [aa for key in logical_core_map.keys() for aa in logical_core_map[key]]
    logical_tuple_list.sort()
    have_enough_cores = True
    numa2physical_list_map = {}
    for numa in numas:
        #Determine number of physical cores required, excluding any negative one values
        num_physical_cores_desired = len(list(set([aa[0] for aa in logical_tuple_list if aa[2] == numa and aa[0] >= 0])))
        physical_core_list = get_available_physical_cores(numa)

        conditional_print("physical_core_list=%s"%physical_core_list, args.quiet)
        conditional_print("numa=%s, desired_core_count=%s, physical_core_count=%s"%(numa,num_physical_cores_desired,len(physical_core_list)), args.quiet)
        if(num_physical_cores_desired > len(physical_core_list)):
            have_enough_cores = False
            print("ERROR:: Not enough physical cores on numa %s.  desired_cores=%s, physical_core_list=%s"%(numa,num_physical_cores_desired,physical_core_list))

        numa2physical_list_map[numa] = physical_core_list
    conditional_print("numa2physical_list_map=%s"%numa2physical_list_map, args.quiet)

    if(not have_enough_cores):
        print("ERROR:: Not enough physical cores to implement logical yaml")
        return False

    #Get full ht mapping across all cores
    #Maps core to list of core siblings (for all cores across all numas)
    ht_core_mapping = get_ht_core_mapping()
    conditional_print("ht_core_mapping=%s"%ht_core_mapping, args.quiet)

    #Warn if HT configuration used on nonHT system
    config_ht_enabled = any([aa[1]>0 for aa in logical_tuple_list])
    system_ht_enabled = all([len(ht_core_mapping[key]) >1 for key in ht_core_mapping.keys()])
    use_siblings = config_ht_enabled and system_ht_enabled
    if(config_ht_enabled and not system_ht_enabled):
        print("WARNING :: HT configuration specified but system does not have HT enabled.  Sibling core specifications will be ignored")

    #Map logical cores to ht mapping
    for numa in numas:
        #Determine all logical cores on this numa, excluding any negative one values
        logical_cores = list(set([aa[0] for key in logical_core_map.keys() for aa in logical_core_map[key] if aa[2]==numa and aa[0] >= 0]))
        logical_cores.sort()
        conditional_print("logical_cores=%s"%logical_cores, args.quiet)

        #Map logical numa core to physical core
        logical_to_physical_map = {aa:ht_core_mapping[bb] for aa,bb in zip(logical_cores,numa2physical_list_map[numa])}
        negative_logical_cores = list(set([aa[0] for key in logical_core_map.keys() for aa in logical_core_map[key] if aa[2]==numa and aa[0] < 0]))
        for negative_val in negative_logical_cores:
            logical_to_physical_map[negative_val] = [negative_val]
        conditional_print("numa=%s"%numa, args.quiet)
        conditional_print("logical_cores=%s"%logical_cores, args.quiet)
        conditional_print("numa2physical_list_map=%s"%numa2physical_list_map, args.quiet)
        conditional_print("ht_core_mapping=%s"%ht_core_mapping, args.quiet)
        conditional_print("logical_to_physical_map=%s"%logical_to_physical_map, args.quiet)

        for key in logical_core_map.keys():
            data = yaml_data[key]

            #First index in logical tuple is logical cores, second index is ht sibling
            if(type(data) != list):
                ltuple = yamlcore2logicaltuple(data)
                if(ltuple[2]==numa):
                    if(use_siblings):
                        sibling = ltuple[1]
                    else:
                        sibling = 0
                    yaml_data[key] = logical_to_physical_map[ltuple[0]][sibling]
            else:
                for ii in range(len(data)):
                    ltuple = yamlcore2logicaltuple(data[ii])
                    if(ltuple[2]==numa):
                        if(use_siblings):
                            sibling = ltuple[1]
                        else:
                            sibling = 0
                        yaml_data[key][ii] = logical_to_physical_map[ltuple[0]][sibling]

    #Remove data for keys we do not want
    unwanted_keys = [key for key in yaml_data.keys() if key not in logical_core_map.keys()]
    for key in unwanted_keys:
        yaml_data.pop(key)

    #Write output yaml, maintaining ordering
    conditional_print("Writing %s..."%output_yaml, args.quiet)
    with open(output_yaml,'w') as fid:
        try:
            yaml.dump(yaml_data, fid, default_flow_style=None, sort_keys=False)
        except yaml.YAMLError as exc:
            print(exc)
            return False

    return True

def main(args):
    #Determine physical cores to use
    # if(args.cpu_override_list):
    #     print("Using cpu_override_list %s"%args.cpu_override_list)
    #     physical_core_list = parse_cpu_list(args.cpu_override_list)
    #     print("Assuming non-HT implementation (only assumption cpu_override_list allows)")
    #     ht_core_mapping = {aa:[aa] for aa in physical_core_list}
    # else:
    #     #Determine all numa nodes requested based on config
    #     physical_core_list,ht_core_mapping = get_available_physical_cores(0);
    
    #Allocate physical cores based on input logical cores.  Write to output file
    success = allocate_physical_cores(args.input_yaml, args.output_yaml, map_ru_emulator=args.map_ru_emulator)

    if(success and os.path.isfile(args.output_yaml)):
        conditional_print("Successfully generated physical cores.  Output file located at %s"%args.output_yaml, args.quiet)
        return 0
    else:
        print("Failed to generate physical cores.") #always print error, regardless of args.quiet value
        return 1

if __name__ == "__main__":
    default_reserved_cores = 0

    parser = argparse.ArgumentParser(
        description="""
        Using input L1 logical core yaml, produces a physical core yaml based on system configuration.
        """

    )
    parser.add_argument(
        "input_yaml", help="Input logical core yaml"
    )
    parser.add_argument(
        "output_yaml", help="Output physical core yaml"
    )
    parser.add_argument(
        "-e", "--map_ru_emulator", action="store_true", help="When this flag is enabled only ru_emulator cores are mapped.  If disabled only testmac/L1 cores are mapped"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Silence all print logs except for warnings and errors"
    )
    # parser.add_argument(
    #     "-r", "--reserved_cores", default=default_reserved_cores, type=int, help="Number of cores to reserve (skip) before running core allocation algorithm (default %i)"%default_reserved_cores
    # )
    # parser.add_argument(
    #     "-c", "--cpu_override_list", help="If specified forces cores to be allocated to list specified.  Example \"1-4,7,9-11\" --> [1,2,3,4,7,9,10,11]"
    # )
    args = parser.parse_args()

    main(args)