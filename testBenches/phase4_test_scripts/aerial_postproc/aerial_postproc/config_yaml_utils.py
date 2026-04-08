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

import yaml
import os
from os import path
import sys
import pandas as pd

def read_yaml_content(file):
    if not path.exists(file) or not path.isfile(file):
        print(f'Error loading file {file}', file=sys.stderr)
        return False
    with open(file, 'r') as stream:
        try:
            content = yaml.safe_load(stream)
            return content
        except yaml.YAMLError as yaml_exc:
            print(f'Error opening yaml file {content}', file=sys.stderr) 
            print(yaml_exc)
            return False
        
def get_scenario_cell_capacity_from_cuphy_config(cuphy_config_filename):
    yaml_content = read_yaml_content(cuphy_config_filename)
    if yaml_content:
        scenario_cell_capacity = int(yaml_content['cuphydriver_config']['cell_group_num'])
    else:
        scenario_cell_capacity = None
    return scenario_cell_capacity

def get_cuphy_controller_file(scenario_folder):
    if not path.exists(scenario_folder) or not path.isdir(scenario_folder):
        print(f'Not valid folder {scenario_folder}', file=sys.stderr)
        return False
    
    cuphy_config_folder_path = os.path.join(scenario_folder, 'cuphy', 'configs')        
    if not path.exists(cuphy_config_folder_path) or not path.isdir(cuphy_config_folder_path):
        print(f'Not valid folder {cuphy_config_folder_path}', file=sys.stderr)
        return False
    
    cuphy_config_folder_files = os.listdir(cuphy_config_folder_path)
    cuphy_controller_file = None
    for candidate_file in cuphy_config_folder_files:
        if not path.isfile(path.join(cuphy_config_folder_path, candidate_file)):
            continue
        if candidate_file.startswith('cuphycontroller_') and candidate_file.endswith('.yaml'):
            cuphy_controller_file = os.path.join(cuphy_config_folder_path, candidate_file)
            break
    
    if not cuphy_controller_file:
        print(f'Could not find a cuphycontroller_*.yaml file inside in {cuphy_config_folder_path}', file=sys.stderr)
        return False
    return cuphy_controller_file

def get_scenario_cell_capacity_from_scenario_folder(scenario_folder):
    cuphy_controller_file = get_cuphy_controller_file(scenario_folder)
    if cuphy_controller_file:        
        return get_scenario_cell_capacity_from_cuphy_config(cuphy_controller_file)
    return False

def get_sm_provisioning_from_scenario_folder(scenario_folder):
    cuphy_controller_file = get_cuphy_controller_file(scenario_folder)
    if not cuphy_controller_file:
        return False
    yaml_content = read_yaml_content(cuphy_controller_file)
    if yaml_content:
        keys = ['mps_sm_pusch', 'mps_sm_pucch', 'mps_sm_prach', 'mps_sm_ul_order', 'mps_sm_pdsch', 'mps_sm_pdcch', 'mps_sm_pbch', 'mps_sm_gpu_comms']
        parent_element = yaml_content['cuphydriver_config']
        sm_provisioning = {
            key.replace('mps_sm_','').upper(): int(parent_element[key]) for key in keys
        }
    else:
        sm_provisioning = None

    df_sm = None
    if sm_provisioning is not None:
        df_sm = pd.DataFrame({
            'channel': list(sm_provisioning.keys()),
            'sm'     : list(sm_provisioning.values())
        })
    return df_sm

def get_timing_delays_from_scenario_folder(scenario_folder):
    cuphy_controller_file = get_cuphy_controller_file(scenario_folder)
    if not cuphy_controller_file:
        return False
    yaml_content = read_yaml_content(cuphy_controller_file)
    if yaml_content:
        keys = ['T1a_max_up_ns', 'T1a_max_cp_ul_ns', 'Ta4_min_ns', 'Ta4_max_ns', 'Tcp_adv_dl_ns', 'ul_u_plane_tx_offset_ns']        
        parent_element = yaml_content['cuphydriver_config']['cells'][0] # go to O-RU 0
        timings = {
            key: int(parent_element[key])//1000 for key in keys
        }
        timings['dl_c_plane_timing_delay'] = timings['T1a_max_up_ns'] + timings['Tcp_adv_dl_ns']
        timings['ul_c_plane_timing_delay'] = timings['T1a_max_cp_ul_ns']
        timings['dl_u_plane_timing_delay'] = timings['T1a_max_up_ns']
        timings['dl_c_plane_windows_size'] = timings['Ta4_min_ns']
        timings['ul_c_plane_windows_size'] = timings['Ta4_min_ns']
        timings['dl_u_plane_windows_size'] = timings['Ta4_min_ns']
    else:
        timings = False
    return timings
