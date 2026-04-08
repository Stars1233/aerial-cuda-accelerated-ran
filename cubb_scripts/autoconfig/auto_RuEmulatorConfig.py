#!/usr/bin/env python3

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
import glob
import h5py as h5
import json
import numpy as np
import os
import re
import sys
import yaml

yaml.Dumper.ignore_aliases = lambda *args : True




eAxC_key_translation = {
    "eAxC_id_ssb_pbch" : "eAxC_DL",
    "eAxC_id_pdcch" : "eAxC_DL",
    "eAxC_id_pdsch" : "eAxC_DL",
    "eAxC_id_csirs" : "eAxC_DL",
    "eAxC_id_prach" : "eAxC_prach_list",
    "eAxC_id_pucch" : "eAxC_UL",
    "eAxC_id_pusch" : "eAxC_UL",
    "eAxC_id_srs" : "eAxC_srs_list",
}

ul_channels = ["eAxC_prach_list","eAxC_UL","eAxC_srs_list"]
dl_channels = ["eAxC_DL"]

def generate_ru_config_files(input_dir, output_dir, ru_emulator_template_file, test_case=None):
    count = 0

    # cupycontroller config files
    cuphyConfigs = glob.glob(os.path.join(input_dir, 'cuphycontroller_*.yaml'))
    for cuphyCfg in cuphyConfigs:
        if test_case is None or test_case in cuphyCfg:
            try:
                with open(cuphyCfg, 'r') as cuphyCfg_content:
                    cuphyCfg_yaml_file = yaml.safe_load(cuphyCfg_content)
                    with open(ru_emulator_template_file, 'r') as ru_emulator_template_content:
                        ru_emulator_yaml_file = yaml.safe_load(ru_emulator_template_content)

                        ru_emulator_cell_count = len(ru_emulator_yaml_file['ru_emulator']['cell_configs'])
                        cuphyCfg_cell_count = len(cuphyCfg_yaml_file['cuphydriver_config']['cells'])
                        cell_count = min(ru_emulator_cell_count, cuphyCfg_cell_count)

                        ru_emulator_yaml_file['ru_emulator']['enable_mmimo'] = cuphyCfg_yaml_file['cuphydriver_config']['mMIMO_enable']

                        try:
                            ru_emulator_yaml_file['ru_emulator']['fix_beta_dl'] = cuphyCfg_yaml_file['cuphydriver_config']['fix_beta_dl']
                        except:
                            ru_emulator_yaml_file['ru_emulator']['fix_beta_dl'] = 0

                        ru_emulator_yaml_file['ru_emulator']['aerial_fh_mtu'] = cuphyCfg_yaml_file['cuphydriver_config']['nics'][0]['mtu']

                        for cell_idx in range(cell_count):
                            #eAxC update
                            for cuphyKey, ruKey in eAxC_key_translation.items():
                                ru_emulator_yaml_file['ru_emulator']['cell_configs'][cell_idx][ruKey]= cuphyCfg_yaml_file['cuphydriver_config']['cells'][cell_idx][cuphyKey]
                                if len(ru_emulator_yaml_file['ru_emulator']['cell_configs'][cell_idx][ruKey]) >= 8:
                                    if ruKey in dl_channels:
                                        ru_emulator_yaml_file['ru_emulator']['aerial_fh_rxq_size'] = 512
                                    else:
                                        ru_emulator_yaml_file['ru_emulator']['aerial_fh_txq_size'] = 512

                            ru_emulator_yaml_file['ru_emulator']['cell_configs'][cell_idx]['dl_iq_data_fmt']['comp_meth'] = cuphyCfg_yaml_file['cuphydriver_config']['cells'][cell_idx]['dl_iq_data_fmt']['comp_meth']
                            ru_emulator_yaml_file['ru_emulator']['cell_configs'][cell_idx]['ul_iq_data_fmt']['comp_meth'] = cuphyCfg_yaml_file['cuphydriver_config']['cells'][cell_idx]['ul_iq_data_fmt']['comp_meth']
                            ru_emulator_yaml_file['ru_emulator']['cell_configs'][cell_idx]['dl_iq_data_fmt']['bit_width'] = cuphyCfg_yaml_file['cuphydriver_config']['cells'][cell_idx]['dl_iq_data_fmt']['bit_width']
                            ru_emulator_yaml_file['ru_emulator']['cell_configs'][cell_idx]['ul_iq_data_fmt']['bit_width'] = cuphyCfg_yaml_file['cuphydriver_config']['cells'][cell_idx]['ul_iq_data_fmt']['bit_width']
                            ru_emulator_yaml_file['ru_emulator']['cell_configs'][cell_idx]['ru_type'] = cuphyCfg_yaml_file['cuphydriver_config']['cells'][cell_idx]['ru_type']


                        ##UL/DL TIMING CONFIGURATION
                        # Please note this couples with auto_controllerConfig.py

                        #Decouple these values as much as possible - RU transmit is configured based on cuphycontroller configuration
                        #ULU timing
                        ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['ul_u_plane_tx_offset'] = cuphyCfg_yaml_file['cuphydriver_config']['cells'][0]['ul_u_plane_tx_offset_ns']//1000
                        #DLU/DLC/ULC timing
                        dlu_offset = cuphyCfg_yaml_file['cuphydriver_config']['cells'][0]['T1a_max_up_ns']//1000
                        ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['dl_u_plane_timing_delay'] = dlu_offset
                        ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['dl_c_plane_timing_delay'] = dlu_offset + cuphyCfg_yaml_file['cuphydriver_config']['cells'][0]['Tcp_adv_dl_ns']//1000
                        ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['ul_c_plane_timing_delay'] = cuphyCfg_yaml_file['cuphydriver_config']['cells'][0]['T1a_max_cp_ul_ns']//1000

                        #Hard-coded SRS transmit offset
                        #Note - this does not live in cuphycontroller configuration
                        if 'F08' in cuphyCfg:
                            ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['ul_u_plane_tx_offset_srs'] = 521
                            ru_emulator_yaml_file['ru_emulator']['enable_srs_eaxcid_pacing'] = 1
                        else:
                            ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['ul_u_plane_tx_offset_srs'] = 1400
                            ru_emulator_yaml_file['ru_emulator']['enable_srs_eaxcid_pacing'] = 0

                        #Hard-coded DLU reception window
                        ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['dl_u_plane_window_size'] = 51

                        #Hard-coded DLC/ULC reception window
                        if int(ru_emulator_yaml_file['ru_emulator']['enable_mmimo']) == 1:

                            #DLC/ULC reception windows for MMIMO
                            ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['dl_c_plane_window_size'] = 250
                            ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['ul_c_plane_window_size'] = 250

                            #Split RX/TX mempool
                            ru_emulator_yaml_file['ru_emulator']['aerial_fh_split_rx_tx_mempool'] = 1

                        else:

                            #DLC/ULC reception windows for 4TR
                            ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['dl_c_plane_window_size'] = 51
                            ru_emulator_yaml_file['ru_emulator']['oran_timing_info']['ul_c_plane_window_size'] = 51

                        l2adapter_Cfg = os.path.join(input_dir, cuphyCfg_yaml_file['l2adapter_filename'])
                        if os.path.exists(l2adapter_Cfg):
                            with open(l2adapter_Cfg, 'r') as l2adapter_Cfg_content:
                                l2adapter_Cfg_yaml_file = yaml.safe_load(l2adapter_Cfg_content)

                                if 'F08' in cuphyCfg:
                                    ru_emulator_yaml_file['ru_emulator']['dl_approx_validation'] = 0
                                elif 'enable_precoding' in l2adapter_Cfg_yaml_file:
                                    ru_emulator_yaml_file['ru_emulator']['dl_approx_validation'] = l2adapter_Cfg_yaml_file['enable_precoding']
                                else:
                                    print(f'No enable_precoding cfg in {l2adapter_Cfg}, ignore it')
                                if 'enable_beam_forming' in l2adapter_Cfg_yaml_file:
                                    ru_emulator_yaml_file['ru_emulator']['enable_beam_forming'] = l2adapter_Cfg_yaml_file['enable_beam_forming']
                                else:
                                    print(f'No enable_beam_forming cfg in {l2adapter_Cfg}, ignore it')
                        else:
                            print(f'{l2adapter_Cfg} does not exist')

                        if '90502' in cuphyCfg:
                            if len(ru_emulator_yaml_file['ru_emulator']['dl_core_list']) < 16:
                                elements_needed = 16 - len(ru_emulator_yaml_file['ru_emulator']['dl_core_list'])
                                ru_emulator_yaml_file['ru_emulator']['dl_core_list'] = ru_emulator_yaml_file['ru_emulator']['ul_core_list'][-elements_needed:] + ru_emulator_yaml_file['ru_emulator']['dl_core_list']
                                ru_emulator_yaml_file['ru_emulator']['ul_core_list'] = ru_emulator_yaml_file['ru_emulator']['ul_core_list'][:-elements_needed]
                                # Print the updated lists
                                print("90502 ul_core_list:", ru_emulator_yaml_file['ru_emulator']['ul_core_list'])
                                print("90502 dl_core_list:", ru_emulator_yaml_file['ru_emulator']['dl_core_list'])

                        if '90083' in cuphyCfg:
                            if len(ru_emulator_yaml_file['ru_emulator']['ul_core_list']) < 18:
                                elements_needed = 18 - len(ru_emulator_yaml_file['ru_emulator']['ul_core_list'])
                                ru_emulator_yaml_file['ru_emulator']['ul_core_list'] = ru_emulator_yaml_file['ru_emulator']['ul_core_list'] + ru_emulator_yaml_file['ru_emulator']['dl_core_list'][:elements_needed]
                                ru_emulator_yaml_file['ru_emulator']['dl_core_list'] = ru_emulator_yaml_file['ru_emulator']['dl_core_list'][elements_needed:]
                                # Print the updated lists
                                print("90083 ul_core_list:", ru_emulator_yaml_file['ru_emulator']['ul_core_list'])
                                print("90083 dl_core_list:", ru_emulator_yaml_file['ru_emulator']['dl_core_list'])

                        filename = os.path.basename(cuphyCfg).replace('cuphycontroller_','ru_emulator_config_')
                        with open(os.path.join(output_dir, filename), 'w+') as stream:
                            yaml.dump(ru_emulator_yaml_file, stream, explicit_start=True, explicit_end=True, default_flow_style=None, width=float("inf"), sort_keys=False ,default_style='')

                count += 1

            except KeyboardInterrupt:
                raise
            except Exception as x:
                print(f'Exception raised for {cuphyCfg}', x)
                continue

    print('Total RuEmulator config files generated ' + str(count))

if __name__ == "__main__":

    script_dir = os.path.realpath(os.path.dirname(__file__))
    cuBB_SDK=os.environ.get("cuBB_SDK", os.path.normpath(os.path.join(script_dir, '../..')))
    CUBB_HOME=os.environ.get("CUBB_HOME", cuBB_SDK)

    RU_EMULATOR_CONFIG_FILE_TEMPLATE = os.path.join(cuBB_SDK, 'cuPHY-CP/ru-emulator/config/config.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help='Input Directory of cuPHYController configurations')
    parser.add_argument("-o", '--output_dir', default=None , help='Output Directory for the Ru Emualtor Configurations')
    parser.add_argument("-t", '--ru_emulator_template_file', default=RU_EMULATOR_CONFIG_FILE_TEMPLATE , help='ru_emulator Config File Template')
    parser.add_argument("-c", '--test_case', help='Specify an nrSim testcase to generate config files for, otherwise generate for all launch_pattern_xyx.yaml files')
    args = parser.parse_args()
    # process template file argument

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir 

    if not os.path.isdir(args.input_dir):
        exit(f"InvalidPathError: The path {args.input_dir} does not exist \nExiting...")
    if not os.path.isdir(output_dir):
        exit(f"InvalidPathError: The path {output_dir} does not exist \nExiting...")

    generate_ru_config_files(args.input_dir, output_dir, args.ru_emulator_template_file, args.test_case)
