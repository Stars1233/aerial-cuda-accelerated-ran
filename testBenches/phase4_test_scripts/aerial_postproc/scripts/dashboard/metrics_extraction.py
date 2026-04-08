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

import argparse
import pandas as pd
import sys
import os
from os import path
from aerial_postproc.parsenator import Parsenator
from aerial_postproc.config_yaml_utils import get_sm_provisioning_from_scenario_folder
from aerial_postproc.monitor_cpu_cores_parsing import read_phy_cpu_consumption
import numpy as np
import json
from datetime import datetime
import subprocess
from aerial_postproc.logparse import get_ref_t0
from aerial_postproc.logparse import parse_testmac_expected_throughput, parse_testmac_observed_throughput_per_second
from aerial_postproc.logparse import parse_ru_observed_throughput_per_second

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def read_file_first_line(file):
    if not path.isfile(file):
        print(f'File {file} does not exist.', file=sys.stderr)
        return None
    with open(file, 'r') as f:
        first_line = f.readline().replace('\n','')
        return first_line


def remove_trailing_separator(folder_path):
    if folder_path.endswith(path.sep):
        folder_path = folder_path[:-len(path.sep)]
    return folder_path

def is_dual_port_test(ru_config):
    nic_text = subprocess.check_output(f"grep \"nic:\" {ru_config} | sort | uniq -c",shell=True)
    is_dual_port = len(nic_text.splitlines()) >= 2
    return is_dual_port

def read_jenkins_test_information(scenario_folder, phase):
    # Failure to read identifiers of the test (pipeline, test_case) will lead
    # to the exit -1 of this python script.
    metadata_file_path = path.join(scenario_folder, 'metadata.txt')
    if not path.isfile(metadata_file_path):
        raise ValueError(f'Failed to read {metadata_file_path}. Exiting...')
    
    pipeline = None
    tc_name = None
    jenkins_id = None
    test_duration = None
    merge_request = None
    jenkins_test_result = None    
    with open(metadata_file_path, 'r') as metadata_file:
        lines = metadata_file.read().splitlines()
        for line in lines:
            tokens = line.split('=')
            if len(tokens) == 2:
                if 'jenkins_pipeline' == tokens[0]:
                    pipeline = tokens[1]
                elif 'test_case' == tokens[0]:
                    tc_name = tokens[1]
                elif 'jenkins_id' == tokens[0]:
                    jenkins_id = tokens[1]
                elif 'test_duration' == tokens[0]:
                    test_duration = tokens[1]
                elif 'mr' == tokens[0]:
                    merge_request = tokens[1]
                elif 'jenkins_test_result' == tokens[0]:
                    jenkins_test_result = tokens[1]
            
    if None in [pipeline, tc_name, jenkins_id]:
        raise ValueError('Either of pipeline, test_case, test_result was incorrectly set. Exiting...')
    
    phase_info = dict()
    try:
        tc_name_tokens = tc_name.split("_")
        if phase == 'phase_2':
            # channel_tv_params            
            phase_info['channel'] = tc_name_tokens[0].replace('run','')
            phase_info['tv'] = '_'.join(tc_name_tokens[1:])
        elif phase == 'phase_4':
            phase_info['cell_count'] = int(tc_name_tokens[2].replace("C",""))
            phase_info['pattern'] = tc_name_tokens[3]
            phase_info['bfp'] = int(tc_name_tokens[4].replace("BFP",""))
            phase_info['dual_port'] = int(tc_name_tokens[-1].replace("P", "")) > 1
            phase_info['eh'] = tc_name.find("_EH") >= 0
            phase_info['gc'] = tc_name.find("_GC") >= 0
    except Exception as e:
        raise ValueError(f'Test case improper format: {tc_name}. Error: {e}')
    
    try:
        tokens = jenkins_id.split('-')
        j_datetime = f'{tokens[1]}-{tokens[2]}'
        dt = datetime.strptime(j_datetime, "%Y%m%d-%H%M%S")
        formatted_datetime = f'{dt.year:04d}/{dt.month:02d}/{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}'

        tokens = tokens[0].split('_')
        jenkins_id = int(tokens[-1])
        platform   = '_'.join(tokens[:-1])
    except Exception as e:
        raise ValueError(f'Jenkins id improper format: {jenkins_id}. Error: {e}')
    
    ##############################################################
    ##### Additional information should not fail the script. #####
    ##############################################################
    additional_info = []
    if phase == 'phase_4':
        if test_duration is None:
            test_duration = -1
            additional_info.append('test_duration property was missing.')
        else:    
            try:
                test_duration = int(test_duration)
            except Exception as e:
                additional_info.append(f'Failure to parse {test_duration}')
                test_duration = -1
        phase_info['test_duration'] = test_duration
    
        if merge_request is None:
            merge_request = 'N/A'
            additional_info.append('mr property was missing.')
        else:
            # aerial_sdk:2348
            tokens = merge_request.split(':')
            if tokens[0] == 'aerial_sdk' and len(tokens)==2:
                merge_request = tokens[1]
        phase_info['merge_request'] = merge_request

    if jenkins_test_result is None:
        jenkins_test_result = 'Unknown'
        additional_info.append('jenkins_test_result property was missing.')
    phase_info['result'] = jenkins_test_result

    if phase == 'phase_4':
        power_enabled = True
        csv_file = None
        try:
            # power.csv exists, is valid CSV file, and has more than one rows
            csv_file = path.join(scenario_folder, 'cuphy', 'perf_results', 'power.csv')
            power_enabled &= len(pd.read_csv(csv_file)) > 0

            # power_summary.csv exists, is valid CSV file, and has more than one rows
            csv_file = path.join(scenario_folder, 'cuphy', 'perf_results', 'power_summary.csv')
            power_enabled &= len(pd.read_csv(csv_file)) > 0
        except Exception:
            print(f'Failure reading csv file {csv_file}. Setting power_enabled flag to False')
            base_name = os.path.basename(csv_file)
            additional_info.append(f'{base_name} file was missing')
            power_enabled = False
        finally:
            print(f'Power measurements enabled: {power_enabled}')
        phase_info['power_enabled'] = power_enabled
    
    jenkins_info = {
        'jenkins_pipeline': pipeline,
        'jenkins_id': jenkins_id,
        'test_name': tc_name,
        'gpu_platform': platform,
        'jenkins_run_datetime':formatted_datetime,
        **phase_info
    }
    return jenkins_info, additional_info

def validate_log_folder(scenario_folder):
    if not path.isdir(scenario_folder):
        print(f'ERROR :: {scenario_folder} log folder path {scenario_folder} is not a folder.')
        return False
    return True

def get_perfcsv_from_test_scenario(scenario_folder):
    perf_csv_path = path.join(scenario_folder, 'cuphy', 'perf_results', 'perf.csv')
    if not path.isfile(perf_csv_path):
        print(f'Performance file {perf_csv_path} not exists.', file=sys.stderr)
        return None
    return perf_csv_path

def append_to_json(*args):
    if len(args) == 0:
        return None
    extracted_data = {}
    for (object_to_save, key) in args:
        if type(object_to_save) not in [dict, pd.DataFrame]:
            print('Object to be saved should be either dict or pd.Dataframe', file=sys.stderr)
            sys.exit(1)
        
        if type(object_to_save) == dict:
            extracted_data[key] = object_to_save
            continue
        
        # Else it is a pandas dataframe
        extracted_data[key] = [{'name': column, 'values': object_to_save[column].tolist()} for column in object_to_save.columns.tolist()]
    return extracted_data

def retrieve_headroom_information(scenario_folder, phy_log_filename, ru_log_filename):
    from aerial_postproc.dashboard_utils import retrieve_perf_metrics
    perf_csv_file = get_perfcsv_from_test_scenario(scenario_folder)
    if perf_csv_file is None:
        print(f'Failure to read perf_csv file...', file=sys.stderr)
        sys.exit(1)
    perf_csv_df, perfcsv_dict = retrieve_perf_metrics(perf_csv_file)
    
    BYTES_TO_READ = 1000000
    from aerial_postproc.logparse import build_df_ru_ontime, build_df_du_ontime

    df_ru_ontime = build_df_ru_ontime(ru_log_filename, BYTES_TO_READ)
    df_du_ontime = build_df_du_ontime(phy_log_filename, BYTES_TO_READ)

    def _worst_ontime(df, ontime_type):
        df_filtered = df[df.type == ontime_type]
        if len(df_filtered) == 0:
            raise ValueError(f"No ontime data found for type '{ontime_type}'. Check that the log file contains [FH.PACKET_SUMMARY] or [DRV.*_PACKET_SUMMARY] messages for this type.")
        min_idx = df_filtered.ontime_percentage.idxmin()
        return int(df_filtered.loc[min_idx, 'slot']), df_filtered.loc[min_idx, 'ontime_percentage']

    dlu_ontime_worst_slot, dlu_ontime_worst_percentage = _worst_ontime(df_ru_ontime, 'dlu')
    dlc_ontime_worst_slot, dlc_ontime_worst_percentage = _worst_ontime(df_ru_ontime, 'dlc')
    ulc_ontime_worst_slot, ulc_ontime_worst_percentage = _worst_ontime(df_ru_ontime, 'ulc')
    ul_ontime_worst_slot, ul_ontime_worst_percentage = _worst_ontime(df_du_ontime, 'ulu')

    dlu_max_idx = perfcsv_dict['dlu_max_idx']
    dlc_max_idx = perfcsv_dict['dlc_max_idx']
    ulc_max_idx = perfcsv_dict['ulc_max_idx']

    ulc_bfw_headroom = perf_csv_df['ulc_bfw_headroom'].min()
    ulc_nonbfw_headroom = perf_csv_df['ulc_nonbfw_headroom'].min()
    ulc_headroom = min(ulc_bfw_headroom, ulc_nonbfw_headroom)
    
    dlu_headroom = perf_csv_df['dlu_headroom'].min()
    
    dlc_bfw_headroom = perf_csv_df['dlc_bfw_headroom'].min()
    dlc_nonbfw_headroom = perf_csv_df['dlc_nonbfw_headroom'].min()
    dlc_headroom = min(dlc_bfw_headroom, dlc_nonbfw_headroom)
    ul_pusch_headroom_s4 = perf_csv_df[perf_csv_df.slot%10==4]['pusch_headroom'].min()
    ul_pusch_eh_headroom_s4 = perf_csv_df[perf_csv_df.slot%10==4]['pusch_eh_headroom'].min()
    ul_pucch_headroom_s4 = perf_csv_df[perf_csv_df.slot%10==4]['pucch_headroom'].min()
    ul_pusch_headroom_s5 = perf_csv_df[perf_csv_df.slot%10==5]['pusch_headroom'].min()
    ul_pusch_eh_headroom_s5 = perf_csv_df[perf_csv_df.slot%10==5]['pusch_eh_headroom'].min()
    ul_pucch_headroom_s5 = perf_csv_df[perf_csv_df.slot%10==5]['pucch_headroom'].min()
    ul_prach_headroom = perf_csv_df['prach_headroom'].min()

    headroom_info = {
        'ulc_headroom_worst_slot': int(perf_csv_df.iloc[ulc_max_idx].slot),
        'ulc_headroom': ulc_headroom,
        'dlc_headroom_worst_slot': int(perf_csv_df.iloc[dlc_max_idx].slot),
        'dlc_headroom': dlc_headroom,
        'dlu_headroom_worst_slot': int(perf_csv_df.iloc[dlu_max_idx].slot),
        'dlu_headroom': dlu_headroom,
        'dlu_ontime_worst_slot': dlu_ontime_worst_slot,
        'dlu_ontime': dlu_ontime_worst_percentage,
        'dlc_ontime_worst_slot': dlc_ontime_worst_slot,
        'dlc_ontime': dlc_ontime_worst_percentage,
        'ulc_ontime_worst_slot': ulc_ontime_worst_slot,
        'ulc_ontime': ulc_ontime_worst_percentage,
        'ul_ontime_worst_slot': ul_ontime_worst_slot,
        'ul_ontime': ul_ontime_worst_percentage,
        'ul_pusch_headroom_s4': ul_pusch_headroom_s4,
        'ul_pusch_eh_headroom_s4': ul_pusch_eh_headroom_s4,
        'ul_pucch_headroom_s4': ul_pucch_headroom_s4,
        'ul_pusch_headroom_s5': ul_pusch_headroom_s5,
        'ul_pusch_eh_headroom_s5': ul_pusch_eh_headroom_s5,
        'ul_pucch_headroom_s5': ul_pucch_headroom_s5,
        'ul_prach_headroom': ul_prach_headroom
    }

    return headroom_info

def store(save_files_data):
    successful = []
    for (data, file_path) in save_files_data:
        file_path: str = file_path
        is_json = file_path.endswith('.json')
        is_csv  = file_path.endswith('.csv')
        if (is_json == is_csv):
            print('Unknown file extension', file=sys.stderr)
            sys.exit(1)

        print(f'Saving {file_path}... ', end='')
        try:  
            if (is_json):
                with open(file_path, 'w') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            else:
                _df: pd.DataFrame = data
                _df.to_csv(file_path, index=False)
            
            if not os.path.basename(file_path) == 'dashboard_data.json':
                # We don't want to delete dashboard_data.json in case any of the next
                # documents fail.
                successful.append(file_path)
            print('OK')
        except Exception as e:
            print('Error')
            print(f'Exception writing to file {file_path}: {e}', file=sys.stderr)
            for saved_file in successful:
                print(f'Deleting file {saved_file}... ', end='')
                try:
                    os.remove(saved_file)
                    print('OK')
                except:
                    print('Error')
                    print(f'Error deleting {saved_file}', file=sys.stderr) 

def phase_2(scenario_folder):
    try:
        gc_info, additional_info = read_jenkins_test_information(scenario_folder, 'phase_2')
    except Exception as e:
        print(f'Error reading basic jenkins test run information: {e}. Exiting...', file=sys.stderr)
        sys.exit(1)

    save_files_data = []
    ncu_report_filename = path.join(scenario_folder, 'report.ncu-rep')
    if not path.isfile(ncu_report_filename):
        print(f'{ncu_report_filename} does not exist.', file=sys.stderr)
        additional_info.append('report.ncu-rep was missing.')
    else:
        try:
            stage = 'Retrieving kernel metrics'
            kernel_metrics = get_kernel_metrics(ncu_report_filename)
            kernel_metrics = post_process_kernel_metrics(kernel_metrics)
            save_files_data.append((kernel_metrics, path.join(scenario_folder, 'kernel_metrics.csv')))
        except Exception as e:
            additional_info.append(f'Error at processing stage {stage}')
            print(f'Error: {e}', file=sys.stderr)   
    
    gc_info['info'] = ','.join(additional_info)
    json_content = append_to_json(
        (gc_info, 'test_run_information')
    )
    save_files_data.insert(0, (json_content, path.join(scenario_folder, 'dashboard_data.json')))
    store(save_files_data)

def phase_4(scenario_folder, ignore_duration, max_duration, num_proc):
    try:
        gc_info, additional_info = read_jenkins_test_information(scenario_folder, 'phase_4')
    except Exception as e:
        print(f'Error reading basic jenkins test run information: {e}. Exiting...', file=sys.stderr)
        sys.exit(1)

    headroom_info = {}
    save_files_data = []
    try:
        phy_log_filename = path.join(scenario_folder, 'cuphy', 'phy.log')
        if not path.isfile(phy_log_filename):
            print(f'{phy_log_filename} does not exist.', file=sys.stderr)
            additional_info.append('phy.log was missing.')
        ref_t0 = get_ref_t0(phy_log_filename)
        
        mac_log_filename = path.join(scenario_folder, 'mac', 'testmac.log')
        if not path.isfile(mac_log_filename):
            print(f'{mac_log_filename} does not exist.', file=sys.stderr)
            additional_info.append('testmac.log was missing.')

        ru_log_filename = path.join(scenario_folder, 'ru', 'ru.log')
        if not path.isfile(ru_log_filename):
            print(f'{ru_log_filename} does not exist.', file=sys.stderr)
            additional_info.append('ru.log was missing.')

        stage = 'Retrieve heardroom/latency information'            
        headroom_info = retrieve_headroom_information(scenario_folder, phy_log_filename, ru_log_filename)

        stage = 'Retrieve MAC throuhgput information'
        mac_throughput_parsenator = Parsenator(mac_log_filename, 
                                            [parse_testmac_observed_throughput_per_second, parse_testmac_expected_throughput],
                                            column_filter_list=[None, None],
                                            start_tir=ignore_duration,
                                            end_tir=max_duration,
                                            num_processes=num_proc,
                                            ref_t0=ref_t0)
        parsed_result = mac_throughput_parsenator.parse()
        mac_observed_throughput_per_second = parsed_result[0]
        mac_expected_throughput            = parsed_result[1]
        headroom_info['mac_exp_thr_dl']= mac_expected_throughput.throughput_dl.mean()
        headroom_info['mac_exp_thr_ul']= mac_expected_throughput.throughput_ul.mean()
        headroom_info['mac_obs_thr_dl']= mac_observed_throughput_per_second.throughput_dl.mean()
        headroom_info['mac_obs_thr_ul']= mac_observed_throughput_per_second.throughput_ul.mean()
        save_files_data.append((mac_observed_throughput_per_second, path.join(scenario_folder, 'cuphy', 'perf_results', 'binary', 'mac_thr_obs.csv')))

        stage = 'Retrieve RU throughput information'
        ru_throuhgput_parsenator = Parsenator(ru_log_filename, 
                                            [parse_ru_observed_throughput_per_second],
                                            column_filter_list=[None],
                                            start_tir=ignore_duration,
                                            end_tir=max_duration,
                                            num_processes=num_proc,
                                            ref_t0=ref_t0)
        parsed_result = ru_throuhgput_parsenator.parse()
        ru_observed_throughput_per_second = parsed_result[0]
        headroom_info['ru_obs_thr_dl']= ru_observed_throughput_per_second.throughput_dl.mean()
        headroom_info['ru_obs_thr_ul']= ru_observed_throughput_per_second.throughput_ul.mean()    
        save_files_data.append((ru_observed_throughput_per_second, path.join(scenario_folder, 'cuphy', 'perf_results', 'binary', 'ru_thr_obs.csv')))
    
        stage = 'Retrieve SM allocation information'
        sm_provisioning               = get_sm_provisioning_from_scenario_folder(scenario_folder)
        save_files_data.append((sm_provisioning, path.join(scenario_folder, 'cuphy', 'perf_results', 'binary', 'allocation_sm.csv')))

        stage = 'Retrieve CPU allocation information'
        cpu_phy_consumption, resource_consumption = read_phy_cpu_consumption(scenario_folder)
        save_files_data.append((resource_consumption, path.join(scenario_folder, 'cuphy', 'perf_results', 'binary', 'allocation_cpu.csv')))
        save_files_data.append((cpu_phy_consumption, path.join(scenario_folder, 'cuphy', 'perf_results', 'binary', 'utilization_cpu.csv')))
    except Exception as e:
        additional_info.append(f'Error at processing stage {stage}')
        print(f'Error: {e}', file=sys.stderr)

    gc_info['info'] = ','.join(additional_info)
    json_content = append_to_json(
        (gc_info, 'test_run_information'),
        (headroom_info, 'headroom'),
    )
    save_files_data.insert(0, (json_content, path.join(scenario_folder, 'cuphy', 'perf_results', 'dashboard_data.json')))
    store(save_files_data)

if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0
    default_num_proc=8

    parser = argparse.ArgumentParser(
        description="Aggregate data from different data sources (log/CICD postprocessing) with the aim to feed this data to NVDF."
    )
    
    parser.add_argument('input_folder', type=str, help='Scenario folder.')
    parser.add_argument('--test_phase', type=str, default='phase_4', choices=['phase_2', 'phase_4'], help='Test phase class.')

    args = parser.parse_args()
    print(args)

    test_phase = args.test_phase
    scenario_folder = args.input_folder
    scenario_folder = os.path.abspath(scenario_folder)
    valid = validate_log_folder(scenario_folder)
    if not valid: exit
    scenario_folder = remove_trailing_separator(scenario_folder)

    if (test_phase == 'phase_4'):
        phase_4(scenario_folder, default_ignore_duration, default_max_duration, default_num_proc)
    elif (test_phase == 'phase_2'):
        from aerial_postproc.ncu_report_utils import get_kernel_metrics, post_process_kernel_metrics
        phase_2(scenario_folder)



        
