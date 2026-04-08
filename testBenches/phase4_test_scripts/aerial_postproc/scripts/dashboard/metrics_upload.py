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
import sys
from os import path
from aerial_postproc.parsenator_formats import PerfMetricsIO, LatencySummaryIO
from aerial_postproc.dashboard.dashboard_data_completor import DashboardDataCompletor
from nvdataflow2.api import post, delete
import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(np.round(obj, 3))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def remove_trailing_separator(folder_path):
    if folder_path.endswith(path.sep):
        folder_path = folder_path[:-len(path.sep)]
    return folder_path

def validate_log_folder(scenario_folder):
    if not path.isdir(scenario_folder):
        print(f'ERROR :: {scenario_folder} log folder path {scenario_folder} is not a folder.')
        return False
    return True

def payload_size(json_object):
    return len(bytes(json.dumps(json_object, cls=NpEncoder).encode()))

def chunk_data(batch_data, chunk_to_enter, i):
    if payload_size(chunk_to_enter) > 99 * (1024 ** 2):
        num_documents = len(chunk_to_enter)
        
        chunk_1 = chunk_to_enter[:num_documents//2]
        next_pos = chunk_data(batch_data, chunk_1, i)

        chunk_2 = chunk_to_enter[num_documents//2:]
        next_pos = chunk_data(batch_data, chunk_2, next_pos)
    else:
        batch_data.insert(i, chunk_to_enter)
        next_pos = i + 1
    
    return next_pos

def rate_limit_nvdf_data(data):
    # Make sure that no individual JSON object has size greater than 16MB
    for json_object in data:
        if payload_size(json_object) >= 16 * (1024 ** 2):
            raise Exception(f"JSON Object {json_object} hits NVDF document limit size of 16MB.")

    batch_data = []
    chunk_data(batch_data, data, 0)

    return batch_data

def test_case_query(pipeline, cicd_id, test_name):
    query = {
        'bool': {
            'must': [
                {
                    'match': {
                        's_pipeline': pipeline
                    }
                },
                {
                    'match': {
                        'l_cicd_id': cicd_id
                    }
                }, 
                {
                    'match': {
                        's_test_name': test_name
                    }
                }
            ]
        }
    }
    return query

def delete_documents(nvdf_index, pipeline, cicd_id, test_name):
    body = {
        'query': test_case_query(pipeline, cicd_id, test_name)
    }
    delete(project=nvdf_index, body=body)

def main(phase, scenario_folder, ignore_duration, max_duration, num_proc, nvdf_index, upload_os):    
    if (phase == 'phase_4'):    
        dashboard_data_file = path.join(scenario_folder, 'cuphy', 'perf_results', 'dashboard_data.json')
    else:
        dashboard_data_file = path.join(scenario_folder, 'dashboard_data.json')
    if not path.isfile(dashboard_data_file):
        print(f'{dashboard_data_file} does not exist. Exiting...', file=sys.stderr)
        sys.exit(1)
    with open(dashboard_data_file) as f:
        extracted_dashboard_data = json.load(f)

    if (phase == 'phase_4'):
        csv_files = [
            (path.join(scenario_folder, 'cuphy', 'perf_results', 'power.csv'), 'csv_power', True, ['row_index']),
            (path.join(scenario_folder, 'cuphy', 'perf_results', 'power_summary.csv'), 'csv_power_summary', False, ['source']),
            (path.join(scenario_folder, 'cuphy', 'perf_results', 'perf.csv') , 'csv_perf', False, ['slot']),
            (path.join(scenario_folder, 'cuphy', 'perf_results', 'binary', 'allocation_sm.csv') , 'csv_alloc_sm', False, ['channel']),
            (path.join(scenario_folder, 'cuphy', 'perf_results', 'binary', 'allocation_cpu.csv') , 'csv_alloc_cpu', False, ['thread_category']),
            (path.join(scenario_folder, 'cuphy', 'perf_results', 'binary', 'utilization_cpu.csv') , 'csv_util_cpu', False, ['timestamp', 'thread_name']),
        ]
    else:
        csv_files = [
            (path.join(scenario_folder, 'kernel_metrics.csv'), 'csv_kernel_metrics', False, ['kernel']),
        ]

    existing_csv_files = []
    for csv_file_info in csv_files:
        _data_file = csv_file_info[0]
        if not path.isfile(_data_file):
            print(f'{_data_file} does not exist. Skipping...')
        else:
            existing_csv_files.append(csv_file_info)    


    if (phase == 'phase_4'):
        metrics_folder      = path.join(scenario_folder, 'cuphy', 'perf_results', 'binary')
        # Retrieve the PerfMetricsIO data.
        perf_metrics_io = PerfMetricsIO()
        try:        
            perf_metrics_io.parse([metrics_folder],
                                        ignore_duration,
                                        max_duration,
                                        slot_list=None,
                                        num_proc=num_proc)
        except Exception as e:
            print(f'PerfMetricsIO Exception {e}', file=sys.stderr)
            perf_metrics_io = None
    else:
        perf_metrics_io = None
        latency_metrics_io = None

    # Retrieve the PerfMetricsIO data.
    latency_metrics_io = None    
    opensearch_data = DashboardDataCompletor(extracted_dashboard_data, 
                                            perf_metrics_io,
                                            latency_metrics_io,
                                            existing_csv_files,
                                            phase)
    pipeline, cicd_id, test_name = opensearch_data.fill_data()
    data = opensearch_data.data

    if (phase == 'phase_4'):
        output_file = path.join(scenario_folder, 'cuphy', 'perf_results', 'opensearch_payload.json')
    else:
        output_file = path.join(scenario_folder, 'opensearch_payload.json')
    print(F'Saving {output_file}...', end='')
    try:
        with open(output_file, 'w') as f:            
            json.dump(data, f, ensure_ascii=False, indent=4, cls=NpEncoder)    
        print('OK')
    except Exception as e:
        print('Error')
        print(f'Error writing to file {output_file}: {e}', file=sys.stderr)
        sys.exit(1)

    print('Rate limit data...', end='')
    try:
        rate_limited_data = rate_limit_nvdf_data(data)
        print('OK')
    except Exception as e:
        print('Error')
        print(f'Error chunking data: {e}', file=sys.stderr)
        sys.exit(1)

    if upload_os:    
        print(f'Posting data to NVDF Index {nvdf_index}...', end='')
        try:
            for datum in rate_limited_data:
                json_encoded_data = json.dumps(datum, cls=NpEncoder)
                result = post(data=json_encoded_data, project=nvdf_index)
                if result >= 400:
                    raise Exception(f"Error code {result}")
            print('OK')
        except Exception as e:
            print('Error')
            print(f'Error posting to NVDF Index {nvdf_index}, Error: {e}', file=sys.stderr)
            import time
            try:
                time.sleep(10)
                delete_documents(nvdf_index, pipeline, cicd_id, test_name)
            except Exception as e:
                print(f'Error deleting documents for tuple({pipeline},{cicd_id},{test_name}) in index {nvdf_index}')
            sys.exit(1)

if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0
    default_num_proc=8

    parser = argparse.ArgumentParser(
        description="Aggregate data from different data sources (log/CICD postprocessing) with the aim to feed this data to NVDF."
    )
    
    parser.add_argument('input_folder', type=str, help='Test folder')
    parser.add_argument('--test_phase', type=str, default='phase_4', choices=['phase_2', 'phase_4'], help='Test phase class.')
    parser.add_argument('--index', type=str, help='NVDataFlow index', default='swgpu-aerial-perflab-cicd')
    parser.add_argument('--upload_opensearch', type=bool, help='Upload to Open Search', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    scenario_folder = args.input_folder
    test_phase      = args.test_phase
    nvdf_index      = args.index
    upload_os       = bool(args.upload_opensearch)
    print(args)

    valid = validate_log_folder(scenario_folder)
    if not valid: exit
    scenario_folder = remove_trailing_separator(scenario_folder)

    output_file = main(test_phase, scenario_folder, default_ignore_duration, default_max_duration, default_num_proc, nvdf_index, upload_os)
        
