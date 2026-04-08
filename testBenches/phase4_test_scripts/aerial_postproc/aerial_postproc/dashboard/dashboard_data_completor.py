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


from aerial_postproc.logparse import get_num_dlc_tasks
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_integer_dtype, is_float_dtype
import numpy as np
import sys
import re
import warnings
from typing import Dict, List
from datetime import datetime
import math

warnings.filterwarnings('error')

def check_column_types(df):
    results = {}
    for col in df.columns:
        col_dtype = df[col].dtype
        if (is_numeric_dtype(col_dtype)):
            results[col] = col_dtype
            continue
        # Drop null values for analysis
        non_null_data = df[col].dropna()
        
        # Check if the column should be numeric
        numeric_column = pd.to_numeric(non_null_data, errors='coerce')
        num_errors = numeric_column.isna().sum()
        
        if num_errors == 0:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1e6).astype(numeric_column.dtype)
            results[col] = numeric_column.dtype
        else:
            results[col] = col_dtype
    
    return results

pattern = r'\[(.*?)\]'
def column_type_to_prefixes(dict_col_type):
    results = {}
    for col, col_type in dict_col_type.items():
        if is_string_dtype(col_type):
            prefix = 's'
        elif is_integer_dtype(col_type):
            prefix = 'l'
        elif is_float_dtype(col_type):
            prefix = 'd'
        else:
            raise ValueError(f'Column {col} has unsupported type {col_type}')
        match = re.search(pattern, col)
        replace_col = col.replace('.','_').replace(' ', '_')
        if match:
            replace_col = replace_col.replace('[', '_').replace(']', '_')
        replace_col = replace_col.replace('__', '_')
        results[col] = f'{prefix}_{replace_col}'
    return results

def remove_nan_from_json(data):
    """
    Recursively removes or replaces NaN values from a JSON-like dictionary or list.
    
    Args:
        data (dict or list): JSON-like data (dictionary or list)
    
    Returns:
        Cleaned data without NaN values
    """
    if isinstance(data, dict):
        return {k: remove_nan_from_json(v) for k, v in data.items() if not isinstance(v, float) or not math.isnan(v)}
    elif isinstance(data, list):
        return [remove_nan_from_json(item) for item in data if not isinstance(item, float) or not math.isnan(item)]
    else:
        return data

class DashboardDataCompletor:
    def __init__(self, 
                 extracted_dashboard_data, 
                 perf_metrics_io, 
                 latency_metrics_io,
                 csv_files,
                 phase) -> None:
        self.extracted_dashboard_data = extracted_dashboard_data
        self.perf_metrics_io = perf_metrics_io
        self.latency_metrics_io = latency_metrics_io
        self.csv_files = csv_files
        self.phase = phase
        self.cicd_index_info: Dict = None
        self.documents_to_send: List[Dict] = []

    def create_document(self, type: str, document: Dict, indeces):
        document = remove_nan_from_json(document)
        document['s_type'] = type
        indeces.insert(0, type)
        self.documents_to_send.append((document, indeces))

    def create_cicd_index_information(self):
        cicd_index = dict()
        cicd_index['s_phase'] = self.phase
        cicd_index['s_pipeline'] = self.extracted_dashboard_data['test_run_information']['jenkins_pipeline']
        cicd_index['l_cicd_id'] = self.extracted_dashboard_data['test_run_information']['jenkins_id']
        cicd_index['s_test_name'] = self.extracted_dashboard_data['test_run_information']['test_name']
        self.cicd_index_info = cicd_index

    def create_cicd_configuration(self):
        cicd_details = dict()
        test_run_information = self.extracted_dashboard_data['test_run_information']
        cicd_details['ts_cicd_datetime'] = int(datetime.strptime(test_run_information['jenkins_run_datetime'], '%Y/%m/%d %H:%M:%S').timestamp())
        cicd_details['s_result'] = test_run_information['result']
        cicd_details['s_gpu'] = test_run_information['gpu_platform']
        cicd_details['s_info'] = test_run_information['info']
        if self.phase == 'phase_4':
            cicd_details['s_mr_id'] = test_run_information['merge_request']
            cicd_details['l_cell_capacity'] = test_run_information['cell_count']
            cicd_details['l_bfp'] = test_run_information['bfp']
            cicd_details['s_pattern'] = test_run_information['pattern']
            cicd_details['b_eh'] = test_run_information['eh']
            cicd_details['b_gc'] = test_run_information['gc']
            cicd_details['b_dp'] = test_run_information['dual_port']
            cicd_details['l_test_duration'] = test_run_information['test_duration']
            cicd_details['b_power_enabled'] = test_run_information['power_enabled']
        else:
            cicd_details['s_channel'] = test_run_information['channel']
            cicd_details['s_tv'] = test_run_information['tv']
        self.create_document('configuration_details', cicd_details, [])

    def create_summary_data(self):
        if (self.phase == 'phase_2'):
            return
        perf_metrics_data = dict()
        headroom_info = self.extracted_dashboard_data['headroom']

        for key, value in headroom_info.items():
            prefix = ''
            if type(value) == float:
                prefix = 'd'
            elif type(value) == int:
                prefix = 'l'
            elif type(value) == str:
                prefix = 's'
            elif type(value) == bool:
                prefix = 'b'
            perf_metrics_data[f'{prefix}_{key}'] = value
        self.create_document('performance_metrics', perf_metrics_data, []) 

        if (self.latency_metrics_io is not None):
            latency_data = dict()   
            df_du_rx_timing = self.latency_metrics_io.data['df_du_rx_timing']
            df_ru_rx_timing = self.latency_metrics_io.data['df_ru_rx_timing']

            # Similar to cicd_summarize_latency.py
            for task, short_task in zip(['UL U Plane', 'DL U Plane', 'DL C Plane', 'UL C Plane'], ['ulu', 'dlu', 'dlc', 'ulc']):
                if task == 'UL U Plane':
                    current_data = df_du_rx_timing
                else:
                    current_data = df_ru_rx_timing

                # group by symbol first
                gb_symbol = current_data.groupby(['t0_timestamp', 'symbol'])
                current_data = current_data.assign(
                    start_min_symbol=gb_symbol['start_offset'].transform(min),
                    end_max_symbol  =gb_symbol['end_offset'].transform(max)
                )
                current_data['duration_symbol'] = current_data['end_max_symbol'] - current_data['start_min_symbol']
                gb_symbol = current_data.groupby(['slot', 'symbol'])
                stat_df = gb_symbol.quantile(.99, numeric_only=True).reset_index()

                latency_data[f'd_{short_task}_med_xlat'] = stat_df.end_max_symbol.median()
                latency_data[f'd_{short_task}_med_xdur'] = stat_df.duration_symbol.median()
                
                worst_idx = stat_df.end_max_symbol.idxmax()
                latency_data[f'l_{short_task}_worst_xlat_slot']   = int(stat_df.iloc[worst_idx]['slot'])
                latency_data[f'l_{short_task}_worst_xlat_symbol'] = int(stat_df.iloc[worst_idx]['symbol'])
                latency_data[f'd_{short_task}_worst_xlat']        = stat_df.iloc[worst_idx]['end_max_symbol']
                
                worst_idx = stat_df.duration_symbol.idxmax()
                latency_data[f'l_{short_task}_worst_xdur_slot']   = int(stat_df.iloc[worst_idx]['slot'])
                latency_data[f'l_{short_task}_worst_xdur_symbol'] = int(stat_df.iloc[worst_idx]['symbol'])
                latency_data[f'd_{short_task}_worst_xdur']        = stat_df.iloc[worst_idx]['duration_symbol']        
            self.create_document('latency_metrics', latency_data, [])
    
    def create_csv_obj_data(self):
        for df_path, s_type, include_row_index, index_columns in self.csv_files:
            try:
                df = pd.read_csv(df_path, na_values=['None'])
                if include_row_index:
                    df['row_index'] = np.arange(len(df), dtype=np.int32)
                df.replace(to_replace=r'(?i)none', value=np.nan, regex=True, inplace=True)
                column_types = check_column_types(df)
                for index_column in index_columns:
                    if index_column not in df.columns.tolist():
                        raise Exception(f'Index column {index_column} not one of columns of {df_path}')
                column_w_prefix = column_type_to_prefixes(column_types)
                for _, row in df.iterrows():
                    row_object = dict()
                    index_values = []
                    for col, value in row.to_dict().items():
                        # Handle pandas StringDtype which numpy can't interpret
                        if is_string_dtype(column_types[col]):
                            row_object[column_w_prefix[col]] = str(value) if pd.notna(value) else None
                        else:
                            row_object[column_w_prefix[col]] = np.array(value).astype(column_types[col]).item()
                        if col in index_columns:
                            index_values.append(value)
                    self.create_document(s_type, row_object, index_values)
            except Exception as e:
                print(f'Exception: {e}')

    def get_whisker_plots_metavalues(self, df_values, cols=None):
        all_columns = set(df_values.columns)
        if not cols:
            cols = []
        rest_columns = list(all_columns - set(cols))
        if len(rest_columns) != 1:
            print(f'::ERROR:: rest columns should be 1: {all_columns}, {rest_columns}')
        rest_column = rest_columns[0]
        if len(cols) > 0:            
            combinations_dict = dict()
            unique_combinations = df_values.loc[:, cols].drop_duplicates()
            for _, row in unique_combinations.iterrows():
                combination_field = tuple([row[col] for col in cols])
                mask = True
                for key, value in {col: row[col] for col in cols}.items():
                    mask &= (df_values[key] == value)

                subset_df = df_values.loc[mask, rest_column]
                arr_values_sorted = np.sort(subset_df)    
                combinations_dict[combination_field] = arr_values_sorted
        else:
            combinations_dict = dict()
            combinations_dict['all'] = df_values

        arr_whisker_metavalues = []
        for selector, arr_values_sorted in combinations_dict.items():
            Q1 = np.percentile(arr_values_sorted, 25)
            Q2 = np.median(arr_values_sorted)
            Q3 = np.percentile(arr_values_sorted, 75)
            IQR = Q3 - Q1
            lower_whisker  = Q1 - 1.5 * IQR
            higher_whisker = Q3 + 1.5 * IQR
            outliers = [x for x in arr_values_sorted if x < lower_whisker or x > higher_whisker]
            if len(outliers) > 100:
                outliers = np.random.choice(outliers, size=100, replace=False)
            data = {
                'd_Q1': Q1,
                'd_Q2': Q2,
                'd_Q3': Q3,
                'd_lower_whisker': lower_whisker,
                'd_higher_whisker': higher_whisker,
                'd_outliers': outliers
            }
            if type(selector) == tuple:
                field_selector = {f'l_{col}': selector[i] for i, col in enumerate(cols)}
            else:            
                field_selector = selector
            whisker_metavalues = {
                **field_selector,
                **data
            }
            arr_whisker_metavalues.append(whisker_metavalues)
        return arr_whisker_metavalues
    
    def get_ccdf_plots_metavalues(self, arr_values, **kwargs):
        xstart = kwargs.get('xstart', -1500)
        xstop  = kwargs.get('xstop', 0)
        xdelta = kwargs.get('xdelta', 1)

        nbins = int(round((xstop - xstart) / xdelta)) + 1
        bin_edges = [xstart + xdelta*(ii-0.5) for ii in range(nbins+1)]
        bins = [xstart + xdelta*ii for ii in range(nbins)]

        try:
            hh, _ = np.histogram(arr_values, bin_edges)
            cdf_ = np.cumsum(hh) # unnormalized
            cdf  = cdf_ / np.max(cdf_) # normalized CDF
            ccdf = 1 - cdf
        except RuntimeWarning:
            print('Gotchaa')
        
        data_ccdf = {
            'l_x_start' : int(xstart),
            'l_x_stop'  : int(xstop),
            'l_x_delta' : int(xdelta),
            'd_y': ccdf
        }
        return data_ccdf

    
    def add_dashboard_data_points(self, datum, title, groupcols=['slot'], **kwargs):
        all_columns = set(datum.columns)
        if len(all_columns) != len(groupcols) + 1:
            print(f'::ERROR:: Invalid set of columns: {all_columns}')
            sys.exit(1)
        if not 'slot' in all_columns:
            print(f'::ERROR:: Invalid slot column not in datum: {all_columns}')
            sys.exit(1)

        print(f"::INFO:: Generating Whisker/CCDF data for {title}")
        value_column = list(all_columns - set(groupcols))[0]
        
        arr_data_whisker = self.get_whisker_plots_metavalues(datum, cols=groupcols)
        obj_data_whisker = {
            's_title': title,
            'obj_whisker': arr_data_whisker
        }
        self.create_document('data_whisker', obj_data_whisker, [title])
        
        do_ccdf = kwargs.pop('ccdf', True)
        if do_ccdf:
            ccdf_data = self.get_ccdf_plots_metavalues(datum[value_column], **kwargs)
            obj_data_ccdf = {
                's_title': title,
                **ccdf_data
            }
            self.create_document('data_ccdf', obj_data_ccdf, [title])
                


    def create_bboxplot_from_df_mask_group(self, df, mask, group_cols, column, title, **kwargs):
        try:
            df = df.loc[mask, [*group_cols, column]]
            if len(df) == 0:
                return
            kwargs['xstart'] = np.floor(df[column].min() - 1)
            kwargs['xstop']  = np.ceil(df[column].max() + 1)
            self.add_dashboard_data_points(df, title, group_cols, **kwargs)
        except Exception as e:
            print(f'Exception: {e}')
    
    def create_bboxplot_from_df_mask(self, df, mask, column, title, **kwargs):        
        self.create_bboxplot_from_df_mask_group(df, mask, ['slot'], column, title, **kwargs)
    
    def create_boxplot_ccdf_data(self):
        if (self.phase == 'phase_2'):
            return
        df_ti = self.perf_metrics_io.data['df_ti']
        num_dlc_tasks = get_num_dlc_tasks(df_ti[(df_ti.tir - df_ti.tir.min()) < .5])
        df_ti.set_index(['task', 'subtask'], inplace=True)
        index_task = 0
        index_subtask = 1
        df_ti.sort_index(inplace=True)

        df_compression = self.perf_metrics_io.data['df_compression']
        df_l2 = self.perf_metrics_io.data['df_l2']
        df_gpu = self.perf_metrics_io.data['df_gpu']
        df_tick = self.perf_metrics_io.data['df_tick']
        df_testmac = self.perf_metrics_io.data['df_testmac']

        enable_subtask_breakdown = False

        if len(df_ti) > 0:
            self.create_bboxplot_from_df_mask(df_ti, ('UL Task UL AGGR 3', 'Wait Order'), 'end_deadline', 'Order Kernel Completion Times')
            self.create_bboxplot_from_df_mask(df_ti, ('UL Task UL AGGR 3', 'Callback PUSCH'), 'end_deadline', 'PUSCH Completion Times')
            self.create_bboxplot_from_df_mask(df_ti, ('UL Task UL AGGR 3', 'Started PUSCH'), 'start_deadline', 'PUSCH Started Times')
            self.create_bboxplot_from_df_mask(df_ti, ('UL Task UL AGGR 3', 'Callback PUCCH'), 'end_deadline', 'PUCCH Completion Times')
            self.create_bboxplot_from_df_mask(df_ti, ('UL Task UL AGGR 3', 'Started PUCCH'), 'start_deadline', 'PUCCH Started Times')
            self.create_bboxplot_from_df_mask(df_ti, ('UL Task UL AGGR 3', 'Callback PRACH'), 'end_deadline', 'PRACH Completion Times')
            self.create_bboxplot_from_df_mask(df_ti, ('UL Task UL AGGR 3', 'Started PRACH'), 'start_deadline', 'PRACH Started Times')                  
            
            dlc_tasks = ['DL Task C-Plane %i'%(ii+1) for ii in range(num_dlc_tasks)]
            dlc_subtask = 'Full Task'
            mask = (df_ti.index.get_level_values(index_task).isin(dlc_tasks)) & (df_ti.index.get_level_values(index_subtask)==dlc_subtask)
            dlc_info = df_ti.loc[mask, ['t0_timestamp','slot', 'start_deadline', 'end_deadline']].groupby(['t0_timestamp', 'slot']).aggregate({'start_deadline': 'min', 'end_deadline': 'max'}).reset_index()
            self.create_bboxplot_from_df_mask(dlc_info, slice(None), 'end_deadline','DLC Completion Times')
            self.create_bboxplot_from_df_mask(dlc_info, slice(None), 'start_deadline','DLC Started Times')      

            if (enable_subtask_breakdown):
                df_ti_full_task_dict  = dict()
                df_ti_all_subtasks_dict = dict()
                tasks = df_ti.index.get_level_values(index_task).unique()
                mask_subtask = df_ti.index.get_level_values(index_subtask) == 'Full Task'
                for tt in tasks:
                    mask_task =  df_ti.index.get_level_values(index_task) == tt
                    mask = (mask_task) & (mask_subtask)
                    df_ti_full_task_dict[tt] = df_ti.loc[mask,['slot','start_deadline','end_deadline','duration']]
                    
                    subtask_data = df_ti.loc[mask_task, ['sequence','slot','start_deadline','end_deadline','duration']]
                    subtask_data.loc[:, 'unique_label'] = subtask_data.sequence.astype(str) + (": " + subtask_data.index.get_level_values(index_subtask))                
                    df_ti_all_subtasks_dict[tt] = subtask_data
                
                cpu_tasks = list(df_ti_full_task_dict.keys())
                cpu_tasks.sort()
                for task in cpu_tasks:
                    self.create_bboxplot_from_df_mask_group(df_ti_full_task_dict[task], slice(None), ['slot'], 'start_deadline', f'CPU Task {task} Start Time')
                    self.create_bboxplot_from_df_mask_group(df_ti_full_task_dict[task], slice(None), ['slot'], 'end_deadline',   f'CPU Task {task} Complete Time', generator='minmax')
                    self.create_bboxplot_from_df_mask_group(df_ti_full_task_dict[task], slice(None), ['slot'], 'duration',       f'CPU Task {task} Total duration', generator='minmax')
                    self.create_bboxplot_from_df_mask_group(df_ti_all_subtasks_dict[task], slice(None), ['slot', 'unique_label'], 'duration', f'Subtask Duration Breakdown {task}', ccdf=False)
                    self.create_bboxplot_from_df_mask_group(df_ti_all_subtasks_dict[task], slice(None), ['slot', 'unique_label'], 'end_deadline', f'Subtask End Deadline Breakdown {task}', ccdf=False)
                    ccdf_subtask_list = ['Signal Slot Channel End','Wait Slot Channel End']
                    df_ti_subtask = df_ti_all_subtasks_dict[task]
                    for ccdf_subtask in ccdf_subtask_list:
                        mask = df_ti_subtask.index.get_level_values(index_subtask) == ccdf_subtask
                        self.create_bboxplot_from_df_mask(df_ti_subtask, mask, 'end_deadline', f'Subtask {task} Signal {ccdf_subtask} Complete Time', generator='minmax')

            # Add other generic UL CPU-based timelines
            cpu_tuples = [
                ("Early UCI Completion","UL Task AGGR3 Early UCI IND", "Signal Completion", "start_deadline"),
                ("task_work_function_ul_aggr_3_early_uci_ind Task Start Time","UL Task AGGR3 Early UCI IND", "Full Task", "start_deadline"),
                ("task_work_function_ul_aggr_3_early_uci_ind Task End Time","UL Task AGGR3 Early UCI IND", "Full Task", "end_deadline"),
                ("task_work_function_ul_aggr_3_early_uci_ind Task Duration","UL Task AGGR3 Early UCI IND", "Full Task", "duration"),
                ("FH Callback Start Time","DL Task FH Callback", "Full Task", "start_deadline"),
                ("FH Callback End Time","DL Task FH Callback", "Full Task", "end_deadline"),
                ("FH Callback Duration","DL Task FH Callback", "Full Task", "duration"),
                ("UPlane Prepare CPU Work Start Time","DL Task GPU Comms", "UPlane Prepare", "start_deadline"),
                ("UPlane Prepare CPU Work End Time","DL Task GPU Comms", "UPlane Prepare", "end_deadline"),
                ("UPlane Prepare CPU Work Duration","DL Task GPU Comms", "UPlane Prepare", "duration"),
                ("Compression Kernel Completion Time", "Debug Task", "Compression Wait", "end_deadline"),
                ("Compression Kernel Start Time", "Debug Task", "Compression Start Wait", "end_deadline"),
                ("Trigger Start Time", "Debug Task", "Trigger synchronize", "start_deadline"),
                ("Trigger End Time", "Debug Task", "Trigger synchronize", "end_deadline"),
                ("Trigger Duration", "Debug Task", "Trigger synchronize", "duration")
            ]
            titles = [aa[0] for aa in cpu_tuples]
            tasks = [aa[1] for aa in cpu_tuples]
            subtasks = [aa[2] for aa in cpu_tuples]
            fields = [aa[3] for aa in cpu_tuples]
            for title, task, subtask, field in zip(titles, tasks, subtasks, fields):
                self.create_bboxplot_from_df_mask(df_ti, (task, subtask), field, title)
            
        if len(df_compression)  > 0:
            mask = df_compression.cell==df_compression.cell.min()
            df_compression_first_cell = df_compression.loc[mask, :]
            titles = ["Channel To Compression Gap","Prepare Memset Duration","Prepare Kernel Duration","Prepare Total Duration","Compression Duration"]
            fields = ['channel_to_compression_gap','prepare_memsets','prepare_kernel','prepare_total','compression_kernel']
            for title, field in zip(titles,fields):
                self.create_bboxplot_from_df_mask(df_compression_first_cell, df_compression_first_cell.event==field, 'duration', title)

        if len(df_gpu) > 0:
            df_gpu_durations_dict = dict()
            tasks = ['PUSCH Aggr', 'PUCCH Aggr', 'PRACH Aggr', 'COMPRESSION DL', 'Aggr CSI-RS', 'PDSCH Aggr', 'Aggr PDCCH DL', 'Aggr PBCH', 'ORDER']
            tasks = [aa for aa in tasks if aa in list(set(df_gpu.task))]
            for tt in tasks:
                field_list = ['slot','cpu_setup_duration','cpu_run_duration','gpu_setup1_duration','gpu_setup2_duration','gpu_run_duration','gpu_total_duration']
                if(tt in ['PUSCH Aggr']):
                    field_list.extend(['gpu_run_eh_duration','gpu_run_noneh_duration','gpu_run_copy_duration','gpu_run_gap_duration'])
                if(tt in ['PDSCH Aggr']):
                    #PDSCH has PDSCH H2D Copy duration field (only if it exists in dataframe)
                    if 'pdsch_h2d_copy_duration' in df_gpu.columns:
                        field_list.extend(['pdsch_h2d_copy_duration'])
                df_gpu_durations_dict[tt] = df_gpu.loc[df_gpu.task==tt, field_list]
            gpu_tasks = list(df_gpu_durations_dict.keys())
            gpu_tasks.sort()
            for gpu_task in gpu_tasks:
                df_gpu_duration_task = df_gpu_durations_dict[gpu_task]
                self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'gpu_total_duration',  f'GPU Execution {gpu_task} Total Duration (Setup+Run)')
                self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'gpu_setup1_duration', f'GPU Execution {gpu_task} Total Duration (Setup1)')
                self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'gpu_setup2_duration', f'GPU Execution {gpu_task} Total Duration (Setup2)')
                self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'gpu_run_duration',    f'GPU Execution {gpu_task} Total Duration (Run)')
                self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'cpu_setup_duration',  f'CPU Setup {gpu_task} Total Duration')
                self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'cpu_run_duration',    f'CPU Run {gpu_task} Total Duration')                

                if gpu_task in ['PUSCH Aggr']:
                    self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'gpu_run_eh_duration',    f'GPU EH Run {gpu_task} Total Duration (Run)')
                    self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'gpu_run_noneh_duration', f'GPU non-EH Run {gpu_task} Total Duration (Run)')
                    self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'gpu_run_copy_duration',  f'GPU Copy {gpu_task} Total Duration (Run)')
                    self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'gpu_run_gap_duration',   f'GPU Gap {gpu_task} Total Duration (Run)')
                
                if gpu_task in ['PDSCH Aggr']:
                    if 'pdsch_h2d_copy_duration' in df_gpu_duration_task.columns:
                        self.create_bboxplot_from_df_mask(df_gpu_duration_task, slice(None), 'pdsch_h2d_copy_duration', f'PDSCH H2D Copy {gpu_task} Duration')

        if len(df_l2) > 0:
            self.create_bboxplot_from_df_mask(df_l2, slice(None), 'start_deadline', 'L2A Start Times')
            self.create_bboxplot_from_df_mask(df_l2, slice(None), 'end_deadline', 'L2A Completion Times')

        if len(df_testmac) > 0:
            df_testmac.loc[:, 'start_deadline'] = (df_testmac['fapi2_start_timestamp'] - df_testmac['t0_timestamp'])/1e3
            df_testmac.loc[:, 'end_deadline'] = (df_testmac['fapi2_stop_timestamp'] - df_testmac['t0_timestamp'])/1e3
            self.create_bboxplot_from_df_mask(df_testmac, slice(None), 'start_deadline', 'TestMAC First Send Times')
            self.create_bboxplot_from_df_mask(df_testmac, slice(None), 'end_deadline', 'TestMAC Last Send Times')

        if len(df_tick) > 0:
            self.create_bboxplot_from_df_mask(df_tick, slice(None), 'cpu_deadline', 'Tick CPU Time (usec)')

    def fill_data(self):
        self.create_cicd_index_information()
        self.create_cicd_configuration()

        for __func in [self.create_summary_data, self.create_boxplot_ccdf_data, self.create_csv_obj_data]:
            try:
                __func()
            except Exception as e:
                print(f'Exception: {e}')

        data = []
        id_prefix = '/'.join(map(str,list(self.cicd_index_info.values())))
        set_id = set()
        for document, indeces in self.documents_to_send:
            id_suffix = '/'.join(map(str, indeces))
            _id = '/'.join([id_prefix, id_suffix])
            data.append(
                {**self.cicd_index_info, **document, "_id": _id}
            )
            if _id in set_id:
                raise Exception(f'Document with _id: {_id} already exists: {document}')
            if len(bytes(_id.encode())) >= 512:
                raise Exception(f'Document has _id: {_id} with length > 512 bytes: {document}')
            set_id.add(_id)
        self.dashboard_data_complete = True
        self.data = data
        return (self.cicd_index_info['s_pipeline'], self.cicd_index_info['l_cicd_id'], self.cicd_index_info['s_test_name'])

    def get_data_as_dict(self):
        if not self.dashboard_data_complete:
            return None
        return self.data.to_dict()
    
