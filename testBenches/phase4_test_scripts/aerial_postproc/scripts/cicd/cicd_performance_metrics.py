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
from aerial_postproc.logparse import get_num_dlc_tasks, get_num_ulc_tasks
from aerial_postproc.logparse import CICD_SLOT_FORMAT, CICD_FLOAT_FORMAT, CICD_NONE_FORMAT
from bokeh.plotting import show
from bokeh.io import output_file
from bokeh.layouts import row, column
import pandas as pd
import argparse
import numpy as np
import csv
import time
import os

from functools import partial

#Variable to reference our 3 slot advance, in usec (used for testmac STT calculation)
L2_ADVANCE_TIME = 1500

#Deadlines for DLU/DLC/ULC
from aerial_postproc.logparse import getReceptionWindow, TrafficType, TestCaseType, SLOT_TIME

#Deadlines for ULU
ULU_DEADLINE_EH = 2000
ULU_DEADLINE_NON_EH = 4500

#ULBFW/DLBFW must be finished by DLC/ULC timeline of next slot
ULBFW_DEADLINE = L2_ADVANCE_TIME + getReceptionWindow(TrafficType.DTT_ULC_BFW,TestCaseType.GH_64TR)[0] + SLOT_TIME
DLBFW_DEADLINE = L2_ADVANCE_TIME + getReceptionWindow(TrafficType.DTT_DLC_BFW,TestCaseType.GH_64TR)[0] + SLOT_TIME

#SRS timeline must be within T0+5500
SRS_DEADLINE = 5500

eh_slots = [ii for ii in range(80) if ii%10 == 4]
eh_slot_deadlines = [ULU_DEADLINE_EH if ii in eh_slots else ULU_DEADLINE_NON_EH for ii in range(80)]
non_eh_slot_deadlines = [ULU_DEADLINE_NON_EH for ii in range(80)]
srs_deadlines = [SRS_DEADLINE for ii in range(80)]
ulbfw_deadlines = [ULBFW_DEADLINE for ii in range(80)]
dlbfw_deadlines = [DLBFW_DEADLINE for ii in range(80)]

def init_output_df(metric_list,slot_list):
    dict_list = []
    for ii in slot_list:
        current_dict = {}
        current_dict['slot'] = ii
        for metric in metric_list:
            current_dict[metric] = np.nan
        
        dict_list.append(current_dict)

    return pd.DataFrame(dict_list)



def add_ti_field(output_df,output_field,df_ti,task,subtask,input_field,qq,offset=0):
    #Create data from summarizing 99% for each slot
    full_df = df_ti[(df_ti.task==task) & (df_ti.subtask==subtask)][['t0_timestamp','slot',input_field]]
    slot_df = full_df.groupby('slot')[input_field].quantile(qq).reset_index()

    #Add results to output df
    for slot in slot_df.slot:
        current_val = slot_df[slot_df.slot == slot].iloc[0][input_field]
        output_df.loc[output_df.slot==slot, output_field] = current_val + offset



def add_headroom_field(output_df,input_field,output_field,slot_deadlines):
    for ii in range(80):
        current_deadline = slot_deadlines[ii]
        output_df.loc[output_df.slot==ii, output_field] = current_deadline - output_df[output_df.slot == ii][input_field] 


def add_tick_fields(output_df, df_tick, qq):
    #Add tick information
    df_tick['tick_to_l2_start'] = (df_tick['cpu_timestamp'] - df_tick['tick_timestamp'])/1000.
    tick_to_l2_start = df_tick.groupby('slot').tick_to_l2_start.quantile(qq).reset_index()
    for slot in tick_to_l2_start.slot:
        current_val = tick_to_l2_start[tick_to_l2_start.slot == slot].iloc[0].tick_to_l2_start
        output_df.loc[output_df.slot==slot, 'tick_to_l2_start'] = current_val


def add_testmac_fields(output_df, df_testmac, qq):
    # Helper function to process testmac timestamps
    def process_testmac_metric(df, col1, col2, output_col, quantile, offset=0):
        df['temp'] = (df[col1] - df[col2])/1000. + offset
        temp_gb = df.groupby('slot').temp
        temp_df = temp_gb.quantile(quantile).reset_index()
        
        for slot in temp_df.slot:
            current_val = temp_df[temp_df.slot == slot].iloc[0].temp
            output_df.loc[output_df.slot==slot, output_col] = current_val

    # Process testmac start timestamps
    process_testmac_metric(df_testmac, 'fapi2_start_timestamp', 't0_timestamp', 'testmac_start_min', 1.0-qq, L2_ADVANCE_TIME)
    process_testmac_metric(df_testmac, 'fapi2_start_timestamp', 't0_timestamp', 'testmac_start_max', qq, L2_ADVANCE_TIME)

    # Process testmac end timestamps  
    process_testmac_metric(df_testmac, 'fapi2_stop_timestamp', 't0_timestamp', 'testmac_end_min', 1.0-qq, L2_ADVANCE_TIME)
    process_testmac_metric(df_testmac, 'fapi2_stop_timestamp', 't0_timestamp', 'testmac_end_max', qq, L2_ADVANCE_TIME)

    # Calculate testmac duration using same helper function
    process_testmac_metric(df_testmac, 'fapi2_stop_timestamp', 'fapi2_start_timestamp', 'testmac_duration', qq, 0)



def add_l2a_fields(output_df,df_l2a,qq):
    #Form 99th percentile statistics on L2A start/stop
    df_l2a['l2a_start'] = df_l2a.start_deadline + L2_ADVANCE_TIME
    df_l2a['l2a_end'] = df_l2a.end_deadline + L2_ADVANCE_TIME
    df_l2a['l2a_duration'] = df_l2a.end_deadline - df_l2a.start_deadline

    l2a_start = df_l2a.groupby('slot').l2a_start.quantile(qq).reset_index()
    l2a_end = df_l2a.groupby('slot').l2a_end.quantile(qq).reset_index()
    l2a_duration = df_l2a.groupby('slot').l2a_duration.quantile(qq).reset_index()

    #Add in l2a information
    for slot in l2a_start.slot:
        current_val = l2a_start[l2a_start.slot == slot].iloc[0].l2a_start
        output_df.loc[output_df.slot==slot, 'l2a_start_max'] = current_val
    for slot in l2a_end.slot:
        current_val = l2a_end[l2a_end.slot == slot].iloc[0].l2a_end
        output_df.loc[output_df.slot==slot, 'l2a_end_max'] = current_val
    for slot in l2a_duration.slot:
        current_val = l2a_duration[l2a_duration.slot == slot].iloc[0].l2a_duration
        output_df.loc[output_df.slot==slot, 'l2a_duration'] = current_val



def add_dlu_fields(output_df, df_ti, df_l2a, qq, legacy_enable, tc_type):
    #Create total dl l1 processing dataframe
    temp_df1 = df_l2a[['t0_timestamp','slot','start_deadline']].copy()
    temp_df2 = df_ti[(df_ti.task=='Debug Task') & (df_ti.subtask=='Trigger synchronize')][['t0_timestamp','end_deadline']].copy()
    found_debug_thread_messages = len(temp_df2) > 0
    if(not found_debug_thread_messages and not legacy_enable):
        print("Error: No debug thread messages found.  Must enable debug thread in cuphy controller yaml (debug_worker: 17)")
        return
    df_dlu_processing = pd.merge(temp_df1,temp_df2,on=["t0_timestamp"],validate="one_to_one")
    df_dlu_processing['tick_deadline'] = df_dlu_processing['end_deadline'] + L2_ADVANCE_TIME

    #Form final dataframes with 1% CCDF statistics
    tick_to_dlu_completion = df_dlu_processing.groupby('slot').tick_deadline.quantile(qq).reset_index()

    tick_to_deadline = L2_ADVANCE_TIME + getReceptionWindow(TrafficType.DTT_DLU, TestCaseType(tc_type))[0]
    for slot in tick_to_dlu_completion.slot:
        current_val = tick_to_dlu_completion[tick_to_dlu_completion.slot == slot].iloc[0].tick_deadline
        output_df.loc[output_df.slot==slot, 'tick_to_dlu_completion'] = current_val
        output_df.loc[output_df.slot==slot, 'dlu_headroom'] = tick_to_deadline - current_val



def add_dlc_fields(output_df, df_ti, df_l2a, qq, tc_type):

    #Determine the number of DLC Tasks
    ref_t0 = df_ti.t0_timestamp.min()
    df_ti_short = df_ti[(df_ti.t0_timestamp - ref_t0)/1e9 < 0.5]
    num_dlc_tasks = get_num_dlc_tasks(df_ti_short)
    
    dlc_tasks = ['DL Task C-Plane %i'%(ii+1) for ii in range(num_dlc_tasks)]
    
    # Check if new subtasks exist
    has_new_subtasks = False
    if len(dlc_tasks) > 0:
        # Check for new subtask pattern "bfw_prepare cell_X sym_0" or "nonbfw_prepare cell_X sym_0"
        dlc_subtasks = df_ti[df_ti.task.isin(dlc_tasks)].subtask.unique()
        bfw_subtasks = [st for st in dlc_subtasks if st.startswith('bfw_prepare cell_')]
        nonbfw_subtasks = [st for st in dlc_subtasks if st.startswith('nonbfw_prepare cell_')]
        has_new_subtasks = len(bfw_subtasks) > 0 or len(nonbfw_subtasks) > 0
    
    if has_new_subtasks:
        # Process BFW subtasks
        if len(bfw_subtasks) > 0:
            temp_df1 = df_l2a[['t0_timestamp','slot','start_deadline']].copy()
            temp_df2 = df_ti[df_ti.task.isin(dlc_tasks) & df_ti.subtask.isin(bfw_subtasks)][['t0_timestamp','task','subtask','end_deadline']].copy()
            temp_df2 = temp_df2.groupby('t0_timestamp').end_deadline.max().reset_index()
            df_bfw_processing = pd.merge(temp_df1,temp_df2,on=["t0_timestamp"],how='left')
            df_bfw_processing['tick_deadline'] = df_bfw_processing['end_deadline'] + L2_ADVANCE_TIME
            
            tick_to_dlc_bfw_completion = df_bfw_processing[df_bfw_processing.end_deadline.notnull()].groupby('slot').tick_deadline.quantile(qq).reset_index()
            
            for slot in tick_to_dlc_bfw_completion.slot:
                current_val = tick_to_dlc_bfw_completion[tick_to_dlc_bfw_completion.slot == slot].iloc[0].tick_deadline
                output_df.loc[output_df.slot==slot, 'tick_to_dlc_bfw_completion'] = current_val
        
        # Process non-BFW subtasks
        if len(nonbfw_subtasks) > 0:
            temp_df1 = df_l2a[['t0_timestamp','slot','start_deadline']].copy()
            temp_df2 = df_ti[df_ti.task.isin(dlc_tasks) & df_ti.subtask.isin(nonbfw_subtasks)][['t0_timestamp','task','subtask','end_deadline']].copy()
            temp_df2 = temp_df2.groupby('t0_timestamp').end_deadline.max().reset_index()
            df_nonbfw_processing = pd.merge(temp_df1,temp_df2,on=["t0_timestamp"],how='left')
            df_nonbfw_processing['tick_deadline'] = df_nonbfw_processing['end_deadline'] + L2_ADVANCE_TIME
            
            tick_to_dlc_nonbfw_completion = df_nonbfw_processing[df_nonbfw_processing.end_deadline.notnull()].groupby('slot').tick_deadline.quantile(qq).reset_index()
            
            for slot in tick_to_dlc_nonbfw_completion.slot:
                current_val = tick_to_dlc_nonbfw_completion[tick_to_dlc_nonbfw_completion.slot == slot].iloc[0].tick_deadline
                output_df.loc[output_df.slot==slot, 'tick_to_dlc_nonbfw_completion'] = current_val
                
    else:
        # Fall back to old method using 'CPlane Prepare'
        dlc_subtask = 'CPlane Prepare'
        temp_df1 = df_l2a[['t0_timestamp','slot','start_deadline']].copy()
        temp_df2 = df_ti[df_ti.task.isin(dlc_tasks) & (df_ti.subtask==dlc_subtask)][['t0_timestamp','task','end_deadline']].copy()
        temp_df2 = temp_df2.groupby('t0_timestamp').max()
        df_dlc_processing = pd.merge(temp_df1,temp_df2,on=["t0_timestamp"],validate="one_to_one")
        df_dlc_processing['tick_deadline'] = df_dlc_processing['end_deadline'] + L2_ADVANCE_TIME

        tick_to_dlc_completion = df_dlc_processing.groupby('slot').tick_deadline.quantile(qq).reset_index()

        # Populate both BFW and non-BFW with the same value
        for slot in tick_to_dlc_completion.slot:
            current_val = tick_to_dlc_completion[tick_to_dlc_completion.slot == slot].iloc[0].tick_deadline
            output_df.loc[output_df.slot==slot, 'tick_to_dlc_nonbfw_completion'] = current_val
            output_df.loc[output_df.slot==slot, 'tick_to_dlc_bfw_completion'] = current_val

    # Calculate headroom
    tick_to_bfw_deadline = L2_ADVANCE_TIME + getReceptionWindow(TrafficType.DTT_DLC_BFW, TestCaseType(tc_type))[0]
    tick_to_nonbfw_deadline = L2_ADVANCE_TIME + getReceptionWindow(TrafficType.DTT_DLC_NONBFW, TestCaseType(tc_type))[0]
    for slot in range(80):
        # Calculate separate headroom for BFW and non-BFW
        bfw_val = output_df[output_df.slot==slot]['tick_to_dlc_bfw_completion'].iloc[0] if len(output_df[output_df.slot==slot]) > 0 else np.nan
        nonbfw_val = output_df[output_df.slot==slot]['tick_to_dlc_nonbfw_completion'].iloc[0] if len(output_df[output_df.slot==slot]) > 0 else np.nan
        
        if not np.isnan(bfw_val):
            output_df.loc[output_df.slot==slot, 'dlc_bfw_headroom'] = tick_to_bfw_deadline - bfw_val
        if not np.isnan(nonbfw_val):
            output_df.loc[output_df.slot==slot, 'dlc_nonbfw_headroom'] = tick_to_nonbfw_deadline - nonbfw_val



def add_ulc_fields(output_df, df_ti, df_l2a, qq, tc_type=True):

    #Determine the number of ULC tasks
    ref_t0 = df_ti.t0_timestamp.min()
    df_ti_short = df_ti[(df_ti.t0_timestamp - ref_t0)/1e9 < 0.5]
    num_ulc_tasks = get_num_ulc_tasks(df_ti_short)

    ulc_tasks = ['UL Task CPlane %i'%(ii+1) for ii in range(num_ulc_tasks)]
    
    # Check if new subtasks exist
    has_new_subtasks = False
    if len(ulc_tasks) > 0:
        # Check for new subtask pattern "bfw_prepare cell_X sym_0" or "nonbfw_prepare cell_X sym_0"
        ulc_subtasks = df_ti[df_ti.task.isin(ulc_tasks)].subtask.unique()
        bfw_subtasks = [st for st in ulc_subtasks if st.startswith('bfw_prepare cell_')]
        nonbfw_subtasks = [st for st in ulc_subtasks if st.startswith('nonbfw_prepare cell_')]
        has_new_subtasks = len(bfw_subtasks) > 0 or len(nonbfw_subtasks) > 0
    
    if has_new_subtasks:
        # Process BFW subtasks
        if len(bfw_subtasks) > 0:
            temp_df1 = df_l2a[['t0_timestamp','slot','start_deadline']].copy()
            temp_df2 = df_ti[df_ti.task.isin(ulc_tasks) & df_ti.subtask.isin(bfw_subtasks)][['t0_timestamp','task','subtask','end_deadline']].copy()
            temp_df2 = temp_df2.groupby('t0_timestamp').end_deadline.max().reset_index()
            df_bfw_processing = pd.merge(temp_df1,temp_df2,on=["t0_timestamp"],how='left')
            df_bfw_processing['tick_deadline'] = df_bfw_processing['end_deadline'] + L2_ADVANCE_TIME
            
            tick_to_ulc_bfw_completion = df_bfw_processing[df_bfw_processing.end_deadline.notnull()].groupby('slot').tick_deadline.quantile(qq).reset_index()
            
            for slot in tick_to_ulc_bfw_completion.slot:
                current_val = tick_to_ulc_bfw_completion[tick_to_ulc_bfw_completion.slot == slot].iloc[0].tick_deadline
                output_df.loc[output_df.slot==slot, 'tick_to_ulc_bfw_completion'] = current_val
        
        # Process non-BFW subtasks
        if len(nonbfw_subtasks) > 0:
            temp_df1 = df_l2a[['t0_timestamp','slot','start_deadline']].copy()
            temp_df2 = df_ti[df_ti.task.isin(ulc_tasks) & df_ti.subtask.isin(nonbfw_subtasks)][['t0_timestamp','task','subtask','end_deadline']].copy()
            temp_df2 = temp_df2.groupby('t0_timestamp').end_deadline.max().reset_index()
            df_nonbfw_processing = pd.merge(temp_df1,temp_df2,on=["t0_timestamp"],how='left')
            df_nonbfw_processing['tick_deadline'] = df_nonbfw_processing['end_deadline'] + L2_ADVANCE_TIME
            
            tick_to_ulc_nonbfw_completion = df_nonbfw_processing[df_nonbfw_processing.end_deadline.notnull()].groupby('slot').tick_deadline.quantile(qq).reset_index()
            
            for slot in tick_to_ulc_nonbfw_completion.slot:
                current_val = tick_to_ulc_nonbfw_completion[tick_to_ulc_nonbfw_completion.slot == slot].iloc[0].tick_deadline
                output_df.loc[output_df.slot==slot, 'tick_to_ulc_nonbfw_completion'] = current_val
                
    else:
        #Find which subtask is being used ('Send C Plane' or 'CPlane Prepare')
        #This allows backwards compatibility with Rel 25-1 logs
        ulc_subtask = 'Send C Plane'
        if(len(df_ti[df_ti.task.isin(ulc_tasks) & (df_ti.subtask=='CPlane Prepare')]) > 0):
            ulc_subtask = 'CPlane Prepare'

        temp_df1 = df_l2a[['t0_timestamp','slot','start_deadline']].copy()
        temp_df2 = df_ti[(df_ti.task.isin(ulc_tasks)) & (df_ti.subtask==ulc_subtask)][['t0_timestamp','end_deadline']].copy()
        temp_df2 = temp_df2.groupby('t0_timestamp').max()
        df_ulc_processing = pd.merge(temp_df1,temp_df2,on=["t0_timestamp"],validate="one_to_one")
        df_ulc_processing['tick_deadline'] = df_ulc_processing['end_deadline'] + L2_ADVANCE_TIME

        tick_to_ulc_completion = df_ulc_processing.groupby('slot').tick_deadline.quantile(qq).reset_index()

        # Populate both BFW and non-BFW with the same value
        for slot in tick_to_ulc_completion.slot:
            current_val = tick_to_ulc_completion[tick_to_ulc_completion.slot == slot].iloc[0].tick_deadline
            output_df.loc[output_df.slot==slot, 'tick_to_ulc_nonbfw_completion'] = current_val
            output_df.loc[output_df.slot==slot, 'tick_to_ulc_bfw_completion'] = current_val

    # Calculate headroom
    tick_to_bfw_deadline = L2_ADVANCE_TIME + getReceptionWindow(TrafficType.DTT_ULC_BFW, TestCaseType(tc_type))[0]
    tick_to_nonbfw_deadline = L2_ADVANCE_TIME + getReceptionWindow(TrafficType.DTT_ULC_NONBFW, TestCaseType(tc_type))[0]
    for slot in range(80):
        # Calculate separate headroom for BFW and non-BFW
        bfw_val = output_df[output_df.slot==slot]['tick_to_ulc_bfw_completion'].iloc[0] if len(output_df[output_df.slot==slot]) > 0 else np.nan
        nonbfw_val = output_df[output_df.slot==slot]['tick_to_ulc_nonbfw_completion'].iloc[0] if len(output_df[output_df.slot==slot]) > 0 else np.nan
        
        if not np.isnan(bfw_val):
            output_df.loc[output_df.slot==slot, 'ulc_bfw_headroom'] = tick_to_bfw_deadline - bfw_val
        if not np.isnan(nonbfw_val):
            output_df.loc[output_df.slot==slot, 'ulc_nonbfw_headroom'] = tick_to_nonbfw_deadline - nonbfw_val



def add_gpu_fields(output_df, df_gpu, gpu_channel_name_map, qq, ignore_list):
    gpu_durations_dict = {}
    for channel,metric in gpu_channel_name_map.items():
        temp = df_gpu[df_gpu.task==channel]
        if(metric.endswith("total_duration")):
            gpu_durations_dict[metric] = temp.groupby('slot').gpu_total_duration.quantile(qq).reset_index()
            gpu_durations_dict[metric] = gpu_durations_dict[metric].rename(columns={"gpu_total_duration":"duration"})
        else:
            gpu_durations_dict[metric] = temp.groupby('slot').gpu_run_duration.quantile(qq).reset_index()
            gpu_durations_dict[metric] = gpu_durations_dict[metric].rename(columns={"gpu_run_duration":"duration"})

    #Add in GPU stats per channel
    for channel,metric in gpu_channel_name_map.items():
        current_durations = gpu_durations_dict[metric]
        if(len(current_durations) == 0 and metric not in ignore_list):
            print("WARNING: Unable to find gpu durations for channel %s.  Please double check your MAP_DL/MAP_UL nvlog settings"%channel)
        for slot in current_durations.slot:
            current_val = current_durations[current_durations.slot == slot].iloc[0].duration
            output_df.loc[output_df.slot==slot, metric] = current_val



def add_ontime_fields(output_df, df_ontime, ontime_type, metric_name):
    df_filtered = df_ontime[df_ontime.type == ontime_type]
    if len(df_filtered) > 0:
        for slot in df_filtered.slot.unique():
            min_idx = df_filtered[df_filtered.slot == slot].ontime_percentage.idxmin()
            current_val = df_filtered.loc[min_idx, 'ontime_percentage']
            output_df.loc[output_df.slot == slot, metric_name] = current_val



def main(args):

    tt1 = time.time()

    print("Running cicd_performance_metrics.py...")
    print("input data: %s"%(args.input_data))
    print("performance_metrics_outfile: %s"%args.performance_metrics_outfile)
    print("max_outfile: %s"%args.max_outfile)
    print("max_duration: %s"%args.max_duration)
    print("ignore_duration: %s"%args.ignore_duration)
    print("num_proc: %s"%args.num_proc)

    # Determine reference time and parsing bounds
    start_tir = args.ignore_duration
    end_tir = args.max_duration

    from aerial_postproc.parsenator_formats import PerfMetricsIO
    pmio = PerfMetricsIO()
    parse_success = pmio.parse(args.input_data,start_tir,end_tir,num_proc=args.num_proc)
    if(parse_success):
        df_tick = pmio.data['df_tick']
        df_ti = pmio.data['df_ti']
        df_l2a = pmio.data['df_l2']
        df_gpu = pmio.data['df_gpu']
        df_testmac = pmio.data['df_testmac']
        df_du_ontime = pmio.data['df_du_ontime']
        df_ru_ontime = pmio.data['df_ru_ontime']
    else:
        print("Error:: Invalid input_data %s"%args.input_data)

    tt2 = time.time()

    min_tir = df_ti.tir.min()
    max_tir = df_ti.tir.max()
    print("Processing df_ti TIR %.1f to %.1f"%(df_ti.tir.min(),df_ti.tir.max()))
    print("Processing df_l2a TIR %.1f to %.1f"%(df_l2a.tir.min(),df_l2a.tir.max()))
    print("Processing df_gpu TIR %.1f to %.1f"%(df_gpu.tir.min(),df_gpu.tir.max()))

    #Check to make sure proper messages are enabled
    found_tick_messages = len(df_tick) > 0
    found_l2a_messages = len(df_l2a) > 0
    found_ti_messages = len(df_ti) > 0
    found_gpu_messages = len(df_gpu) > 0
    if(not found_tick_messages and not args.ignore_ticks):
        print("Error: No tick messages found.  Must enable tick messages for this script to function (nvlog TAG=L2A.TICK_TIMES)")
    if(not found_l2a_messages):
        print("Error: No L2A messages found.  Must enable L2A messages for this script to function (nvlog TAG=L2A.PROCESSING_TIMES)")
    if(not found_ti_messages):
        print("Error: No TI messages found.  Must enable cpu task tracing and TI messages (nvlog TAG=DRV.FUNC_DL)")
    if(not found_gpu_messages):
        print("Error: No GPU messages found.  Must enable DL gpu duration messages (nvlog tag=MAP_DL)")
    if((not found_tick_messages and not args.ignore_ticks) or not found_l2a_messages or not found_ti_messages or not found_gpu_messages):
        return
    

    #Full list of non-GPU duration metrics
    metric_list = ["tick_to_l2_start","testmac_start_min","testmac_start_max","testmac_end_min", "testmac_end_max",
                   "l2a_start_max", "l2a_end_max",
                   "tick_to_fhcb_completion",
                   "tick_to_dlu_completion", "tick_to_dlc_bfw_completion", "tick_to_dlc_nonbfw_completion", "tick_to_ulc_bfw_completion", "tick_to_ulc_nonbfw_completion",
                   "tick_to_ulbfw_completion", "tick_to_dlbfw_completion",
                   "t0_to_order_kernel_completion",
                   "t0_to_pucch_completion", "t0_to_pusch_eh_completion", "t0_to_pusch_completion", "t0_to_prach_completion","t0_to_srs_completion",
                   "dlu_headroom", "dlc_bfw_headroom", "dlc_nonbfw_headroom", "ulc_bfw_headroom", "ulc_nonbfw_headroom",
                   "ulbfw_headroom",
                   "dlbfw_headroom",
                   "pucch_headroom", "pusch_eh_headroom", "pusch_headroom", "prach_headroom", "srs_headroom",
                   "dlu_ontime_percentage", "dlc_ontime_percentage", "ulc_ontime_percentage",
                   "ulutx_prach_ontime_percentage", "ulutx_pucch_ontime_percentage", "ulutx_pusch_ontime_percentage", "ulutx_srs_ontime_percentage",
                   "ulu_ontime_percentage", "srs_ontime_percentage",
                   "testmac_duration", "l2a_duration"
                   ]
    #Also keep track of mapping of MAP_UL/MAP_DL messages
    gpu_channel_name_map = {
        'PDSCH Aggr':'pdsch_gpu_total_duration',
        'Aggr PDCCH DL':'pdcch_gpu_total_duration',
        'Aggr CSI-RS':'csirs_gpu_total_duration',
        'Aggr PBCH':'pbch_gpu_total_duration',
        'COMPRESSION DL':'compression_gpu_total_duration',                    
        'PUSCH Aggr':'pusch_gpu_run_duration',
        'PUCCH Aggr':'pucch_gpu_run_duration',
        'PRACH Aggr':'prach_gpu_run_duration',
        'SRS Aggr': 'srs_gpu_run_duration',
        'Aggr DL_BFW': 'dlbfw_gpu_run_duration',
        'UL_BFW Aggr': 'ulbfw_gpu_run_duration',
    }
    for channel,metric in gpu_channel_name_map.items():
        metric_list.append(metric)


    #MMIMO is expected to populated these fields, and these fields are not expected in 4TR
    mmimo_only_metric_list = [
        "t0_to_srs_completion", "srs_headroom", "srs_ontime_percentage",
        "tick_to_dlbfw_completion", "dlbfw_headroom",
        "srs_gpu_run_duration",
        "dlbfw_gpu_run_duration",
        "ulutx_srs_ontime_percentage",
    ]

    st_list = list(df_ti[df_ti.task=='UL Task UL AGGR 3'].subtask.unique())
    PUSCH_ONLY = ('Callback PUSCH' in st_list) and ('Callback PUCCH' not in st_list) and ('Callback PRACH' not in st_list) 
    pusch_only_ignore_list = [
        "t0_to_pucch_completion", "t0_to_prach_completion",
        "pucch_headroom", "prach_headroom",
        "pucch_gpu_run_duration", "prach_gpu_run_duration"
    ]
    #These fields are currently not populated in any test case (plan is eventually they will be populated)
    ignore_list = [
        "tick_to_ulbfw_completion",
        "ulbfw_headroom",
        "ulbfw_gpu_run_duration",
    ]

    ul_channel_metric_map = {
        'PUCCH': ['t0_to_pucch_completion', 'pucch_headroom', 'pucch_gpu_run_duration', 'ulutx_pucch_ontime_percentage'],
        'PRACH': ['t0_to_prach_completion', 'prach_headroom', 'prach_gpu_run_duration', 'ulutx_prach_ontime_percentage'],
        'SRS': ['t0_to_srs_completion', 'srs_headroom', 'srs_ontime_percentage', 'srs_gpu_run_duration', 'ulutx_srs_ontime_percentage'],
        'PUSCH': ['t0_to_pusch_completion', 'pusch_headroom', 'pusch_gpu_run_duration',
                  't0_to_pusch_eh_completion', 'pusch_eh_headroom', 'ulutx_pusch_ontime_percentage'],
    }
    ignored_ul_metrics = []
    ignored_ul_channels = set()
    for channel in args.ignore_ul_channels:
        channel_upper = channel.upper()
        if channel_upper in ul_channel_metric_map:
            ignored_ul_metrics.extend(ul_channel_metric_map[channel_upper])
            ignored_ul_channels.add(channel_upper)
            print("Note: Ignoring UL channel %s metrics: %s" % (channel_upper, ', '.join(ul_channel_metric_map[channel_upper])))
        else:
            print("WARNING: Unknown UL channel '%s'. Valid channels: %s" % (channel, ', '.join(ul_channel_metric_map.keys())))

    non_srs_ul_channels = {'PUCCH', 'PRACH', 'PUSCH'}
    if non_srs_ul_channels.issubset(ignored_ul_channels):
        ignored_ul_metrics.append('ulu_ontime_percentage')
        print("Note: Also ignoring ulu_ontime_percentage (all non-SRS UL channels are ignored)")

    #Start with an empty set of metrics (base this on L2A information)
    min_slot = df_l2a.slot.min()
    max_slot = df_l2a.slot.max()
    merged_df = init_output_df(metric_list,[ii for ii in range(min_slot,max_slot+1)])

    #Add tick information
    #Populates: tick_to_l2_start
    if not args.ignore_ticks:
        add_tick_fields(merged_df, df_tick, args.quantile)

    #Add testmac information
    #Populates: testmac_start_min, testmac_start_max, testmac_end_min, testmac_end_max, testmac_duration
    add_testmac_fields(merged_df, df_testmac, args.quantile)

    #Add l2a information
    #Populates: l2a_start_max, l2a_end_max, l2a_duration
    add_l2a_fields(merged_df, df_l2a, args.quantile)

    #Add Fronthaul Callback timeline
    add_ti_field(merged_df,'tick_to_fhcb_completion',df_ti,'DL Task FH Callback','Full Task','end_deadline',args.quantile, offset=L2_ADVANCE_TIME)

    #Add ULBFW/DLBFW fields
    if 'UL Task UL AGGR 3 ULBFW' in df_ti.task.unique():
        add_ti_field(merged_df,'tick_to_ulbfw_completion',df_ti,'UL Task UL AGGR 3 ULBFW','Callback ULBFW','end_deadline',args.quantile, offset=L2_ADVANCE_TIME)
    else:
        add_ti_field(merged_df,'tick_to_ulbfw_completion',df_ti,'UL Task UL AGGR 3','Callback ULBFW','end_deadline',args.quantile, offset=L2_ADVANCE_TIME)

    add_ti_field(merged_df,'tick_to_dlbfw_completion',df_ti,'DL Task BFW','Wait Run Completion','end_deadline',args.quantile, offset=L2_ADVANCE_TIME)

    #Add DLBFW/ULBFW headroom fields
    add_headroom_field(merged_df,'tick_to_ulbfw_completion','ulbfw_headroom',ulbfw_deadlines)
    add_headroom_field(merged_df,'tick_to_dlbfw_completion','dlbfw_headroom',dlbfw_deadlines)

    #Add DLU information
    #Populates: tick_to_dlu_completion, dlu_headroom
    add_dlu_fields(merged_df, df_ti, df_l2a, args.quantile, args.legacy_enable, int(args.mmimo_enable))

    #Add order kernel timeline
    add_ti_field(merged_df,'t0_to_order_kernel_completion',df_ti,'UL Task UL AGGR 3','Wait Order','end_deadline', args.quantile)

    #Add PUCCH timeline
    add_ti_field(merged_df,'t0_to_pucch_completion',df_ti,'UL Task UL AGGR 3','Callback PUCCH','end_deadline', args.quantile)
    add_headroom_field(merged_df,'t0_to_pucch_completion','pucch_headroom',eh_slot_deadlines)

    #Create PUSCH timeline
    add_ti_field(merged_df,'t0_to_pusch_completion',df_ti,'UL Task UL AGGR 3','Callback PUSCH','end_deadline', args.quantile)
    add_headroom_field(merged_df,'t0_to_pusch_completion','pusch_headroom',non_eh_slot_deadlines)

    #Create PRACH timeline as well
    add_ti_field(merged_df,'t0_to_prach_completion',df_ti,'UL Task UL AGGR 3','Callback PRACH','end_deadline', args.quantile)
    add_headroom_field(merged_df,'t0_to_prach_completion','prach_headroom',non_eh_slot_deadlines)

    #Create PUSCH Early HARQ timeline
    EH_ENABLED = len(df_ti[(df_ti.task=='UL Task AGGR3 Early UCI IND') & (df_ti.subtask=='Signal Completion')]) > 0
    if EH_ENABLED:
        add_ti_field(merged_df,'t0_to_pusch_eh_completion',df_ti,'UL Task AGGR3 Early UCI IND','Signal Completion','start_deadline', args.quantile)
        add_headroom_field(merged_df,'t0_to_pusch_eh_completion','pusch_eh_headroom',eh_slot_deadlines)

    #Create SRS timeline
    if 'UL Task UL AGGR 3 SRS' in df_ti.task.unique():
        add_ti_field(merged_df,'t0_to_srs_completion',df_ti,'UL Task UL AGGR 3 SRS','Callback SRS','end_deadline', args.quantile)
    else:
        add_ti_field(merged_df,'t0_to_srs_completion',df_ti,'UL Task UL AGGR 3','Callback SRS','end_deadline', args.quantile)
        
    add_headroom_field(merged_df,'t0_to_srs_completion','srs_headroom',srs_deadlines)

    #DLC Fields
    #Populates: tick_to_dlc_nonbfw_completion, tick_to_dlc_bfw_completion, dlc_bfw_headroom, dlc_nonbfw_headroom
    add_dlc_fields(merged_df, df_ti, df_l2a, args.quantile, int(args.mmimo_enable))

    #ULC Fields
    #Populates: tick_to_ulc_nonbfw_completion, tick_to_ulc_bfw_completion, ulc_bfw_headroom, ulc_nonbfw_headroom
    add_ulc_fields(merged_df, df_ti, df_l2a, args.quantile, int(args.mmimo_enable))

    #GPU Fields
    #Populates everything defined in gpu_channel_name_map
    gpu_ignore_list = ignore_list + ignored_ul_metrics
    if not args.mmimo_enable:
        gpu_ignore_list += mmimo_only_metric_list
    add_gpu_fields(merged_df, df_gpu, gpu_channel_name_map, args.quantile, gpu_ignore_list)

    du_ontime_map = {
        'ulu': 'ulu_ontime_percentage',
        'srs': 'srs_ontime_percentage',
    }
    for ontime_type, metric_name in du_ontime_map.items():
        add_ontime_fields(merged_df, df_du_ontime, ontime_type, metric_name)

    ru_ontime_map = {
        'dlu': 'dlu_ontime_percentage',
        'dlc': 'dlc_ontime_percentage',
        'ulc': 'ulc_ontime_percentage',
        'ulutx_prach': 'ulutx_prach_ontime_percentage',
        'ulutx_pucch': 'ulutx_pucch_ontime_percentage',
        'ulutx_pusch': 'ulutx_pusch_ontime_percentage',
        'ulutx_srs': 'ulutx_srs_ontime_percentage',
    }
    for ontime_type, metric_name in ru_ontime_map.items():
        add_ontime_fields(merged_df, df_ru_ontime, ontime_type, metric_name)

    #Print worst case slot information
    for metric in metric_list:
        current_df = merged_df[merged_df[metric].notnull()]
        sorted_merged_metrics = current_df.sort_values(metric)
        NUM_WORST_CASE_SLOTS = 5
        print("Worst case slot (%s):"%metric)
        for ii in range(min(len(current_df),NUM_WORST_CASE_SLOTS)):
            current_slot = sorted_merged_metrics.iloc[-1-ii].slot
            print_str = "  [%02i] slot: %02i"%(ii,current_slot)
            for mname2 in metric_list:
                mval = sorted_merged_metrics.iloc[-1-ii][mname2]
                if(mval is not None):
                    print_str += ", %s: %.3f"%(mname2,mval)
                else:
                    print_str += ", %s: %s"%(mname2,mval)

            print(print_str)

    #Validate that at each metric has at least one slot populated
    print("Running column validation...")
    all_valid = True
    
    for col in [aa for aa in merged_df.columns if aa not in ['slot']]:
        #Ignore tick_to_l2_start if ignore_ticks is enabled
        if args.ignore_ticks and col == 'tick_to_l2_start':
            print("Note: skipping check on %s (--ignore_ticks is enabled)"%col)
            continue
            
        #Ignore EH fields if this is a non EH run
        if not EH_ENABLED and col in ['t0_to_pusch_eh_completion','pusch_eh_headroom']:
            print("Note: skipping check on %s for non EH run"%col)
            continue

        #Ignore fields that are not currently populated (should be in next pattern iteration)
        if (col in ignore_list):
            print("Note: Skipping check on ignored metric %s (currently not implemented)"%col)
            continue

        #Ignore if this is a PUSCH only test case ignore PUCCH/PRACH fields
        if (PUSCH_ONLY and (col in pusch_only_ignore_list)):
            print("Note: Skipping check on ignored metric %s (PUSCH only)"%col)
            continue

        #For 4TR ignore MMIMO fields that have all nan values 
        if (not args.mmimo_enable and col in mmimo_only_metric_list and all(np.isnan(merged_df[col]))):
            print("Note: Skipping check on ignored metric %s (ignoring MMIMO field for 4TR test case)"%col)
            continue
        
        #Skip metrics for ignored UL channels (specified via --ignore_ul_channels)
        if col in ignored_ul_metrics:
            print("Note: Skipping check on %s (UL channel ignored via --ignore_ul_channels)"%col)
            continue
            
        #For BFW/non-BFW fields, it's okay if one of them is empty as long as at least one has values
        bfw_nonbfw_pairs = [
            ('tick_to_dlc_bfw_completion', 'tick_to_dlc_nonbfw_completion'),
            ('tick_to_ulc_bfw_completion', 'tick_to_ulc_nonbfw_completion')
        ]
        skip_bfw_check = False
        for bfw_field, nonbfw_field in bfw_nonbfw_pairs:
            if col == bfw_field or col == nonbfw_field:
                # Check if at least one of the pair has non-null values
                bfw_has_values = not all(np.isnan(merged_df[bfw_field]))
                nonbfw_has_values = not all(np.isnan(merged_df[nonbfw_field]))
                if bfw_has_values or nonbfw_has_values:
                    skip_bfw_check = True
                    if all(np.isnan(merged_df[col])):
                        print("Note: %s is empty but its pair has values (this is expected for slots without %s)"%(col, 'BFW' if 'bfw' in col else 'non-BFW'))
                    break
        
        if skip_bfw_check and all(np.isnan(merged_df[col])):
            continue

        if all(np.isnan(merged_df[col])):
            print("ERROR :: col %s has all None values"%col)
            all_valid = False
            
    if not all_valid:
        print("ERROR :: Not outputting any files because at least one column is broken")
        #Do not output any files - something is wrong
        return
    else:
        print("All columns have at least one slot populated.")
               
    #Convert 'slot' to 2-char string
    merged_df.slot = merged_df.slot.map(lambda xx: CICD_SLOT_FORMAT%xx)

    #Write out performance metrics file if enabled
    if(args.performance_metrics_outfile):
        print("Writing performance metrics csv to %s..."%args.performance_metrics_outfile)
        merged_df.to_csv(args.performance_metrics_outfile,float_format=CICD_FLOAT_FORMAT,index=False,na_rep=CICD_NONE_FORMAT)

    #Write out max file if enabled
    if(args.max_outfile):
        print("Writing max csv to %s..."%args.max_outfile)
        max_dict = {}
        for metric in metric_list:
            max_dict[metric] = merged_df[metric].max()
        temp_df = pd.DataFrame([max_dict])
        temp_df.to_csv(args.max_outfile,float_format="%.3f",index=False)

    tt3 = time.time()
    print("Processed %.1fsec of log data in %.1fsec. parsing/processing = %.1f/%.1f"%(max_tir-min_tir,tt3-tt1,tt2-tt1,tt3-tt2))


if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0
    default_num_proc=8
    default_quantile=0.99

    parser = argparse.ArgumentParser(
        description="Analyze PHY/RU log to determine worst case 1%% CCDF performance.  Optionally generate output files that CICD can use to manage merge gating",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Basic usage with a pre-parsed results folder
  %(prog)s /path/to/results/

  # Generate per-slot and max CSV outputs
  %(prog)s /path/to/results/ -p perf_metrics.csv -s max_metrics.csv

  # Parse raw log files (phy / testmac / ru)
  %(prog)s phy.log testmac.log ru.log -p perf_metrics.csv

  # Ignore first 10 seconds and cap at 60 seconds of log data
  %(prog)s /path/to/results/ -p perf_metrics.csv -i 10 -m 60

  # MMIMO (64TR) test case
  %(prog)s /path/to/results/ -p perf_metrics.csv -e

  # PUSCH-only test case: skip PUCCH and PRACH validation
  %(prog)s /path/to/results/ -p perf_metrics.csv -c PUCCH PRACH

  # Skip PUCCH, PRACH, and SRS validation
  %(prog)s /path/to/results/ -p perf_metrics.csv -c PUCCH PRACH SRS
"""
    )
    parser.add_argument(
        "input_data", type=str, nargs="+", help="Can be single folder containing pre-parsed result or three arguments: phy filename / testmac filename / ru filename (space separated)"
    )
    parser.add_argument(
        "-p", "--performance_metrics_outfile", help="Output csv file that contains 1%% CCDF performance metrics as function of slot"
    )
    parser.add_argument(
        "-s", "--max_outfile", help="Output csv file that can be used to output the maximum of each columns into a csv (worst case across all slots)"
    )
    parser.add_argument(
        "-m", "--max_duration", default=default_max_duration, type=float, help="Maximum run duration to process"
    )
    parser.add_argument(
        "-i", "--ignore_duration", default=default_ignore_duration, type=float, help="Number of seconds at beginning of file to ignore"
    )
    parser.add_argument(
        "-n", "--num_proc", default=default_num_proc, type=int, help="Number of parsing processes"
    )
    parser.add_argument(
        "-l", "--legacy_enable", action="store_true", help="Enable legacy processing (allows for processing Rel22-4 logs, not all fields available)"
    )
    parser.add_argument(
        "-q", "--quantile", default=default_quantile, type=float, help="Quantile to cut metrics (default=%f)"%default_quantile
    )
    parser.add_argument(
        "-e", "--mmimo_enable", action="store_true", help="Modifies timeline requirement to mmimo settings"
    )
    parser.add_argument(
        "-t", "--ignore_ticks", action="store_true", help="Ignore the requirement for tick messages. If set, df_tick does not need to be populated and tick_to_l2_start will be empty in the output"
    )
    parser.add_argument(
        "-c", "--ignore_ul_channels", nargs="+", default=[], help="List of UL channels to ignore in validation (e.g., PUCCH PRACH SRS PUSCH). Both timeline and GPU duration metrics for specified channels will be skipped."
    )
    args = parser.parse_args()

    main(args)
