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
from aerial_postproc.logparse import TrafficType, TestCaseType, getReceptionWindow, getTCType, get_num_dlc_tasks, get_num_ulc_tasks, get_num_gpu_comms_prepare_tasks
from aerial_postproc.logplot import whisker_comparison, ccdf_comparison, comparison_legend
from bokeh.plotting import show, save
from bokeh.io import output_file
from bokeh.layouts import row, column
from bokeh.models import Div
import argparse
import numpy as np
import os
import pandas as pd
import time
from aerial_postproc.parsenator_formats import parse_all_PerfMetricsIO, parse_labels

L2_ADVANCE_TIME = 1500

def has_bfw_nonbfw_subtasks(df_ti, task_list):
    """
    Check if the new BFW/non-BFW subtask format exists (bfw_prepare cell_* / nonbfw_prepare cell_*)
    
    Args:
        df_ti: DataFrame containing task information
        task_list: List of task names to check (e.g., DLC tasks, ULC tasks)
        
    Returns:
        tuple: (has_new_subtasks, bfw_subtasks, nonbfw_subtasks)
    """
    if len(task_list) == 0:
        return False, [], []
        
    # Check for new subtask pattern "bfw_prepare cell_X sym_0" or "nonbfw_prepare cell_X sym_0"
    subtasks = df_ti[df_ti.task.isin(task_list)].subtask.unique()
    bfw_subtasks = [st for st in subtasks if st.startswith('bfw_prepare cell_')]
    nonbfw_subtasks = [st for st in subtasks if st.startswith('nonbfw_prepare cell_')]
    has_new_subtasks = len(bfw_subtasks) > 0 or len(nonbfw_subtasks) > 0
    
    return has_new_subtasks, bfw_subtasks, nonbfw_subtasks

def main(args):

    time1 = time.time()

    df_ti_list,df_l2_list,df_gpu_list,df_compression_list,df_tick_list,df_testmac_list,slot_list = parse_all_PerfMetricsIO(args.input_data_list,
                                                                                                                           args.ignore_duration,
                                                                                                                           args.max_duration,
                                                                                                                           args.num_proc,
                                                                                                                           args.slot_selection)

    time2 = time.time()

    #Create plots
    fig = get_compare_logs_fig(args, df_ti_list,df_testmac_list,df_l2_list,df_gpu_list,df_compression_list,df_tick_list,slot_list=slot_list,labels=args.labels,
                               enable_ul_timeline=True,enable_dl_timeline=True,enable_l2a_timeline=True,enable_gpu_durations=True,
                               enable_subtask_breakdown=args.enable_subtask_breakdown,
                               enable_gpu_prep_breakdown=args.enable_gpu_prep_breakdown,
                               enable_ulc_dlc_breakouts=args.enable_ulc_dlc_breakouts)
    
    time3 = time.time()

    print("Processing complete.  Total duration: %.1f Parsing duration: %.1f  Processing duration: %.1f"%(time3-time1,time2-time1,time3-time2))

    #Output plots
    if(args.out_filename):
        out_filename = args.out_filename
    else:
        out_filename = "result.html"
    # Get base name without extension
    html_title = os.path.splitext(os.path.basename(out_filename))[0]
    output_file(filename=out_filename, title=html_title)
    print("Writing output to %s"%out_filename)
    if(args.out_filename):
        save(fig)
        time4 = time.time()
        print("Saving complete. Saving duration: %.1f"%(time4-time3))
    else:
        show(fig)

def get_compare_logs_fig(args, df_ti_list,df_testmac_list,df_l2_list,df_gpu_list,df_compression_list,df_tick_list,slot_list=None,labels=None,
                         enable_ul_timeline=True,enable_dl_timeline=True,enable_l2a_timeline=True,
                         enable_gpu_durations=True,enable_subtask_breakdown=False,enable_gpu_prep_breakdown=False,enable_ulc_dlc_breakouts=False):
    print("Running get_compare_logs_fig, lens: %i %i %i"%(len(df_ti_list),len(df_l2_list),len(df_gpu_list)))

    labels = parse_labels(labels,len(df_ti_list))

    NUM_DATASETS = len(df_ti_list)
    HAS_TICK_TIMES = all([len(aa)>0 for aa in df_tick_list])

    df_order_complete_list = []
    df_srs_order_complete_list = []
    df_pusch_complete_list = []
    df_pusch_started_list = []
    df_srs_complete_list = []
    df_srs_started_list = []
    df_ulbfw_complete_list = []
    df_ulbfw_started_list = []
    df_dlbfw_started_list = []
    df_dlbfw_complete_list = []
    df_pucch_complete_list = []
    df_pucch_started_list = []
    df_prach_complete_list = []
    df_prach_started_list = []
    df_trigger_complete_list = []
    df_compression_first_cell_list = []
    df_compression_complete_list = []
    df_compression_start_list = []
    df_prepare_submitted_list = []
    df_gpu_durations_dict = {} #Indexed based on channel
    df_l2a_start_list = []
    df_l2a_complete_list = []
    df_tick_complete_list = []
    df_ti_full_task_dict = {} #Indexed based on task
    df_ti_all_subtasks_dict = {}

    for ii in range(NUM_DATASETS):
        print("Processing dataset %i"%ii)
        df_ti = df_ti_list[ii]
        df_l2 = df_l2_list[ii]
        df_gpu = df_gpu_list[ii]
        df_compression = df_compression_list[ii]

        if(HAS_TICK_TIMES):
            df_tick = df_tick_list[ii]
            df_tick_complete_list.append(df_tick)

        if(slot_list):
            # Save on memory
            if(len(df_gpu) > 0):
                df_ti = df_ti[df_ti.slot.isin(slot_list)]
            if(len(df_gpu) > 0):
                df_gpu = df_gpu[df_gpu.slot.isin(slot_list)]
            if(len(df_l2) > 0):
                df_l2 = df_l2[df_l2.slot.isin(slot_list)]
            if(len(df_compression) > 0):
                df_compression = df_compression[df_compression.slot.isin(slot_list)]

        # Add several deadline times based on df_ti
        if(len(df_ti) > 0):
            df_order_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Wait Order')][['slot','end_deadline']].copy())

            df_pusch_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Callback PUSCH')][['slot','end_deadline']].copy())
            df_pusch_started_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Started PUSCH')][['slot','start_deadline']].rename(columns={'start_deadline':'end_deadline'}).copy())

            if 'UL Task UL AGGR 3 SRS' in df_ti.task.unique():
                df_srs_order_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3 SRS') & (df_ti.subtask=='Wait SRS Order')][['slot','end_deadline']].copy())
                df_srs_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3 SRS') & (df_ti.subtask=='Callback SRS')][['slot','end_deadline']].copy())
                df_srs_started_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3 SRS') & (df_ti.subtask=='Started SRS')][['slot','start_deadline']].rename(columns={'start_deadline':'end_deadline'}).copy())
            else:
                df_srs_order_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Wait SRS Order')][['slot','end_deadline']].copy())
                df_srs_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Callback SRS')][['slot','end_deadline']].copy())
                df_srs_started_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Started SRS')][['slot','start_deadline']].rename(columns={'start_deadline':'end_deadline'}).copy())

            if 'UL Task UL AGGR 3 ULBFW' in df_ti.task.unique():
                df_ulbfw_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3 ULBFW') & (df_ti.subtask=='Callback ULBFW')][['slot','end_deadline']].copy())
                df_ulbfw_started_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3 ULBFW') & (df_ti.subtask=='Started ULBFW')][['slot','start_deadline']].rename(columns={'start_deadline':'end_deadline'}).copy())
            else:
                df_ulbfw_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Callback ULBFW')][['slot','end_deadline']].copy())
                df_ulbfw_started_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Started ULBFW')][['slot','start_deadline']].rename(columns={'start_deadline':'end_deadline'}).copy())

            df_pucch_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Callback PUCCH')][['slot','end_deadline']].copy())
            df_pucch_started_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Started PUCCH')][['slot','start_deadline']].rename(columns={'start_deadline':'end_deadline'}).copy())
            df_prach_complete_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Callback PRACH')][['slot','end_deadline']].copy())
            df_prach_started_list.append(df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Started PRACH')][['slot','start_deadline']].rename(columns={'start_deadline':'end_deadline'}).copy())

            df_trigger_complete_list.append(df_ti[(df_ti.task=='Debug Task') & (df_ti.subtask=='Trigger synchronize')][['slot','end_deadline']].copy())
            df_compression_complete_list.append(df_ti[(df_ti.task=='Debug Task') & (df_ti.subtask=='Compression Wait')][['slot','end_deadline']].copy())
            df_compression_start_list.append(df_ti[(df_ti.task=='Debug Task') & (df_ti.subtask=='Compression Start Wait')][['slot','end_deadline']].copy())
            df_dlbfw_started_list.append(df_ti[(df_ti.task=='DL Task BFW') & (df_ti.subtask=='Wait Run Start')][['slot','end_deadline']].copy())
            df_dlbfw_complete_list.append(df_ti[(df_ti.task=='DL Task BFW') & (df_ti.subtask=='Wait Run Completion')][['slot','end_deadline']].copy())

        # Add GPU duration times
        if(len(df_gpu) > 0):
            tasks = ['PUSCH Aggr', 'PUCCH Aggr', 'PRACH Aggr', 'COMPRESSION DL', 'Aggr CSI-RS', 
                     'PDSCH Aggr', 'Aggr PDCCH DL', 'Aggr PBCH', 'ORDER',
                     'SRS Aggr', 'Aggr DL_BFW', 'UL_BFW Aggr']
            tasks = [aa for aa in tasks if aa in list(set(df_gpu.task))]
            for tt in tasks:
                field_list = ['slot','cpu_setup_duration','cpu_run_duration','gpu_setup1_duration','gpu_setup2_duration','gpu_run_duration','gpu_total_duration']
                if(tt in ['PUSCH Aggr']):
                    #PUSCH has some extra fields
                    field_list.extend(['gpu_run_eh_duration','gpu_run_noneh_duration','gpu_run_copy_duration','gpu_run_gap_duration'])
                if(tt in ['PDSCH Aggr']):
                    #PDSCH has PDSCH H2D Copy duration field (only if it exists in dataframe)
                    if 'pdsch_h2d_copy_duration' in df_gpu.columns:
                        field_list.extend(['pdsch_h2d_copy_duration'])

                data = df_gpu[df_gpu.task==tt][field_list].copy()

                if tt not in df_gpu_durations_dict.keys():
                    df_gpu_durations_dict[tt] = [data]
                else:
                    df_gpu_durations_dict[tt].append(data)

        # Add L2 / TestMAC times
        if(len(df_l2) > 0):
            temp_df = df_l2[['slot','start_deadline']].copy()
            temp_df['end_deadline'] = temp_df['start_deadline']
            temp_df = temp_df[['slot','end_deadline']]
            df_l2a_start_list.append(temp_df)
            df_l2a_complete_list.append(df_l2[['slot','end_deadline']].copy())

        # Add CPU start/end/durations
        if(len(df_ti) > 0):
            tasks = list(set(df_ti.task))
            for tt in tasks:
                data = df_ti[(df_ti.task==tt) & (df_ti.subtask=='Full Task')][['slot','start_deadline','end_deadline','duration']].copy()
                if tt not in df_ti_full_task_dict.keys():
                    df_ti_full_task_dict[tt] = [data]
                else:
                    df_ti_full_task_dict[tt].append(data)

                subtask_data = df_ti[(df_ti.task==tt)][['sequence','subtask','slot','start_deadline','end_deadline','duration']].copy()
                subtask_data['unique_label'] = subtask_data.sequence.astype(str) + (": " + subtask_data.subtask)
                if tt not in df_ti_all_subtasks_dict.keys():
                    df_ti_all_subtasks_dict[tt] = [subtask_data]
                else:
                    df_ti_all_subtasks_dict[tt].append(subtask_data)

        # Filter compression to only first cell
        if(len(df_compression) > 0):
            df_compression_first_cell_list.append(df_compression[df_compression.cell==df_compression.cell.min()].copy())
        else:
            df_compression_first_cell_list.append(df_compression.copy())

    if(len(df_trigger_complete_list) == 0):
        print("No end of compute timeline data found (are there TI message that include the debug thread?).")
        return


    ### CREATE PLOTS

    print("Creating plots...")

    fig_list = [];

    # Add table with the command line args of the script to the top of the html page (for future reference)
    cmd_dict = vars(args)
    cmdline_table_html = """
    <div style='padding-bottom: 20px;'>
    <b>compare_logs.py's command-line arguments:</b><br>
    <table border='1' style='border-collapse: collapse;'>
    """
    for key, value in cmd_dict.items():
        cmdline_table_html += f"<tr><td><b>{key}</b></td><td>{value}</td></tr>"

    # Add label to input data list fields, if it helps
    cmdline_table_html += f"<tr><td colspan='2'></td></tr>"
    cmdline_table_html += f"<tr><td><b>Input data per label</b></td><td></td></tr>"

    input_data_list = []
    labels = []
    if 'input_data_list' in cmd_dict:
       input_data_list=cmd_dict['input_data_list']
    if 'labels' in cmd_dict:
        labels=cmd_dict['labels']
    
    # If no labels provided (empty list or None), create default "Unlabeled" labels
    if not labels and input_data_list:
        # Determine how many labels we need based on input_data_list length
        # Assume 1-to-1 mapping by default, or 1-to-3 if input_data_list length suggests it
        num_labels_needed = len(input_data_list) // 3 if len(input_data_list) % 3 == 0 and len(input_data_list) > 3 else len(input_data_list)
        labels = ["Unlabeled"] * num_labels_needed
    
    # there's either 1-to-1 or 1-to-3 mapping of labels to input_data_list entries
    labels_to_entries = 1 if (len(labels) == len(input_data_list)) else 3
    for i in range(len(labels)):
        cmdline_table_html += f"<tr><td><b>{labels[i]}</b></td><td>{input_data_list[labels_to_entries*i:labels_to_entries*(i+1)]}</td></tr>"

    cmdline_table_html += "</table></div>"

    script_cmdline_info = Div(text=cmdline_table_html, sizing_mode='stretch_width')
    fig_list.append(row(script_cmdline_info))

    #Label to use for deadline time ccdf plots
    if(slot_list):
        deadline_label = 'Time Relative to OTA Time T0 [Slots %s] (usec)'%slot_list
    else:
        deadline_label = 'Time Relative to OTA Time T0 [All Slots] (usec)'

    if(slot_list):
        duration_label = 'Duration [Slots %s] (usec)'%slot_list
    else:
        duration_label = 'Duration [All Slots] (usec)'

    # Add file legend
    legend_fig = comparison_legend(labels)

    # Add pusch completion times
    PUSCH_DEADLINE = 4500
    PUCCH_DEADLINE = 2000
    PRACH_DEADLINE = 4500
    SAMPLES_READY_DEADLINE = 331+500
    SAMPLES_READY_DEADLINE_SRS = 1831

    if args.mmimo_enable:
        SRS_DEADLINE = 5500
    else:
        SRS_DEADLINE = 4500

    tc_type = getTCType(args.mmimo_enable)
    ULBFW_DEADLINE = getReceptionWindow(TrafficType.DTT_ULC_BFW,tc_type)[0] + 500 #ULBFW must be finished by ULC BFW send time for next slot
    DLBFW_DEADLINE = getReceptionWindow(TrafficType.DTT_DLC_BFW,tc_type)[0] + 500 #DLBFW must be finished by ULC BFW send time for next slot
    DLU_WINDOW_OFFSET1 = getReceptionWindow(TrafficType.DTT_DLU,tc_type)[0]
    DLC_BFW_WINDOW_OFFSET1 = getReceptionWindow(TrafficType.DTT_DLC_BFW,tc_type)[0]
    DLC_NONBFW_WINDOW_OFFSET1 = getReceptionWindow(TrafficType.DTT_DLC_NONBFW,tc_type)[0]
    ULC_BFW_WINDOW_OFFSET1 = getReceptionWindow(TrafficType.DTT_ULC_BFW,tc_type)[0]
    ULC_NONBFW_WINDOW_OFFSET1 = getReceptionWindow(TrafficType.DTT_ULC_NONBFW,tc_type)[0]

    print("Generating UL timeline plots...")
    if enable_ul_timeline:

        # Define timeline data with associated titles and deadlines
        timeline_configs = [
            (df_order_complete_list, "Order Kernel Completion Times", SAMPLES_READY_DEADLINE),
            (df_srs_order_complete_list, "SRS Order Kernel Completion Times", SAMPLES_READY_DEADLINE_SRS),
            (df_pusch_complete_list, "PUSCH Completion Times", PUSCH_DEADLINE),
            (df_pusch_started_list, "PUSCH Started Times", PUSCH_DEADLINE),
            (df_pucch_complete_list, "PUCCH Completion Times", PUCCH_DEADLINE),
            (df_pucch_started_list, "PUCCH Started Times", PUCCH_DEADLINE),
            (df_prach_complete_list, "PRACH Completion Times", PRACH_DEADLINE),
            (df_prach_started_list, "PRACH Started Times", PRACH_DEADLINE),
            (df_srs_complete_list, "SRS Completion Times", SRS_DEADLINE),
            (df_srs_started_list, "SRS Started Times", SRS_DEADLINE),
            (df_ulbfw_complete_list, "ULBFW Completion Times", ULBFW_DEADLINE),
            (df_ulbfw_started_list, "ULBFW Started Times", ULBFW_DEADLINE),
        ]

        for current_data, current_title, current_deadline in timeline_configs:
            if any([len(df)>0 for df in current_data]):
                fig = whisker_comparison(current_data,'end_deadline',['slot','file_index'],
                                         title=current_title, ylabel='Time Relative to OTA Time T0 (usec)')
                
                if(len(fig.x_range.factors) > 0):
                    fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=current_deadline,color='black')

                min_deadline = min([aa['end_deadline'].min() for aa in current_data if len(aa) > 0])
                max_deadline = max([aa['end_deadline'].max() for aa in current_data if len(aa) > 0])
                ccdf_fig = ccdf_comparison(current_data,'end_deadline',title=current_title,xlabel=deadline_label,xstart=min(0,min_deadline),xstop=max_deadline+500)
                ccdf_fig.line(x=current_deadline,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black',legend_label=str(current_deadline))
                ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        # Define duration analysis configurations
        duration_configs = [
            (df_pusch_started_list, df_pusch_complete_list, "PUSCH", PUSCH_DEADLINE - SAMPLES_READY_DEADLINE),
            (df_pucch_started_list, df_pucch_complete_list, "PUCCH", PUCCH_DEADLINE - SAMPLES_READY_DEADLINE),
            (df_prach_started_list, df_prach_complete_list, "PRACH", PRACH_DEADLINE - SAMPLES_READY_DEADLINE),
            (df_srs_started_list, df_srs_complete_list, "SRS", SRS_DEADLINE - SAMPLES_READY_DEADLINE),
            (df_ulbfw_started_list, df_ulbfw_complete_list, "ULBFW", ULBFW_DEADLINE + L2_ADVANCE_TIME),
        ]

        for start_data, end_data, channel_name, max_duration in duration_configs:
            current_data = [aa.reset_index(drop=True).assign(duration=bb['end_deadline'].reset_index(drop=True)-aa['end_deadline'].reset_index(drop=True))[['slot','duration']] for aa,bb in zip(start_data,end_data)]
            if any([len(df)>0 for df in current_data]):
                fig = whisker_comparison(current_data,'duration',['slot','file_index'],
                                         title='%s Total Duration'%channel_name, ylabel='Duration (usec)')
                if(len(fig.x_range.factors) > 0):
                    fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=max_duration,color='black')
                ccdf_fig = ccdf_comparison(current_data,'duration',title="%s Total Duration"%channel_name,xlabel=deadline_label,xstart=0,xstop=max_duration+500)
                ccdf_fig.line(x=max_duration,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black',legend_label=str(max_duration))
                ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
            

        # Add other generic UL CPU-based timelines
        cpu_tuples = [
            ("Early UCI Completion","UL Task AGGR3 Early UCI IND", "Signal Completion", "start_deadline"),
            ("task_work_function_ul_aggr_3_early_uci_ind Task Start Time","UL Task AGGR3 Early UCI IND", "Full Task", "start_deadline"),
            ("task_work_function_ul_aggr_3_early_uci_ind Task End Time","UL Task AGGR3 Early UCI IND", "Full Task", "end_deadline"),
            ("task_work_function_ul_aggr_3_early_uci_ind Task Duration","UL Task AGGR3 Early UCI IND", "Full Task", "duration"),]
        titles = [aa[0] for aa in cpu_tuples]
        tasks = [aa[1] for aa in cpu_tuples]
        subtasks = [aa[2] for aa in cpu_tuples]
        fields = [aa[3] for aa in cpu_tuples]
        for title,task,subtask,field in zip(titles,tasks,subtasks,fields):
            current_data = [df[(df.task==task) & (df.subtask==subtask)] if len(df)>0 else df for df in df_ti_list]

            if any([len(df)>0 for df in current_data]):
                if(field == 'duration'):
                    current_label = duration_label
                    xstart = min([df.duration.min() for df in current_data if len(df) > 0]) - 100
                    xstop = max([df.duration.max() for df in current_data if len(df) > 0]) + 100
                else:
                    current_label = deadline_label
                    xstart = 0
                    xstop = 5000

                fig = whisker_comparison(current_data,field,['slot','file_index'],
                                        title=title,ylabel=current_label)
                ccdf_fig = ccdf_comparison(current_data,field,title=title, xlabel=current_label,xstart=xstart,xstop=xstop)
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

    print("Generating DL timeline plots...")
    if enable_dl_timeline:

        # Define timeline data with associated titles and deadlines
        timeline_configs = [
            (df_dlbfw_started_list, "DLBFW Started Times", DLBFW_DEADLINE),
            (df_dlbfw_complete_list, "DLBFW Completion Times", DLBFW_DEADLINE),
        ]

        for current_data, current_title, current_deadline in timeline_configs:
            if any([len(df)>0 for df in current_data]):
                fig = whisker_comparison(current_data,'end_deadline',['slot','file_index'],
                                         title=current_title, ylabel='Time Relative to OTA Time T0 (usec)')
                
                if(len(fig.x_range.factors) > 0):
                    fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=current_deadline,color='black')

                min_deadline = min([aa['end_deadline'].min() for aa in current_data if len(aa) > 0])
                max_deadline = max([aa['end_deadline'].max() for aa in current_data if len(aa) > 0])
                ccdf_fig = ccdf_comparison(current_data,'end_deadline',title=current_title,xlabel=deadline_label,xstart=min(0,min_deadline),xstop=max_deadline+500)
                ccdf_fig.line(x=current_deadline,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black',legend_label=str(current_deadline))
                ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        # Define duration analysis configurations
        duration_configs = [
            (df_dlbfw_started_list, df_dlbfw_complete_list, "DLBFW", DLBFW_DEADLINE + L2_ADVANCE_TIME),
        ]

        for start_data, end_data, channel_name, max_duration in duration_configs:
            current_data = [aa.reset_index(drop=True).assign(duration=bb['end_deadline'].reset_index(drop=True)-aa['end_deadline'].reset_index(drop=True))[['slot','duration']] for aa,bb in zip(start_data,end_data)]
            if any([len(df)>0 for df in current_data]):
                fig = whisker_comparison(current_data,'duration',['slot','file_index'],
                                         title='%s Total Duration'%channel_name, ylabel='Duration (usec)')
                if(len(fig.x_range.factors) > 0):
                    fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=max_duration,color='black')
                ccdf_fig = ccdf_comparison(current_data,'duration',title="%s Total Duration"%channel_name,xlabel=deadline_label,xstart=0,xstop=max_duration+500)
                ccdf_fig.line(x=max_duration,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black',legend_label=str(max_duration))
                ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        # Add Trigger completion times
        if any([len(df)>0 for df in df_trigger_complete_list]):
            title = "Trigger Kernel Completion Times"
            fig = whisker_comparison(df_trigger_complete_list,'end_deadline',['slot','file_index'],
                                    title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=DLU_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(df_trigger_complete_list,'end_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=DLU_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        dlc_bfw_started_list = []
        dlc_bfw_complete_list = []
        dlc_bfw_duration_list = []
        dlc_nonbfw_started_list = []
        dlc_nonbfw_complete_list = []
        dlc_nonbfw_duration_list = []
        df_ti_short = df_ti[df_ti.tir - df_ti.tir.min() < 0.5]
        num_dlc_tasks = get_num_dlc_tasks(df_ti_short)
        
        for df_ti in df_ti_list:
            dlc_tasks = ['DL Task C-Plane %i'%(ii+1) for ii in range(num_dlc_tasks)]
            
            # Check if new subtasks exist
            has_new_subtasks, bfw_subtasks, nonbfw_subtasks = has_bfw_nonbfw_subtasks(df_ti, dlc_tasks)
            
            if has_new_subtasks:
                # Process BFW subtasks
                if len(bfw_subtasks) > 0:
                    # Calculate first start time across all dlc bfw subtasks
                    temp_df1 = df_ti[df_ti.task.isin(dlc_tasks) & df_ti.subtask.isin(bfw_subtasks)][['t0_timestamp','slot','task','start_deadline']].copy()
                    temp_df1 = temp_df1.groupby('t0_timestamp').min().reset_index()

                    # Calculate last end time across all dlc bfw subtasks
                    temp_df2 = df_ti[df_ti.task.isin(dlc_tasks) & df_ti.subtask.isin(bfw_subtasks)][['t0_timestamp','slot','task','end_deadline']].copy()
                    temp_df2 = temp_df2.groupby('t0_timestamp').max().reset_index()

                    # Calculate duration by subtracting start_deadline from end_deadline between temp_df1 and temp_df2
                    temp_df3 = temp_df2.merge(temp_df1, on=['t0_timestamp','slot'], validate='one_to_one')
                    temp_df3['duration'] = temp_df3['end_deadline'] - temp_df3['start_deadline']

                    dlc_bfw_started_list.append(temp_df1)
                    dlc_bfw_complete_list.append(temp_df2)
                    dlc_bfw_duration_list.append(temp_df3)
                else:
                    # No BFW subtasks, append empty dataframes
                    empty_df = df_ti[df_ti.task.isin(dlc_tasks)][['t0_timestamp','slot','task','start_deadline']].iloc[0:0].copy()
                    dlc_bfw_started_list.append(empty_df)
                    dlc_bfw_complete_list.append(empty_df.rename(columns={'start_deadline': 'end_deadline'}))
                    dlc_bfw_duration_list.append(empty_df.assign(duration=[]).rename(columns={'start_deadline': 'end_deadline'}))

                # Process non-BFW subtasks
                if len(nonbfw_subtasks) > 0:
                    # Calculate first start time across all dlc non-bfw subtasks
                    temp_df1 = df_ti[df_ti.task.isin(dlc_tasks) & df_ti.subtask.isin(nonbfw_subtasks)][['t0_timestamp','slot','task','start_deadline']].copy()
                    temp_df1 = temp_df1.groupby('t0_timestamp').min().reset_index()

                    # Calculate last end time across all dlc non-bfw subtasks
                    temp_df2 = df_ti[df_ti.task.isin(dlc_tasks) & df_ti.subtask.isin(nonbfw_subtasks)][['t0_timestamp','slot','task','end_deadline']].copy()
                    temp_df2 = temp_df2.groupby('t0_timestamp').max().reset_index()

                    # Calculate duration by subtracting start_deadline from end_deadline between temp_df1 and temp_df2
                    temp_df3 = temp_df2.merge(temp_df1, on=['t0_timestamp','slot'], validate='one_to_one')
                    temp_df3['duration'] = temp_df3['end_deadline'] - temp_df3['start_deadline']

                    dlc_nonbfw_started_list.append(temp_df1)
                    dlc_nonbfw_complete_list.append(temp_df2)
                    dlc_nonbfw_duration_list.append(temp_df3)
                else:
                    # No non-BFW subtasks, append empty dataframes
                    empty_df = df_ti[df_ti.task.isin(dlc_tasks)][['t0_timestamp','slot','task','start_deadline']].iloc[0:0].copy()
                    dlc_nonbfw_started_list.append(empty_df)
                    dlc_nonbfw_complete_list.append(empty_df.rename(columns={'start_deadline': 'end_deadline'}))
                    dlc_nonbfw_duration_list.append(empty_df.assign(duration=[]).rename(columns={'start_deadline': 'end_deadline'}))
            else:
                # Fall back to old method using 'CPlane Prepare' - populate both BFW and non-BFW with same data
                dlc_subtask = 'CPlane Prepare'

                #Calculate first start time across all dlc tasks
                temp_df1 = df_ti[df_ti.task.isin(dlc_tasks) & (df_ti.subtask==dlc_subtask)][['t0_timestamp','slot','task','start_deadline']].copy()
                temp_df1 = temp_df1.groupby('t0_timestamp').min().reset_index()

                #Calculate last end time across all dlc tasks
                temp_df2 = df_ti[df_ti.task.isin(dlc_tasks) & (df_ti.subtask==dlc_subtask)][['t0_timestamp','slot','task','end_deadline']].copy()
                temp_df2 = temp_df2.groupby('t0_timestamp').max().reset_index()

                # Calculate duration by subtracting start_deadline from end_deadline between temp_df1 and temp_df2
                temp_df3 = temp_df2.merge(temp_df1, on=['t0_timestamp','slot'], validate='one_to_one')
                temp_df3['duration'] = temp_df3['end_deadline'] - temp_df3['start_deadline']

                # Populate both BFW and non-BFW with the same data
                dlc_bfw_started_list.append(temp_df1)
                dlc_bfw_complete_list.append(temp_df2)
                dlc_bfw_duration_list.append(temp_df3)
                dlc_nonbfw_started_list.append(temp_df1)
                dlc_nonbfw_complete_list.append(temp_df2)
                dlc_nonbfw_duration_list.append(temp_df3)

        # Create separate plots for BFW
        if any([len(df)>0 for df in dlc_bfw_started_list]): 
            title = "DLC BFW Started Times"
            fig = whisker_comparison(dlc_bfw_started_list,'start_deadline',['slot','file_index'],
                                     title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=DLC_BFW_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(dlc_bfw_started_list,'start_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=DLC_BFW_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        if any([len(df)>0 for df in dlc_bfw_complete_list]): 
            title = "DLC BFW Completion Times"
            fig = whisker_comparison(dlc_bfw_complete_list,'end_deadline',['slot','file_index'],
                                    title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=DLC_BFW_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(dlc_bfw_complete_list,'end_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=DLC_BFW_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        if any([len(df)>0 for df in dlc_bfw_duration_list]):
            max_duration = max([df.duration.max() for df in dlc_bfw_duration_list if len(df) > 0])
            title = "DLC BFW Duration"
            fig = whisker_comparison(dlc_bfw_duration_list,'duration',['slot','file_index'],
                                     title=title, ylabel='Duration (usec)')
            ccdf_fig = ccdf_comparison(dlc_bfw_duration_list,'duration',title=title,xlabel=duration_label,xstart=0,xstop=max_duration+100)
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        # Create separate plots for non-BFW
        if any([len(df)>0 for df in dlc_nonbfw_started_list]): 
            title = "DLC non-BFW Started Times"
            fig = whisker_comparison(dlc_nonbfw_started_list,'start_deadline',['slot','file_index'],
                                     title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=DLC_NONBFW_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(dlc_nonbfw_started_list,'start_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=DLC_NONBFW_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        if any([len(df)>0 for df in dlc_nonbfw_complete_list]): 
            title = "DLC non-BFW Completion Times"
            fig = whisker_comparison(dlc_nonbfw_complete_list,'end_deadline',['slot','file_index'],
                                    title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=DLC_NONBFW_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(dlc_nonbfw_complete_list,'end_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=DLC_NONBFW_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        if any([len(df)>0 for df in dlc_nonbfw_duration_list]):
            max_duration = max([df.duration.max() for df in dlc_nonbfw_duration_list if len(df) > 0])
            title = "DLC non-BFW Duration"
            fig = whisker_comparison(dlc_nonbfw_duration_list,'duration',['slot','file_index'],
                                     title=title, ylabel='Duration (usec)')
            ccdf_fig = ccdf_comparison(dlc_nonbfw_duration_list,'duration',title=title,xlabel=duration_label,xstart=0,xstop=max_duration+100)
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        #For each dlc task, create an individual CCDF plot (only if breakouts enabled)
        if(enable_ulc_dlc_breakouts):
            dlc_task_duration_ccdf_list = []
            dlc_task_duration_whisker_list = []
            dlc_task_start_ccdf_list = []
            dlc_task_start_whisker_list = []
            dlc_task_end_ccdf_list = []
            dlc_task_end_whisker_list = []
            
            for subtask_label in ["BFW","non-BFW"]:
                for task in dlc_tasks:
                    # Build current_data by checking each df_ti individually for subtask format
                    current_data = []
                    for df_ti in df_ti_list:
                        current_has_new_subtasks, current_bfw_subtasks, current_nonbfw_subtasks = has_bfw_nonbfw_subtasks(df_ti, dlc_tasks)
                        if current_has_new_subtasks:
                            # Use the appropriate subtask list for this df_ti
                            if subtask_label == "BFW":
                                current_subtask_list = current_bfw_subtasks
                            else:  # non-BFW
                                current_subtask_list = current_nonbfw_subtasks
                            df_subset = df_ti[(df_ti.task==task) & df_ti.subtask.isin(current_subtask_list)]
                        else:
                            # Use the old 'CPlane Prepare' subtask
                            df_subset = df_ti[(df_ti.task==task) & (df_ti.subtask=='CPlane Prepare')]
                        current_data.append(df_subset)
                        
                    if any([len(df) > 0 for df in current_data]):
                        # Duration plots
                        task_title = "%s : %s Prepare (Duration)"%(task,subtask_label)
                        max_duration = max([df.duration.max() for df in current_data if len(df) > 0])
                        fig = whisker_comparison(current_data,'duration',['slot','file_index'],
                                                 title=task_title, ylabel='Duration (usec)')
                        ccdf_fig = ccdf_comparison(current_data,'duration',title=task_title,xlabel=duration_label,xstart=0,xstop=max_duration+100)
                        ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'

                        # Create a new figure with the same properties instead of trying to copy
                        ccdf_fig2 = ccdf_comparison(current_data,'duration',title=task_title,xlabel=duration_label,xstart=0,xstop=max_duration+100)
                        ccdf_fig2.yaxis.axis_label = '(1-CDF) Probability'
                        
                        dlc_task_duration_ccdf_list.append(column(legend_fig,ccdf_fig2))
                        dlc_task_duration_whisker_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
                        
                        # Start Deadline plots
                        task_title_start = "%s : %s Prepare (Start Time)"%(task,subtask_label)
                        fig_start = whisker_comparison(current_data,'start_deadline',['slot','file_index'],
                                                       title=task_title_start, ylabel='Time Relative to OTA Time T0 (usec)')
                        ccdf_fig_start = ccdf_comparison(current_data,'start_deadline',title=task_title_start,xlabel=deadline_label)
                        ccdf_fig_start.yaxis.axis_label = '(1-CDF) Probability'
                        
                        ccdf_fig_start2 = ccdf_comparison(current_data,'start_deadline',title=task_title_start,xlabel=deadline_label)
                        ccdf_fig_start2.yaxis.axis_label = '(1-CDF) Probability'
                        
                        dlc_task_start_ccdf_list.append(column(legend_fig,ccdf_fig_start2))
                        dlc_task_start_whisker_list.append(row(column(legend_fig,fig_start),column(legend_fig,ccdf_fig_start)))
                        
                        # End Deadline plots
                        task_title_end = "%s : %s Prepare (End Time)"%(task,subtask_label)
                        fig_end = whisker_comparison(current_data,'end_deadline',['slot','file_index'],
                                                     title=task_title_end, ylabel='Time Relative to OTA Time T0 (usec)')
                        ccdf_fig_end = ccdf_comparison(current_data,'end_deadline',title=task_title_end,xlabel=deadline_label)
                        ccdf_fig_end.yaxis.axis_label = '(1-CDF) Probability'
                        
                        ccdf_fig_end2 = ccdf_comparison(current_data,'end_deadline',title=task_title_end,xlabel=deadline_label)
                        ccdf_fig_end2.yaxis.axis_label = '(1-CDF) Probability'
                        
                        dlc_task_end_ccdf_list.append(column(legend_fig,ccdf_fig_end2))
                        dlc_task_end_whisker_list.append(row(column(legend_fig,fig_end),column(legend_fig,ccdf_fig_end)))

            # Add duration plots as separate section
            if len(dlc_task_duration_ccdf_list) > 0:
                fig_list.append(row(dlc_task_duration_ccdf_list))
                fig_list.append(column(dlc_task_duration_whisker_list))
            
            # Add start deadline plots as separate section
            if len(dlc_task_start_ccdf_list) > 0:
                fig_list.append(row(dlc_task_start_ccdf_list))
                fig_list.append(column(dlc_task_start_whisker_list))
            
            # Add end deadline plots as separate section
            if len(dlc_task_end_ccdf_list) > 0:
                fig_list.append(row(dlc_task_end_ccdf_list))
                fig_list.append(column(dlc_task_end_whisker_list))

        # Add ULC BFW/non-BFW plots
        ulc_bfw_started_list = []
        ulc_bfw_complete_list = []
        ulc_bfw_duration_list = []
        ulc_nonbfw_started_list = []
        ulc_nonbfw_complete_list = []
        ulc_nonbfw_duration_list = []
        df_ti_short = df_ti[df_ti.tir - df_ti.tir.min() < 0.5]
        num_ulc_tasks = get_num_ulc_tasks(df_ti_short)
        
        for df_ti in df_ti_list:
            ulc_tasks = ['UL Task CPlane %i'%(ii+1) for ii in range(num_ulc_tasks)]
            
            # Check if new subtasks exist
            has_new_subtasks, bfw_subtasks, nonbfw_subtasks = has_bfw_nonbfw_subtasks(df_ti, ulc_tasks)
            
            if has_new_subtasks:
                # Process BFW subtasks
                if len(bfw_subtasks) > 0:
                    # Calculate first start time across all ulc bfw subtasks
                    temp_df1 = df_ti[df_ti.task.isin(ulc_tasks) & df_ti.subtask.isin(bfw_subtasks)][['t0_timestamp','slot','task','start_deadline']].copy()
                    temp_df1 = temp_df1.groupby('t0_timestamp').min().reset_index()

                    # Calculate last end time across all ulc bfw subtasks
                    temp_df2 = df_ti[df_ti.task.isin(ulc_tasks) & df_ti.subtask.isin(bfw_subtasks)][['t0_timestamp','slot','task','end_deadline']].copy()
                    temp_df2 = temp_df2.groupby('t0_timestamp').max().reset_index()

                    # Calculate duration by subtracting start_deadline from end_deadline between temp_df1 and temp_df2
                    temp_df3 = temp_df2.merge(temp_df1, on=['t0_timestamp','slot'], validate='one_to_one')
                    temp_df3['duration'] = temp_df3['end_deadline'] - temp_df3['start_deadline']

                    ulc_bfw_started_list.append(temp_df1)
                    ulc_bfw_complete_list.append(temp_df2)
                    ulc_bfw_duration_list.append(temp_df3)
                else:
                    # No BFW subtasks, append empty dataframes
                    empty_df = df_ti[df_ti.task.isin(ulc_tasks)][['t0_timestamp','slot','task','start_deadline']].iloc[0:0].copy()
                    ulc_bfw_started_list.append(empty_df)
                    ulc_bfw_complete_list.append(empty_df.rename(columns={'start_deadline': 'end_deadline'}))
                    ulc_bfw_duration_list.append(empty_df.assign(duration=[]).rename(columns={'start_deadline': 'end_deadline'}))

                # Process non-BFW subtasks
                if len(nonbfw_subtasks) > 0:
                    # Calculate first start time across all ulc non-bfw subtasks
                    temp_df1 = df_ti[df_ti.task.isin(ulc_tasks) & df_ti.subtask.isin(nonbfw_subtasks)][['t0_timestamp','slot','task','start_deadline']].copy()
                    temp_df1 = temp_df1.groupby('t0_timestamp').min().reset_index()

                    # Calculate last end time across all ulc non-bfw subtasks
                    temp_df2 = df_ti[df_ti.task.isin(ulc_tasks) & df_ti.subtask.isin(nonbfw_subtasks)][['t0_timestamp','slot','task','end_deadline']].copy()
                    temp_df2 = temp_df2.groupby('t0_timestamp').max().reset_index()

                    # Calculate duration by subtracting start_deadline from end_deadline between temp_df1 and temp_df2
                    temp_df3 = temp_df2.merge(temp_df1, on=['t0_timestamp','slot'], validate='one_to_one')
                    temp_df3['duration'] = temp_df3['end_deadline'] - temp_df3['start_deadline']

                    ulc_nonbfw_started_list.append(temp_df1)
                    ulc_nonbfw_complete_list.append(temp_df2)
                    ulc_nonbfw_duration_list.append(temp_df3)
                else:
                    # No non-BFW subtasks, append empty dataframes
                    empty_df = df_ti[df_ti.task.isin(ulc_tasks)][['t0_timestamp','slot','task','start_deadline']].iloc[0:0].copy()
                    ulc_nonbfw_started_list.append(empty_df)
                    ulc_nonbfw_complete_list.append(empty_df.rename(columns={'start_deadline': 'end_deadline'}))
                    ulc_nonbfw_duration_list.append(empty_df.assign(duration=[]).rename(columns={'start_deadline': 'end_deadline'}))
            else:
                # Fall back to old method with special ULC logic ('Send C Plane' or 'CPlane Prepare')
                ulc_subtask = 'Send C Plane'
                if(len(df_ti[df_ti.task.isin(ulc_tasks) & (df_ti.subtask=='CPlane Prepare')]) > 0):
                    ulc_subtask = 'CPlane Prepare'

                #Calculate first start time across all ulc tasks
                temp_df1 = df_ti[df_ti.task.isin(ulc_tasks) & (df_ti.subtask==ulc_subtask)][['t0_timestamp','slot','task','start_deadline']].copy()
                temp_df1 = temp_df1.groupby('t0_timestamp').min().reset_index()

                #Calculate last end time across all ulc tasks
                temp_df2 = df_ti[df_ti.task.isin(ulc_tasks) & (df_ti.subtask==ulc_subtask)][['t0_timestamp','slot','task','end_deadline']].copy()
                temp_df2 = temp_df2.groupby('t0_timestamp').max().reset_index()

                # Calculate duration by subtracting start_deadline from end_deadline between temp_df1 and temp_df2
                temp_df3 = temp_df2.merge(temp_df1, on=['t0_timestamp','slot'], validate='one_to_one')
                temp_df3['duration'] = temp_df3['end_deadline'] - temp_df3['start_deadline']

                # Populate both BFW and non-BFW with the same data
                ulc_bfw_started_list.append(temp_df1)
                ulc_bfw_complete_list.append(temp_df2)
                ulc_bfw_duration_list.append(temp_df3)
                ulc_nonbfw_started_list.append(temp_df1)
                ulc_nonbfw_complete_list.append(temp_df2)
                ulc_nonbfw_duration_list.append(temp_df3)

        # Create separate plots for ULC BFW
        if any([len(df)>0 for df in ulc_bfw_started_list]): 
            title = "ULC BFW Started Times"
            fig = whisker_comparison(ulc_bfw_started_list,'start_deadline',['slot','file_index'],
                                     title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=ULC_BFW_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(ulc_bfw_started_list,'start_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=ULC_BFW_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        if any([len(df)>0 for df in ulc_bfw_complete_list]): 
            title = "ULC BFW Completion Times"
            fig = whisker_comparison(ulc_bfw_complete_list,'end_deadline',['slot','file_index'],
                                    title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=ULC_BFW_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(ulc_bfw_complete_list,'end_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=ULC_BFW_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        if any([len(df)>0 for df in ulc_bfw_duration_list]):
            max_duration = max([df.duration.max() for df in ulc_bfw_duration_list if len(df) > 0])
            title = "ULC BFW Duration"
            fig = whisker_comparison(ulc_bfw_duration_list,'duration',['slot','file_index'],
                                     title=title, ylabel='Duration (usec)')
            ccdf_fig = ccdf_comparison(ulc_bfw_duration_list,'duration',title=title,xlabel=duration_label,xstart=0,xstop=max_duration+100)
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        # Create separate plots for ULC non-BFW
        if any([len(df)>0 for df in ulc_nonbfw_started_list]): 
            title = "ULC non-BFW Started Times"
            fig = whisker_comparison(ulc_nonbfw_started_list,'start_deadline',['slot','file_index'],
                                     title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=ULC_NONBFW_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(ulc_nonbfw_started_list,'start_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=ULC_NONBFW_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        if any([len(df)>0 for df in ulc_nonbfw_complete_list]): 
            title = "ULC non-BFW Completion Times"
            fig = whisker_comparison(ulc_nonbfw_complete_list,'end_deadline',['slot','file_index'],
                                    title=title, ylabel='Time Relative to OTA Time T0 (usec)')
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=ULC_NONBFW_WINDOW_OFFSET1,color='black')
            ccdf_fig = ccdf_comparison(ulc_nonbfw_complete_list,'end_deadline',title=title,xlabel=deadline_label)
            ccdf_fig.line(x=ULC_NONBFW_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        if any([len(df)>0 for df in ulc_nonbfw_duration_list]):
            max_duration = max([df.duration.max() for df in ulc_nonbfw_duration_list if len(df) > 0])
            title = "ULC non-BFW Duration"
            fig = whisker_comparison(ulc_nonbfw_duration_list,'duration',['slot','file_index'],
                                     title=title, ylabel='Duration (usec)')
            ccdf_fig = ccdf_comparison(ulc_nonbfw_duration_list,'duration',title=title,xlabel=duration_label,xstart=0,xstop=max_duration+100)
            ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
            fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        #For each ulc task, create an individual CCDF plot (only if breakouts enabled)
        if(enable_ulc_dlc_breakouts):
            ulc_task_duration_ccdf_list = []
            ulc_task_duration_whisker_list = []
            ulc_task_start_ccdf_list = []
            ulc_task_start_whisker_list = []
            ulc_task_end_ccdf_list = []
            ulc_task_end_whisker_list = []
            
            for subtask_label in ["BFW","non-BFW"]:
                for task in ulc_tasks:
                    # Build current_data by checking each df_ti individually for subtask format
                    current_data = []
                    for df_ti in df_ti_list:
                        current_has_new_subtasks, current_bfw_subtasks, current_nonbfw_subtasks = has_bfw_nonbfw_subtasks(df_ti, ulc_tasks)
                        if current_has_new_subtasks:
                            # Use the appropriate subtask list for this df_ti
                            if subtask_label == "BFW":
                                current_subtask_list = current_bfw_subtasks
                            else:  # non-BFW
                                current_subtask_list = current_nonbfw_subtasks
                            df_subset = df_ti[(df_ti.task==task) & df_ti.subtask.isin(current_subtask_list)]
                        else:
                            # Use the appropriate legacy subtask ('Send C Plane' or 'CPlane Prepare')
                            ulc_subtask = 'Send C Plane'
                            if(len(df_ti[df_ti.task.isin(ulc_tasks) & (df_ti.subtask=='CPlane Prepare')]) > 0):
                                ulc_subtask = 'CPlane Prepare'
                            df_subset = df_ti[(df_ti.task==task) & (df_ti.subtask==ulc_subtask)]
                        current_data.append(df_subset)
                        
                    if any([len(df) > 0 for df in current_data]):
                        # Duration plots
                        task_title = "%s : %s Prepare (Duration)"%(task,subtask_label)
                        max_duration = max([df.duration.max() for df in current_data if len(df) > 0])
                        fig = whisker_comparison(current_data,'duration',['slot','file_index'],
                                                 title=task_title, ylabel='Duration (usec)')
                        ccdf_fig = ccdf_comparison(current_data,'duration',title=task_title,xlabel=duration_label,xstart=0,xstop=max_duration+100)
                        ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'

                        # Create a new figure with the same properties instead of trying to copy
                        ccdf_fig2 = ccdf_comparison(current_data,'duration',title=task_title,xlabel=duration_label,xstart=0,xstop=max_duration+100)
                        ccdf_fig2.yaxis.axis_label = '(1-CDF) Probability'
                        
                        ulc_task_duration_ccdf_list.append(column(legend_fig,ccdf_fig2))
                        ulc_task_duration_whisker_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
                        
                        # Start Deadline plots
                        task_title_start = "%s : %s Prepare (Start Time)"%(task,subtask_label)
                        fig_start = whisker_comparison(current_data,'start_deadline',['slot','file_index'],
                                                       title=task_title_start, ylabel='Time Relative to OTA Time T0 (usec)')
                        ccdf_fig_start = ccdf_comparison(current_data,'start_deadline',title=task_title_start,xlabel=deadline_label)
                        ccdf_fig_start.yaxis.axis_label = '(1-CDF) Probability'
                        
                        ccdf_fig_start2 = ccdf_comparison(current_data,'start_deadline',title=task_title_start,xlabel=deadline_label)
                        ccdf_fig_start2.yaxis.axis_label = '(1-CDF) Probability'
                        
                        ulc_task_start_ccdf_list.append(column(legend_fig,ccdf_fig_start2))
                        ulc_task_start_whisker_list.append(row(column(legend_fig,fig_start),column(legend_fig,ccdf_fig_start)))
                        
                        # End Deadline plots
                        task_title_end = "%s : %s Prepare (End Time)"%(task,subtask_label)
                        fig_end = whisker_comparison(current_data,'end_deadline',['slot','file_index'],
                                                     title=task_title_end, ylabel='Time Relative to OTA Time T0 (usec)')
                        ccdf_fig_end = ccdf_comparison(current_data,'end_deadline',title=task_title_end,xlabel=deadline_label)
                        ccdf_fig_end.yaxis.axis_label = '(1-CDF) Probability'
                        
                        ccdf_fig_end2 = ccdf_comparison(current_data,'end_deadline',title=task_title_end,xlabel=deadline_label)
                        ccdf_fig_end2.yaxis.axis_label = '(1-CDF) Probability'
                        
                        ulc_task_end_ccdf_list.append(column(legend_fig,ccdf_fig_end2))
                        ulc_task_end_whisker_list.append(row(column(legend_fig,fig_end),column(legend_fig,ccdf_fig_end)))

            # Add duration plots as separate section
            if len(ulc_task_duration_ccdf_list) > 0:
                fig_list.append(row(ulc_task_duration_ccdf_list))
                fig_list.append(column(ulc_task_duration_whisker_list))
            
            # Add start deadline plots as separate section
            if len(ulc_task_start_ccdf_list) > 0:
                fig_list.append(row(ulc_task_start_ccdf_list))
                fig_list.append(column(ulc_task_start_whisker_list))
            
            # Add end deadline plots as separate section
            if len(ulc_task_end_ccdf_list) > 0:
                fig_list.append(row(ulc_task_end_ccdf_list))
                fig_list.append(column(ulc_task_end_whisker_list))

        # Add Compression completion times
        if any([len(df)>0 for df in df_compression_complete_list]):
            title = "Compression Kernel Completion Times"
            fig = whisker_comparison(df_compression_complete_list,'end_deadline',['slot','file_index'],
                                    title=title, ylabel='Compression Complete Deadline Time (usec)')
            ccdf_fig = ccdf_comparison(df_compression_complete_list,'end_deadline',title=title,xlabel=deadline_label)
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=DLU_WINDOW_OFFSET1,color='black')
            fig_list.append(row(fig,ccdf_fig))

        # Add Compression start times
        if any([len(df)>0 for df in df_compression_start_list]):
            title = "Compression Kernel Start Times"
            fig = whisker_comparison(df_compression_start_list,'end_deadline',['slot','file_index'],
                                    title=title, ylabel='Compression Start Deadline Time (usec)')
            ccdf_fig = ccdf_comparison(df_compression_start_list,'end_deadline',title=title,xlabel=deadline_label)
            if(len(fig.x_range.factors) > 0):
                fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=DLU_WINDOW_OFFSET1,color='black')
            fig_list.append(row(fig,ccdf_fig))

        # Add channel gap/prepare/compression information
        titles = ["Channel To Compression Gap","Prepare Memset Duration","Pre-Prepare Kernel Duration","Prepare Kernel Duration","Prepare Total Duration","Compression Duration"]
        fields = ['channel_to_compression_gap','prepare_memsets','pre_prepare_kernel','prepare_kernel','prepare_total','compression_kernel']
        for title,field in zip(titles,fields):
            current_data = [df[df.event==field] if len(df) > 0 else df for df in df_compression_first_cell_list]
            if any([len(df)>0 for df in current_data]):
                fig = whisker_comparison(current_data,'duration',['slot','file_index'],
                                        title=title,ylabel=duration_label)
                ccdf_fig = ccdf_comparison(current_data,'duration',title=title, xlabel=duration_label,
                                        xstart=min([df.duration.min() for df in current_data if len(df) > 0]) - 100,
                                        xstop=max([df.duration.max() for df in current_data if len(df) > 0]) + 100)
                fig_list.append(row(fig,ccdf_fig))

        #Determine if prepare/tx has been separated
        num_gpu_comms_prepare_tasks = get_num_gpu_comms_prepare_tasks(df_ti)

        # Add prepare completion information
        gpu_comms_tuples = []
        if num_gpu_comms_prepare_tasks > 0:
            #Around May 2025 - prepare work was separated into task that can be fanned out, and TX portion in separate task

            #Add GPU Comms Prepare timelines (one for each task)
            for i in range(1, num_gpu_comms_prepare_tasks + 1):
                gpu_comms_tuples.extend([
                    (f"UPlane Prepare CPU Work Start Time {i}", f"DL Task GPU Comms Prepare {i}", "UPlane Prepare", "start_deadline"),
                    (f"UPlane Prepare CPU Work End Time {i}", f"DL Task GPU Comms Prepare {i}", "UPlane Prepare", "end_deadline"),
                    (f"UPlane Prepare CPU Work Duration {i}", f"DL Task GPU Comms Prepare {i}", "UPlane Prepare", "duration"),
                ])

            #Add GPU Comms TX timelines
            gpu_comms_tuples.extend([
                ("UPlane Tx CPU Work Start Time","DL Task GPU Comms TX", "Uplane TX", "start_deadline"),
                ("UPlane Tx CPU Work End Time","DL Task GPU Comms TX", "Uplane TX", "end_deadline"),
                ("UPlane Tx CPU Work Duration","DL Task GPU Comms TX", "Uplane TX", "duration"),
            ])
            
        else:
            #Keep backwards compatibility where uplane prepare is inside the GPU Comms task
            gpu_comms_tuples.extend([
                ("UPlane Prepare CPU Work Start Time","DL Task GPU Comms", "UPlane Prepare", "start_deadline"),
                ("UPlane Prepare CPU Work End Time","DL Task GPU Comms", "UPlane Prepare", "end_deadline"),
                ("UPlane Prepare CPU Work Duration","DL Task GPU Comms", "UPlane Prepare", "duration"),
            ])


        cpu_tuples = [("FH Callback Start Time","DL Task FH Callback", "Full Task", "start_deadline"),
                      ("FH Callback End Time","DL Task FH Callback", "Full Task", "end_deadline"),
                      ("FH Callback Duration","DL Task FH Callback", "Full Task", "duration"),
                      ("Compression Completion Time", "Debug Task", "Compression Wait", "start_deadline"),
                      ("Trigger Start Time", "Debug Task", "Trigger synchronize", "start_deadline"),
                      ("Trigger End Time", "Debug Task", "Trigger synchronize", "end_deadline"),
                      ("Trigger Duration", "Debug Task", "Trigger synchronize", "duration"),]
        
        # Extend cpu_tuples with the GPU Comms tuples
        cpu_tuples.extend(gpu_comms_tuples)

        titles = [aa[0] for aa in cpu_tuples]
        tasks = [aa[1] for aa in cpu_tuples]
        subtasks = [aa[2] for aa in cpu_tuples]
        fields = [aa[3] for aa in cpu_tuples]
        for title,task,subtask,field in zip(titles,tasks,subtasks,fields):
            current_data = [df[(df.task==task) & (df.subtask==subtask)] if len(df)>0 else df for df in df_ti_list]

            if any(len(aa)>0 for aa in current_data):
                if(field == 'duration'):
                    current_label = duration_label
                    xstart = min([df.duration.min() for df in current_data if len(df) > 0]) - 100
                    xstop = max([df.duration.max() for df in current_data if len(df) > 0]) + 100
                else:
                    current_label = deadline_label
                    xstart = -1500
                    xstop = 0

                fig = whisker_comparison(current_data,field,['slot','file_index'],
                                        title=title,ylabel=current_label)
                ccdf_fig = ccdf_comparison(current_data,field,title=title, xlabel=current_label,xstart=xstart,xstop=xstop)
                if(field != 'duration' and len(fig.x_range.factors) > 0):
                    fig.line(x=[fig.x_range.factors[0],fig.x_range.factors[-1]],y=DLU_WINDOW_OFFSET1,color='black')
                    ccdf_fig.line(x=DLU_WINDOW_OFFSET1,y=[ccdf_fig.y_range.start,ccdf_fig.y_range.end],color='black')
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

    print("Generating L2A timeline plots...")
    # Add L2/testmac completion times
    if(all(len(df)>0 for df in df_l2a_complete_list) and enable_l2a_timeline):
        title = "L2A Completion Times"
        fig = whisker_comparison(df_l2a_complete_list,'end_deadline',['slot','file_index'],
                                 title=title, ylabel='L2A Complete Deadline Time (usec)')
        ccdf_fig = ccdf_comparison(df_l2a_complete_list,'end_deadline',title=title,xlabel=deadline_label)
        fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        title="L2A Start Times"
        fig = whisker_comparison(df_l2a_start_list,'end_deadline',['slot','file_index'],
                                 title=title, ylabel='L2A Start Deadline Time (usec)')
        ccdf_fig = ccdf_comparison(df_l2a_start_list,'end_deadline',title=title,xlabel=deadline_label)
        fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

    if(all(len(df)>0 for df in df_testmac_list) and enable_l2a_timeline):
        #Populate deadline-based times for testmac start/stop
        for df_testmac in df_testmac_list:
            df_testmac['start_deadline'] = (df_testmac['fapi2_start_timestamp'] - df_testmac['t0_timestamp'])/1e3
            df_testmac['end_deadline'] = (df_testmac['fapi2_stop_timestamp'] - df_testmac['t0_timestamp'])/1e3

        title="TestMAC Last Send Times"
        fig = whisker_comparison(df_testmac_list,'end_deadline',['slot','file_index'],
                                 title=title, ylabel='TestMAC Last Send Deadline Time (usec)')
        ccdf_fig = ccdf_comparison(df_testmac_list,'end_deadline',title=title,xlabel=deadline_label)
        fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

        title="TestMAC First Send Times"
        fig = whisker_comparison(df_testmac_list,'start_deadline',['slot','file_index'],
                                 title=title, ylabel='TestMAC First Send Deadline Time (usec)')
        ccdf_fig = ccdf_comparison(df_testmac_list,'start_deadline',title=title,xlabel=deadline_label)
        fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))


    if(HAS_TICK_TIMES and enable_l2a_timeline):
        title = "Tick CPU Times"
        fig = whisker_comparison(df_tick_complete_list,'cpu_deadline',['slot','file_index'],
                                 title=title, ylabel='Tick CPU Deadline Time (usec)')
        ccdf_fig = ccdf_comparison(df_tick_complete_list,'cpu_deadline',title=title,xlabel=deadline_label)
        fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

    print("Generating GPU duration plots...")
    # Add GPU durations
    if(enable_gpu_durations):
        gpu_tasks = list(df_gpu_durations_dict.keys())
        gpu_tasks.sort()
        for task in gpu_tasks:
            max_entries = max([len(df) for df in df_gpu_durations_dict[task]])
            if(max_entries > 0):
                fig = whisker_comparison(df_gpu_durations_dict[task],'gpu_total_duration',['slot','file_index'],
                                        "GPU Execution (%s) Total Duration (Setup+Run)"%task, 'Total GPU Execution Duration (usec)')
                ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],'gpu_total_duration',
                                        xlabel="GPU Total Duration",
                                        xdelta=1,
                                        xstart=min([df.gpu_total_duration.min() for df in df_gpu_durations_dict[task]]) - 100,
                                        xstop=max([df.gpu_total_duration.max() for df in df_gpu_durations_dict[task]]) + 100)
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
                fig = whisker_comparison(df_gpu_durations_dict[task],'gpu_setup1_duration',['slot','file_index'],
                                        "GPU Execution (%s) Total Duration (Setup1)"%task, 'Total GPU Execution Duration (usec)')
                ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],'gpu_setup1_duration',
                                        xlabel="GPU Setup1 Duration",
                                        xdelta=1,
                                        xstart=min([df.gpu_setup1_duration.min() for df in df_gpu_durations_dict[task]]) - 100,
                                        xstop=max([df.gpu_setup1_duration.max() for df in df_gpu_durations_dict[task]]) + 100)
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
                fig = whisker_comparison(df_gpu_durations_dict[task],'gpu_setup2_duration',['slot','file_index'],
                                        "GPU Execution (%s) Total Duration (Setup2)"%task, 'Total GPU Execution Duration (usec)')
                ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],'gpu_setup2_duration',
                                        xlabel="GPU Setup2 Duration",
                                        xdelta=1,
                                        xstart=min([df.gpu_setup2_duration.min() for df in df_gpu_durations_dict[task]]) - 100,
                                        xstop=max([df.gpu_setup2_duration.max() for df in df_gpu_durations_dict[task]]) + 100)
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
                fig = whisker_comparison(df_gpu_durations_dict[task],'gpu_run_duration',['slot','file_index'],
                                        "GPU Execution (%s) Total Duration (Run)"%task, 'Total GPU Execution Duration (usec)')
                ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],'gpu_run_duration',
                                        xlabel="GPU Run Duration",
                                        xdelta=1,
                                        xstart=min([df.gpu_run_duration.min() for df in df_gpu_durations_dict[task]]) - 100,
                                        xstop=max([df.gpu_run_duration.max() for df in df_gpu_durations_dict[task]]) + 100)
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

                if(task in ['PUSCH Aggr']):
                    field_name = 'gpu_run_eh_duration'
                    title = "GPU EH Run (%s) Total Duration (Run)"%task
                    axis_title = "GPU Execution Duration (usec)"
                    fig = whisker_comparison(df_gpu_durations_dict[task],field_name,['slot','file_index'],
                                             title, axis_title)
                    ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],field_name,
                                               xlabel=axis_title,
                                               xdelta=1,
                                               xstart=min([df[field_name].min() for df in df_gpu_durations_dict[task]]) - 100,
                                               xstop=max([df[field_name].max() for df in df_gpu_durations_dict[task]]) + 100)
                    fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

                    field_name = 'gpu_run_noneh_duration'
                    title = "GPU non-EH Run (%s) Total Duration (Run)"%task
                    axis_title = "GPU Execution Duration (usec)"
                    fig = whisker_comparison(df_gpu_durations_dict[task],field_name,['slot','file_index'],
                                             title, axis_title)
                    ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],field_name,
                                               xlabel=axis_title,
                                               xdelta=1,
                                               xstart=min([df[field_name].min() for df in df_gpu_durations_dict[task]]) - 100,
                                               xstop=max([df[field_name].max() for df in df_gpu_durations_dict[task]]) + 100)
                    fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

                    field_name = 'gpu_run_copy_duration'
                    title = "GPU Copy (%s) Total Duration (Run)"%task
                    axis_title = "GPU Execution Duration (usec)"
                    fig = whisker_comparison(df_gpu_durations_dict[task],field_name,['slot','file_index'],
                                             title, axis_title)
                    ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],field_name,
                                               xlabel=axis_title,
                                               xdelta=1,
                                               xstart=min([df[field_name].min() for df in df_gpu_durations_dict[task]]) - 100,
                                               xstop=max([df[field_name].max() for df in df_gpu_durations_dict[task]]) + 100)
                    fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

                    field_name = 'gpu_run_gap_duration'
                    title = "GPU Gap (%s) Total Duration (Run)"%task
                    axis_title = "GPU Execution Duration (usec)"
                    fig = whisker_comparison(df_gpu_durations_dict[task],field_name,['slot','file_index'],
                                             title, axis_title)
                    ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],field_name,
                                               xlabel=axis_title,
                                               xdelta=1,
                                               xstart=min([df[field_name].min() for df in df_gpu_durations_dict[task]]) - 100,
                                               xstop=max([df[field_name].max() for df in df_gpu_durations_dict[task]]) + 100)
                    fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

                if(task in ['PDSCH Aggr']):
                    # Check if pdsch_h2d_copy_duration field exists and has non-zero values
                    if len(df_gpu_durations_dict[task]) > 0 and len(df_gpu_durations_dict[task][0]) > 0:
                        if 'pdsch_h2d_copy_duration' in df_gpu_durations_dict[task][0].columns:
                            # Check if any dataframe has non-zero values
                            has_nonzero = any([df['pdsch_h2d_copy_duration'].max() > 0 for df in df_gpu_durations_dict[task] if len(df) > 0])
                            if has_nonzero:
                                field_name = 'pdsch_h2d_copy_duration'
                                title = "PDSCH H2D Copy (%s) Duration"%task
                                axis_title = "GPU Execution Duration (usec)"
                                fig = whisker_comparison(df_gpu_durations_dict[task],field_name,['slot','file_index'],
                                                         title, axis_title)
                                ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],field_name,
                                                           xlabel=axis_title,
                                                           xdelta=1,
                                                           xstart=min([df[field_name].min() for df in df_gpu_durations_dict[task] if len(df) > 0]) - 100,
                                                           xstop=max([df[field_name].max() for df in df_gpu_durations_dict[task] if len(df) > 0]) + 100)
                                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

                if(enable_gpu_prep_breakdown):
                    fig = whisker_comparison(df_gpu_durations_dict[task],'cpu_setup_duration',['slot','file_index'],
                                        "CPU Setup (%s) Total Duration"%task, 'Total CPU Setup Duration (usec)')
                    ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],'cpu_setup_duration',
                                            xlabel="CPU Setup Duration",
                                            xdelta=1,
                                            xstart=min([df.cpu_setup_duration.min() for df in df_gpu_durations_dict[task]]) - 100,
                                            xstop=max([df.cpu_setup_duration.max() for df in df_gpu_durations_dict[task]]) + 100)
                    fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
                    fig = whisker_comparison(df_gpu_durations_dict[task],'cpu_run_duration',['slot','file_index'],
                                        "CPU Run (%s) Total Duration"%task, 'Total CPU Run Duration (usec)')
                    ccdf_fig = ccdf_comparison(df_gpu_durations_dict[task],'cpu_run_duration',
                                            xlabel="CPU Run Duration",
                                            xdelta=1,
                                            xstart=min([df.cpu_run_duration.min() for df in df_gpu_durations_dict[task]]) - 100,
                                            xstop=max([df.cpu_run_duration.max() for df in df_gpu_durations_dict[task]]) + 100)
                    fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))

    print("Generating CPU subtask plots...")
    # Add CPU Task start/completion/durations
    if(enable_subtask_breakdown):
        cpu_tasks = list(df_ti_full_task_dict.keys())
        cpu_tasks.sort()
        for task in cpu_tasks:
            max_entries = max([len(df) for df in df_ti_full_task_dict[task]])
            if(max_entries > 0):
                fig = whisker_comparison(df_ti_full_task_dict[task],'start_deadline',['slot','file_index'],
                                        "CPU Task (%s) Start Time"%task, 'CPU Task Start Deadline Time (usec)')
                ccdf_fig = ccdf_comparison(df_ti_full_task_dict[task],'start_deadline',
                                        xlabel=deadline_label,
                                        xdelta=1,
                                        xstart=min([df.start_deadline.min() for df in df_ti_full_task_dict[task]]) - 100,
                                        xstop=max([df.start_deadline.max() for df in df_ti_full_task_dict[task]]) + 100)
                ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
                ccdf_fig.xaxis.axis_label = 'CPU Task Start Deadline Time (usec)'
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
                fig = whisker_comparison(df_ti_full_task_dict[task],'end_deadline',['slot','file_index'],
                                        "CPU Task (%s) Completion Time"%task, 'CPU Task Complete Deadline Time (usec)')
                ccdf_fig = ccdf_comparison(df_ti_full_task_dict[task],'end_deadline',
                                        xlabel=deadline_label,
                                        xdelta=1,
                                        xstart=min([df.end_deadline.min() for df in df_ti_full_task_dict[task]]) - 100,
                                        xstop=max([df.end_deadline.max() for df in df_ti_full_task_dict[task]]) + 100)
                ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
                ccdf_fig.xaxis.axis_label = 'CPU Task Completion Deadline Time (usec)'
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))
                fig = whisker_comparison(df_ti_full_task_dict[task],'duration',['slot','file_index'],
                                        "CPU Task (%s) Total Duration"%task, 'CPU Task Duration (usec)')
                ccdf_fig = ccdf_comparison(df_ti_full_task_dict[task],'duration',
                                        xlabel="CPU Duration",
                                        xdelta=1,
                                        xstart=min([df.duration.min() for df in df_ti_full_task_dict[task]]) - 100,
                                        xstop=max([df.duration.max() for df in df_ti_full_task_dict[task]]) + 100)
                ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
                ccdf_fig.xaxis.axis_label = 'CPU Task Duration (usec)'
                fig_list.append(row(column(legend_fig,fig),column(legend_fig,ccdf_fig)))


                fig1 = whisker_comparison(df_ti_all_subtasks_dict[task],'duration',['unique_label','slot','file_index'],
                                        "Subtask Duration Breakdown (%s)"%task, 'CPU Task Duration (usec)')
                
                fig2 = whisker_comparison(df_ti_all_subtasks_dict[task],'start_deadline',['unique_label','slot','file_index'],
                                        "Subtask Start Deadline Breakdown (%s)"%task, 'CPU Task Start Time (Deadline )')
                
                fig3 = whisker_comparison(df_ti_all_subtasks_dict[task],'end_deadline',['unique_label','slot','file_index'],
                                        "Subtask End Deadline Breakdown (%s)"%task, 'CPU Task End Time (Deadline )')

                ccdf_subtask_list = ['Signal Slot Channel End','Wait Slot Channel End']
                ccdf_fig_list = []
                for current_subtask in ccdf_subtask_list:
                    if(len(df_ti_all_subtasks_dict[task]) > 0 and current_subtask in list(set(df_ti_all_subtasks_dict[task][0].subtask))):
                        current_data = [df[df.subtask==current_subtask] for df in df_ti_all_subtasks_dict[task]]
                        ccdf_fig = ccdf_comparison(current_data,'end_deadline',title="CCDF (%s: %s)"%(task,current_subtask),
                                                xlabel=deadline_label,
                                                xdelta=1,
                                                xstart=min([df.end_deadline.min() for df in current_data]) - 100,
                                                xstop=max([df.end_deadline.max() for df in current_data]) + 100)
                        ccdf_fig.yaxis.axis_label = '(1-CDF) Probability'
                        ccdf_fig.xaxis.axis_label = '%s: %s'%(task,current_subtask)
                        ccdf_fig_list.append(ccdf_fig)

                if(len(ccdf_fig_list) > 0):
                    fig_list.append(row(fig1,row(ccdf_fig_list)))
                else:
                    fig_list.append(fig1)

                fig_list.append(fig2)
                fig_list.append(fig3)

    return column(fig_list)
    


if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0
    default_num_proc=8

    parser = argparse.ArgumentParser(
        description="Produces several box-whisker/CCDF plots based on PerfMetrics parsed format"
    )
    parser.add_argument(
        "input_data_list", type=str, nargs="+", help="List of input folders or rotating list phy/testmac/ru logs (space separated)"
    )
    parser.add_argument(
        "-l", "--labels", type=str, nargs="+", help="Names for each phy/ru pair"
    )
    parser.add_argument(
        "-o", "--out_filename", help="Filename for the output"
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
        "-s", "--slot_selection", help="List of comma separate ranges for slot selection. Example \"1-4,7,9-11\" --> [1,2,3,4,7,9,10,11]"
    )
    parser.add_argument(
        "-a", "--enable_subtask_breakdown", action="store_true", help="Enables task/subtask breakdown for all UL/DL tasks"
    )
    parser.add_argument(
        "-b", "--enable_gpu_prep_breakdown", action="store_true", help="Enables CPU durations involved with setting up GPU"
    )
    parser.add_argument(
        "-c", "--enable_ulc_dlc_breakouts", action="store_true", help="Enables per-task breakdowns for ULC/DLC"
    )
    parser.add_argument(
        "-e", "--mmimo_enable", action="store_true", help="Modifies timeline requirement to mmimo settings"
    )
    args = parser.parse_args()

    main(args)
