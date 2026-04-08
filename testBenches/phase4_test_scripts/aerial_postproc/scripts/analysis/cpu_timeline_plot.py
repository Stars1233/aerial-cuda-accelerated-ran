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
from aerial_postproc.logplot import plot_all_tasks_cpu_timeline, ti_boxplot, timeline_plot_generic
from bokeh.plotting import show
from bokeh.io import output_file
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import CrosshairTool
from bokeh.layouts import layout
import pandas as pd
import argparse
import numpy as np
import os
import time
from aerial_postproc.parsenator_formats import parse_all_PerfMetricsIO, parse_labels

#Adds start/end datetime using t0_timestamp, start_deadline, and end_deadline
def add_datetimes(io_df):
    start_timestamp = io_df['t0_timestamp'] + (io_df['start_deadline']*1000).astype(int)
    end_timestamp = io_df['t0_timestamp'] + (io_df['end_deadline']*1000).astype(int)
    io_df['start_datetime'] = pd.to_datetime(start_timestamp,unit='ns')
    io_df['end_datetime'] = pd.to_datetime(end_timestamp,unit='ns')

def main(args):

    print("Running "+os.path.split(__file__)[1])

    time1 = time.time()

    df_ti_list,df_l2_list,df_gpu_list,df_compression_list,df_tick_list,df_testmac_list,slot_list = parse_all_PerfMetricsIO(args.input_data_list,
                                                                                                                           args.ignore_duration,
                                                                                                                           args.max_duration,
                                                                                                                           args.num_proc,
                                                                                                                           args.slot_selection)
    labels = parse_labels(args.labels,len(df_ti_list))

    time2 = time.time()

    panel_list = []
    ii = 0
    start_tir = args.ignore_duration
    end_tir = args.max_duration
    for df_ti,df_l2,df_gpu,df_compression,df_tick,df_testmac in zip(df_ti_list,df_l2_list,df_gpu_list,df_compression_list,df_tick_list,df_testmac_list):

        # Get tab name
        tab_name = labels[ii]
        ii += 1

        if(len(df_ti) > 0):
            df_ti_subtask = df_ti[df_ti.subtask!='Full Task']
            df_ti_task = df_ti[df_ti.subtask=='Full Task']
        else:
            df_ti_subtask = []
            df_ti_task = []

        # Generate plots
        fig_list = []

        # # CPU Timeline plot (y=CPU/SUBTASK, x=time)
        if(len(df_ti_subtask) > 0):
            #Cut WORKER/L2 data
            timeline_max_dur = 2.0
            data1 = df_ti_subtask[df_ti_subtask.tir < (start_tir + timeline_max_dur)].copy()
            data2 = df_l2[df_l2.tir < (start_tir + timeline_max_dur)].copy()
            data3 = df_tick
            data4 = df_testmac

            ENABLE_TICK = len(df_tick) > 0
            ENABLE_TESTMAC = len(df_testmac) > 0

            
            

            #Create CPU plot
            add_datetimes(data1)
            add_datetimes(data2)

            if(ENABLE_TICK):
                data3['cpu_datetime'] = pd.to_datetime(data3['cpu_timestamp'],unit='ns')

            if(ENABLE_TESTMAC):
                data4['start_datetime'] = pd.to_datetime(data4['fapi2_start_timestamp'],unit='ns')
                data4['end_datetime'] = pd.to_datetime(data4['fapi2_stop_timestamp'],unit='ns')
                
            crosshair_tool = CrosshairTool()
            fig1 = plot_all_tasks_cpu_timeline(data1, data2, data3, data4, x_range=None, traces_only=not args.enable_cpu_percentages, cpu_only=False, crosshair_tool=crosshair_tool,ul_only=args.ul_only)

            # Set fixed y-axis properties for the first plot in the column
            plot1 = fig1.children[0]  # Get the actual plot from the column layout
            plot1.yaxis.axis_label_standoff = 50  # Fixed standoff distance
            plot1.min_border_left = 100  # Fixed left border width
            plot1.sizing_mode = "stretch_width"

            #Cut/merge GPU UL data
            data3 = df_gpu[df_gpu.tir < (start_tir + timeline_max_dur)].copy()

            #Create GPU plot
            # pusch_start_timeline = pd.merge(data3[data3.task=='PUSCH Aggr']['t0_timestamp','gpu_run_duration'],
            #                                 data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Started PUSCH')]['t0_timestamp','start_timestamp'],
            #                                 validate='one_to_one',on='t0_timestamp')
            # pusch_stop_timeline = pd.merge(data3[data3.task=='PUSCH Aggr']['t0_timestamp','gpu_run_duration'],
            #                                 data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Done PUSCH')]['t0_timestamp','start_timestamp'],
            #                                 validate='one_to_one',on='t0_timestamp')
            # pucch_start_timeline = pd.merge(data3[data3.task=='PUCCH Aggr']['t0_timestamp','gpu_run_duration'],
            #                                 data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Started PUCCH')]['t0_timestamp','start_timestamp'],
            #                                 validate='one_to_one',on='t0_timestamp')
            # pucch_stop_timeline = pd.merge(data3[data3.task=='PUCCH Aggr']['t0_timestamp','gpu_run_duration'],
            #                                data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Done PUCCH')]['t0_timestamp','start_timestamp'],
            #                                validate='one_to_one',on='t0_timestamp')

            compression_df = pd.merge(data1[(data1.task=='Debug Task') & (data1.subtask=='Compression Start Wait')][['t0_timestamp','end_deadline','slot','duration']],
                             data1[(data1.task=='Debug Task') & (data1.subtask=='Compression Wait')][['t0_timestamp','end_deadline']],
                             validate='one_to_one',on='t0_timestamp').rename(columns={'end_deadline_x':'start_deadline','end_deadline_y':'end_deadline'})
            compression_df['Kernel Type'] = 'COMPRESSION'
            add_datetimes(compression_df)

            pusch_df = pd.merge(data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Started PUSCH')][['t0_timestamp','start_deadline','slot','duration']],
                             data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Validate PUSCH')][['t0_timestamp','start_deadline']],
                             validate='one_to_one',on='t0_timestamp').rename(columns={'start_deadline_x':'start_deadline','start_deadline_y':'end_deadline'})
            pusch_df['Kernel Type'] = 'PUSCH'+ " " + (pusch_df['slot']%10).astype(str)
            add_datetimes(pusch_df)
            
            pucch_df = pd.merge(data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Started PUCCH')][['t0_timestamp','start_deadline','slot','duration']],
                             data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Validate PUCCH')][['t0_timestamp','start_deadline']],
                             validate='one_to_one',on='t0_timestamp').rename(columns={'start_deadline_x':'start_deadline','start_deadline_y':'end_deadline'})
            pucch_df['Kernel Type'] = 'PUCCH'+ " " + (pucch_df['slot']%10).astype(str)
            add_datetimes(pucch_df)

            prach_df = pd.merge(data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Started PRACH')][['t0_timestamp','start_deadline','slot','duration']],
                             data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Validate PRACH')][['t0_timestamp','start_deadline']],
                             validate='one_to_one',on='t0_timestamp').rename(columns={'start_deadline_x':'start_deadline','start_deadline_y':'end_deadline'})
            prach_df['Kernel Type'] = 'PRACH'+ " " + (prach_df['slot']%10).astype(str)
            add_datetimes(prach_df)

            order_df = pd.merge(data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Start Task')][['t0_timestamp','start_deadline','slot','duration']],
                                data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Wait Order')][['t0_timestamp','start_deadline']],
                                validate='one_to_one',on='t0_timestamp').rename(columns={'start_deadline_x':'start_deadline','start_deadline_y':'end_deadline'})
            order_df['Kernel Type'] = 'ORDER'+ " " + (order_df['slot']%10).astype(str)
            add_datetimes(order_df)

            srs_order_df = pd.merge(data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Start Task')][['t0_timestamp','start_deadline','slot','duration']],
                                    data1[(data1.task=='UL Task UL AGGR 3') & (data1.subtask=='Wait SRS Order')][['t0_timestamp','start_deadline']],
                                    validate='one_to_one',on='t0_timestamp').rename(columns={'start_deadline_x':'start_deadline','start_deadline_y':'end_deadline'})
            srs_order_df['Kernel Type'] = 'SRS ORDER'+ " " + (srs_order_df['slot']%10).astype(str)
            add_datetimes(srs_order_df)
            
            print("Generating GPU timeline view...")
            data3 = pd.concat([compression_df,pusch_df,pucch_df,prach_df,order_df,srs_order_df])
            fig2 = timeline_plot_generic(data3,'Kernel Type',title="GPU Timeline View",height=600,width=1600,x_range=fig1.children[0].x_range,
                                         tooltips=[('slot','@slot'),('duration','@duration'),('start_deadline','@start_deadline'),('end_deadline','@end_deadline')],
                                         crosshair_tool=crosshair_tool)
            
            # Set matching y-axis properties for fig2
            fig2.yaxis.axis_label_standoff = 50  # Same standoff as fig1
            fig2.min_border_left = 100  # Same left border width as fig1
            fig2.sizing_mode = "stretch_width"

            fig_list.append(fig1)
            fig_list.append(fig2)

        # Add all of these figures to a panel
        print("Adding panel...")
        panel_list.append(Panel(child=layout(fig_list), title=tab_name))

    # Add all of these panels to the Tabs object
    print("Creating tabs...")
    tabs = Tabs(tabs=panel_list)
    print("Writing output to %s..."%args.out_filename)
    output_file(filename=args.out_filename, title=os.path.split(__file__)[1])
    show(tabs)


if __name__ == "__main__":
    default_ignore_duration = 0.0
    default_max_duration = 999999999.0
    parser = argparse.ArgumentParser(
        description="Generates CPU/GPU timeline view based on PerfMetrics parsed format"
    )
    parser.add_argument(
        "input_data_list", type=str, nargs="+", help="List of input folders or rotating list phy/testmac/ru logs (space separated)"
    )
    parser.add_argument(
        "-l", "--labels", type=str, nargs="+", help="Names for each phy/ru pair"
    )
    parser.add_argument(
        "-o", "--out_filename", default="result.html", help="Filename for the output"
    )
    parser.add_argument(
        "-m", "--max_duration", default=default_max_duration, type=float, help="Maximum run duration to process"
    )
    parser.add_argument(
        "-i", "--ignore_duration", default=default_ignore_duration, type=float, help="Duration at beginning of run to ignore"
    )
    parser.add_argument(
        "-n", "--num_proc", default=8, type=int, help="Number of processes to use for parsing"
    )
    parser.add_argument(
        "-s", "--slot_selection", help="List of comma separate ranges for slot selection. Example \"1-4,7,9-11\" --> [1,2,3,4,7,9,10,11]"
    )
    parser.add_argument(
        "-c", "--enable_cpu_percentages", action="store_true", help="Enables CPU percentage estimates alongside CPU traces"
    )
    parser.add_argument(
        "-u", "--ul_only", action="store_true", help="Only timeline UL data"
    )
    args = parser.parse_args()

    main(args)
