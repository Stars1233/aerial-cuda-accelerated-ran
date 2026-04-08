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
from aerial_postproc.logparse import parse_phylog_ti_gpu_l2, GH_4TR_DLU_WINDOW_OFFSET1, parse_phylog_tick_times
from aerial_postproc.logplot import comparison_legend, ccdf_breakout, bar_plot

import argparse
import numpy as np
import os
import pandas as pd
import time
import copy

from bokeh.plotting import save, show, figure
from bokeh.io import output_file
from bokeh.models import Range1d, ColumnDataSource, HoverTool, Div, FactorRange
from bokeh.layouts import row, column, layout

from fhgen_threshold import calculate_average_quantiles

from aerial_postproc.logplot import whisker_comparison, ccdf_comparison, comparison_legend

def main(args):
    
    #Prepare input_data for PerfMetricsIO
    from aerial_postproc.parsenator_formats import input_data_parser
    input_data_list = input_data_parser(args.input_data_list)
    if(len(input_data_list) == 0):
        print("ERROR :: Unable to parse input data %s.  Must be preparsed folders or phy/testmac/ru alternating logs"%args.input_data_list)
        return

    #Parse or load all data
    df_du_rx_timing_list = []
    df_ru_rx_timing_list = []
    for input_data in input_data_list:
        from aerial_postproc.parsenator_formats import LatencySummaryIO
        io_class = LatencySummaryIO()
        print("Running parse... %s %s %s %s"%(input_data,
                                              args.ignore_duration,
                                              args.max_duration,
                                              args.num_proc))
        parse_success = io_class.parse(input_data,args.ignore_duration,args.max_duration,slot_list=None,num_proc=args.num_proc)
        if(parse_success):
            #Only pluck off data that we need
            df_du_rx_timing_list.append(io_class.data['df_du_rx_timing'])
            df_ru_rx_timing_list.append(io_class.data['df_ru_rx_timing'])
        else:
            print("Error:: Invalid input_data %s"%input_data)

        #Remove all of the allocated data that we do not need
        del io_class

    #Summarize data across all cells
    print("Summarizing data across all cells...")
    for ii in range(len(df_du_rx_timing_list)):
        #For UL data, all data is ULU (no task field)
        temp_gb = df_du_rx_timing_list[ii].groupby(['t0_timestamp','symbol'])
        temp_df = df_du_rx_timing_list[ii].assign(start_offset_min=temp_gb.start_offset.transform(min),
                                                  end_offset_max=temp_gb.end_offset.transform(max))
        temp_df = temp_df.assign(duration=temp_df.end_offset_max - temp_df.start_offset_min)
        #Downselect to only fields we care about
        df_du_rx_timing_list[ii] = temp_df[temp_df.cell==0][['t0_timestamp','sfn','slot','symbol','start_offset_min','end_offset_max','duration']]
    for ii in range(len(df_ru_rx_timing_list)):
        #For DL data, make sure we separate tasks
        temp_gb = df_ru_rx_timing_list[ii].groupby(['t0_timestamp','symbol','task'])
        temp_df = df_ru_rx_timing_list[ii].assign(start_offset_min=temp_gb.start_offset.transform(min),
                                                  end_offset_max=temp_gb.end_offset.transform(max))
        temp_df = temp_df.assign(duration=temp_df.end_offset_max - temp_df.start_offset_min)
        #Downselect to only fields we care about
        df_ru_rx_timing_list[ii] = temp_df[temp_df.cell==0][['t0_timestamp','sfn','slot','symbol','start_offset_min','end_offset_max','duration','task']]


    #Initialize to generic file index if not specified
    labels = args.labels
    if not labels:
        labels = ["Unlabeled File %02i"%ii for ii in range(len(df_du_rx_timing_list))]
    while(len(labels) < len(df_du_rx_timing_list)):
        labels.append("Unlabeled File %02i"%(len(labels)-1))

    #Determine worst slots
    print("Using first file to determine worst symbols...")
    mean_df_list = []
    for ii in range(len(df_du_rx_timing_list)):
        mean_df_list.append(calculate_average_quantiles([df_du_rx_timing_list[ii]],[df_ru_rx_timing_list[ii]],stats_percentage=99.5))

    print("Finding worst slots based on first file...")
    num_worst_slots = args.num_worst_slots
    worst_slot_sym_list = {}
    for channel_name in ["ulu","ulc","dlc","dlu"]:
        temp_df = mean_df_list[0].sort_values(by=['%s_rx_end_q99.5'%channel_name],ascending=False)
        worst_slot_sym_list[channel_name] = [(aa,bb) for aa,bb in zip(temp_df[0:num_worst_slots].slot,temp_df[0:num_worst_slots].symbol)]

    #Given a single input returns a list of dataframes that summarizes over all cells and separates for each slot/sym pair
    def summarize_data(input_data, slot_sym_tuple_list):

        #Filter based on slot_sym_tuple
        out_data = []
        for slot_sym_tuple in slot_sym_tuple_list:
            out_data.append(input_data[[aa==slot_sym_tuple for aa in input_data[['slot','symbol']].itertuples(index=False,name=None)]])

        return out_data

    print("Filtering worst slot data...")
    #Filter based on worst case slots
    all_ulu_data = df_du_rx_timing_list
    all_ulc_data = [aa[(aa.task=="UL C Plane")] for aa in df_ru_rx_timing_list]
    all_dlc_data = [aa[(aa.task=="DL C Plane")] for aa in df_ru_rx_timing_list]
    all_dlu_data = [aa[(aa.task=="DL U Plane")] for aa in df_ru_rx_timing_list]

    ulu_data_list = [bb for aa in all_ulu_data for bb in summarize_data(aa,worst_slot_sym_list["ulu"])]
    ulc_data_list = [bb for aa in all_ulc_data for bb in summarize_data(aa,worst_slot_sym_list["ulc"])]
    dlc_data_list = [bb for aa in all_dlc_data for bb in summarize_data(aa,worst_slot_sym_list["dlc"])]
    dlu_data_list = [bb for aa in all_dlu_data for bb in summarize_data(aa,worst_slot_sym_list["dlu"])]

    #Create CCDF for start
    print("Generating plots...")
    col_list = []

    if(args.input_threshold_csv):
        threshold_df = pd.read_csv(args.input_threshold_csv,dtype=np.float64,na_values="        None")

    #Create summary over all slots
    common_x_range = None
    common_y_range = Range1d(0,0)
    for channel_name in ["ulu","dlu","dlc","ulc"]:
        row_list = []
        for mean_df,label in zip(mean_df_list,labels):

            #Bar plot for start
            fig_bar1 = bar_plot(mean_df,["slot","symbol"],"%s_rx_start_q99.5"%channel_name,yfield_name="Latency (usec)",
                                title='%s Start 99.5%% (%s)'%(channel_name.upper(),label),xlabel="slot,symbol",ylabel='Latency (usec)',
                                x_range=common_x_range,y_range=common_y_range)

            if(args.input_threshold_csv):
                xx = list(threshold_df[["slot","symbol"]].astype('int').astype('str').itertuples(index=False, name=None))
                yy1 = list(mean_df["%s_rx_start_q99.5"%channel_name])
                yy2 = list(threshold_df["%s_rx_start_q99.5"%channel_name])
                color = ['green' if yy2[ii] > yy1[ii] else 'red' for ii in range(len(yy1))]
                source = ColumnDataSource(data=dict(xx=xx, yy1=yy1, yy2=yy2, color=color))
                fig_bar1.vbar(x='xx', bottom='yy1', top='yy2', width=0.9, source=source, color='color')

            row_list.append(fig_bar1)
            
            #Update range if not yet set
            if(common_x_range is None):
                common_x_range = fig_bar1.x_range

            #Bar plot for end
            fig_bar2 = bar_plot(mean_df,["slot","symbol"],"%s_rx_end_q99.5"%channel_name,yfield_name="Latency (usec)",
                                title='%s End 99.5%% (%s)'%(channel_name.upper(),label),xlabel="slot,symbol",ylabel='Latency (usec)',
                                x_range=common_x_range,y_range=common_y_range)
            
            if(args.input_threshold_csv):
                xx = list(threshold_df[["slot","symbol"]].astype('int').astype('str').itertuples(index=False, name=None))
                yy1 = list(mean_df["%s_rx_end_q99.5"%channel_name])
                yy2 = list(threshold_df["%s_rx_end_q99.5"%channel_name])
                color = ['green' if yy2[ii] > yy1[ii] else 'red' for ii in range(len(yy1))]
                source = ColumnDataSource(data=dict(xx=xx, yy1=yy1, yy2=yy2, color=color))
                fig_bar2.vbar(x='xx', bottom='yy1', top='yy2', width=0.9, source=source, color='color')

            row_list.append(fig_bar2)

            #Drive up the common y range as far as it needs to go
            common_y_range.end = max(common_y_range.end,mean_df["%s_rx_start_q99.5"%channel_name].max()*1.1)
            common_y_range.end = max(common_y_range.end,mean_df["%s_rx_end_q99.5"%channel_name].max()*1.1)
        
        col_list.append(row(row_list))

    #For worst slots, produce ccdf comparisons
    for data_list,channel_name in zip([ulu_data_list,dlu_data_list,dlc_data_list,ulc_data_list],["ULU","DLU","DLC","ULC"]):
        legend_fig = comparison_legend([aa + " (slot=%2i,sym=%2i)"%(bb[0],bb[1]) for aa in labels for bb in worst_slot_sym_list["%s"%channel_name.lower()]])
        ccdf1_fig = ccdf_comparison(data_list,'start_offset_min',title="%s Start"%channel_name,
                                    xstart=min([aa.start_offset_min.min() for aa in data_list])-1,
                                    xstop=max([aa.start_offset_min.max() for aa in data_list])+1,
                                    xdelta=0.01)
        ccdf2_fig = ccdf_comparison(data_list,'end_offset_max',title="%s End"%channel_name,
                                    xstart=min([aa.end_offset_max.min() for aa in data_list])-1,
                                    xstop=max([aa.end_offset_max.max() for aa in data_list])+1,
                                    xdelta=0.01)
        ccdf3_fig = ccdf_comparison(data_list,'duration',title="%s Duration"%channel_name,
                                    xstart=min([aa.duration.min() for aa in data_list])-1,
                                    xstop=max([aa.duration.max() for aa in data_list])+1,
                                    xdelta=0.01)
        col_list.append(row(ccdf1_fig,ccdf2_fig,ccdf3_fig,legend_fig))
    

    time3 = time.time()

    #Output plots
    if(args.out_filename):
        out_filename = args.out_filename
    else:
        out_filename = "result.html"
    output_file(filename=out_filename, title=os.path.split(__file__)[1])
    print("Writing output to %s"%out_filename)
    if(args.out_filename):
        save(column(col_list))
        time4 = time.time()
        print("Saving complete. Saving duration: %.1f"%(time4-time3))
    else:
        show(column(col_list))

    
    return 0


if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0
    default_num_proc=8
    default_num_worst_slots=1

    parser = argparse.ArgumentParser(
        description="Produces several box-whisker/CCDF plots to analyze instrumented PHY/RU logs"
    )
    parser.add_argument(
        "input_data_list", type=str, nargs="+", help="List of input folders or rotating list phy/testmac/ru logs (space separated)"
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
        "-w", "--num_worst_slots", default=default_num_worst_slots, type=int, help="Number of worst slots to put in CCDF (default=%i)"%default_num_worst_slots
    )
    parser.add_argument(
        "-t", "--input_threshold_csv", help="If specified enables thresholding, returns 0 on success, 1 on failure"
    )
    parser.add_argument(
        "-l", "--labels", type=str, nargs="+", help="Name for each data set"
    )
    parser.add_argument(
        "-o", "--out_filename", help="Filename for the output"
    )
    args = parser.parse_args()

    main(args)
