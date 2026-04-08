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

from bokeh.plotting import save, show, figure
from bokeh.io import output_file
from bokeh.models import Range1d, ColumnDataSource, HoverTool, Div, FactorRange
from bokeh.layouts import row, column, layout

#All logic for threshold generation here
def gen_threshold(mean_stats_df):
    MIN_INCREASE_USEC = 1.0
    PERCENT_INCREASE = 5.0

    thresh_df = mean_stats_df.copy()
    for col in [aa for aa in thresh_df.columns if aa not in ['slot','symbol']]:
        #Apply minimum increase or percentage
        thresh_df[col] = np.maximum(thresh_df[col] + MIN_INCREASE_USEC, thresh_df[col] * ((100+PERCENT_INCREASE)/100.0))

        #Round up to nearest 0.1usec
        thresh_df[col] = np.ceil(thresh_df[col]*10.0)/10.0

    return thresh_df

#All logic for threshold application here
def execute_threshold(mean_stats_df,thresh_df):
    assert(all(mean_stats_df.columns == thresh_df.columns))
    assert(len(mean_stats_df) == len(thresh_df))

    col_list = [aa for aa in list(mean_stats_df.columns) if aa not in ['slot','symbol']]

    success = True
    for col in col_list:
        fail_info_list = []
        for ii in range(len(mean_stats_df)):
            slot = mean_stats_df.iloc[ii].slot
            symbol = mean_stats_df.iloc[ii].symbol
            val1 = mean_stats_df.iloc[ii][col]
            val2 = thresh_df.iloc[ii][col]

            if val2 is not np.nan and val1 >= val2:
                success = False
                fail_info_list.append({'slot': slot,
                                       'symbol': symbol,
                                       'val1': val1,
                                       'val2': val2})
                
        if len(fail_info_list) > 0:
            print("Thresholding for %s failed, failed slots = %s"%(col,list(set([aa['slot'] for aa in fail_info_list]))))
            for ff in fail_info_list:
                print("slot %i sym %i: failed by %fus"%(ff['slot'],ff['symbol'],ff['val1']-ff['val2']))
        else:
            print("Thresholding for %s succeeded"%col)
        
                    
    return success

#All logic for summary print
def summary_print(mean_stats_df):
    print("")
    print("")

#Function that takes in DU/RU timing information (lists, as they can be more than one result) and extracts a single
# dataframe representing the average of the Xth quantile (default 99.5%)
def calculate_average_quantiles(df_du_rx_timing_list,df_ru_rx_timing_list,stats_percentage=99.5):
    stats_list = []
    for ii,aa in enumerate(zip(df_du_rx_timing_list,df_ru_rx_timing_list)):
        du_data = aa[0]
        ru_data = aa[1]
        print("Processing file %i"%ii)

        final_merged_df = None
        for channel_data,channel_name in zip([du_data,
                                              ru_data[ru_data.task=="UL C Plane"],
                                              ru_data[ru_data.task=="DL C Plane"],
                                              ru_data[ru_data.task=="DL U Plane"]],
                                              ["ulu","ulc","dlc","dlu"]):

            #Generate statistics at slot/symbol level
            print("Generating stats...")
            temp_gb = channel_data.groupby(['slot','symbol'])
            start_stats = temp_gb.start_offset_min.quantile(stats_percentage/100.0).reset_index().rename(columns={"start_offset_min":channel_name+"_"+"rx_start_q%04.1f"%(stats_percentage)})
            end_stats = temp_gb.end_offset_max.quantile(stats_percentage/100.0).reset_index().rename(columns={"end_offset_max":channel_name+"_"+"rx_end_q%04.1f"%(stats_percentage)})
            merged_df = pd.merge(start_stats,end_stats,how="outer",on=["slot","symbol"],validate="one_to_one")

            #Merge results together
            print("Merge results...")
            if final_merged_df is None:
                final_merged_df = merged_df
            else:
                final_merged_df = pd.merge(final_merged_df,merged_df,how="outer",on=["slot","symbol"],validate="one_to_one")

        print("Finalizing output...")
        final_merged_df = final_merged_df.sort_values(['slot','symbol'])
        stats_list.append(final_merged_df)

    #Find mean across all results
    print("Concatenating results...")
    all_stats = pd.concat(stats_list)
    print("Calculating average...")
    mean_stats_df = all_stats.groupby(['slot','symbol']).mean().reset_index()
    summary_print(mean_stats_df)

    return mean_stats_df

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

    mean_stats_df = calculate_average_quantiles(df_du_rx_timing_list,df_ru_rx_timing_list,stats_percentage=99.5)

    #Write out statistics if desired
    if args.stats_csv is not None:
        print("Writing stats_csv %s..."%args.stats_csv)
        mean_stats_df.to_csv(args.stats_csv,float_format="%12.9f",index=False,na_rep="        None")

    #Generate output threshold file is desired
    if args.output_threshold_csv is not None:
        print("Generating threshold csv to %s..."%args.output_threshold_csv)
        threshold_df = gen_threshold(mean_stats_df)
        threshold_df.to_csv(args.output_threshold_csv,index=False)

    #Perform thresholding if desired
    if args.input_threshold_csv is not None:
        print("Performing thresholding using %s..."%args.input_threshold_csv)
        threshold_df = pd.read_csv(args.input_threshold_csv,dtype=np.float64,na_values="        None")
        success = execute_threshold(mean_stats_df,threshold_df)
        if not success:
            print("FAILED :: Input failed to meet thresholds")
            return 1
        else:
            print("SUCCESS :: All results met input thresholds across all slot/symbol pairs")
            return 0

    
    return 0


if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0
    default_num_proc=8

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
        "-t", "--input_threshold_csv", help="If specified enables thresholding, returns 0 on success, 1 on failure"
    )
    parser.add_argument(
        "-g", "--output_threshold_csv", help="If specified script will generate a threshold file to output_threshold_csv based on input_csvs"
    )
    parser.add_argument(
        "-s", "--stats_csv", help="If specified script will generate a stats file to stats_csv based on mean results from input data"
    )
    args = parser.parse_args()

    main(args)
