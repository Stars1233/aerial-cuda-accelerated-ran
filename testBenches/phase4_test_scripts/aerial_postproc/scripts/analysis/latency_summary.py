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
from aerial_postproc.logplot import comparison_legend, ccdf_breakout, bar_plot, ccdf_comparison
from aerial_postproc.logparse import TrafficType, getTCType, getReceptionWindow

import argparse
import numpy as np
import os
import pandas as pd

from bokeh.plotting import save, show, figure
from bokeh.io import output_file
from bokeh.models import Range1d, ColumnDataSource, HoverTool, Div, FactorRange
from bokeh.layouts import row, column, layout
from aerial_postproc.sections import SectionGenerator

#Parse a string list of integer values (all ranges inclusive)
#Example "1-4,7,9-11" --> [1,2,3,4,7,9,10,11]
def parse_range_list(range_list):
    result = []
    for part in range_list.split(","):
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start),int(end)+1))
        else:
            result.append(int(part))

    return result

def main(args):

    #Get slot list if enabled
    if(args.slot_selection):
        #Parse the slot selection values
        slot_list = parse_range_list(args.slot_selection)
    else:
        slot_list = None
    
    #Prepare input_data for PerfMetricsIO
    from aerial_postproc.parsenator_formats import input_data_parser
    input_data_list = input_data_parser(args.input_data_list)
    if(len(input_data_list) == 0):
        print("ERROR :: Unable to parse input data %s.  Must be preparsed folders or phy/testmac/ru alternating logs"%args.input_data_list)
        return
    
    #Create labels
    if(args.labels):
        labels = args.labels
    else:
        labels = ["File %02i"%ii for ii in range(len(input_data_list))]

    #Parse or load all data
    df_du_rx_timing_list = []
    df_du_rx_srs_timing_list = []
    df_ru_rx_timing_list = []
    df_tx_timing_list = []
    for input_data in input_data_list:
        from aerial_postproc.parsenator_formats import LatencySummaryIO
        io_class = LatencySummaryIO(mmimo_enable=args.mmimo_enable)
        print("Running parse... %s %s %s %s %s"%(input_data,
                                                args.ignore_duration,
                                                args.max_duration,
                                                slot_list,
                                                args.num_proc))
        parse_success = io_class.parse(input_data,args.ignore_duration,args.max_duration,slot_list=slot_list,num_proc=args.num_proc)
        if(parse_success):
            #Only pluck off data that we need
            df_du_rx_timing_list.append(io_class.data['df_du_rx_timing'])
            df_du_rx_srs_timing_list.append(io_class.data['df_du_rx_srs_timing'])
            df_ru_rx_timing_list.append(io_class.data['df_ru_rx_timing'])
            df_tx_timing_list.append(io_class.data['df_tx_timing'])
        else:
            print("Error:: Invalid input_data %s"%input_data)

        #Remove all of the allocated data that we do not need
        del io_class

    #Create plots
    fig = get_figs(df_du_rx_timing_list,df_du_rx_srs_timing_list,df_ru_rx_timing_list,df_tx_timing_list,labels,slot_list=slot_list,mmimo_enable=args.mmimo_enable,percentile=args.percentile)

    #Output plots
    if(args.out_filename):
        out_filename = args.out_filename
    else:
        out_filename = "result.html"
    output_file(filename=out_filename, title=os.path.split(__file__)[1])
    print("Writing output to %s"%out_filename)
    if(args.out_filename):
        save(fig)
    else:
        show(fig)

def get_figs(df_du_rx_timing_list,df_du_rx_srs_timing_list,df_ru_rx_timing_list,df_tx_timing_list,labels,slot_list=None,mmimo_enable=False,percentile=0.99):

    # Initialize section generator for consistent styling
    section_generator = SectionGenerator()
    
    # Initialize summary table data for cross-file analysis
    summary_table_data = []
    
    row_list = []

    #Create common x range for all slot/symbol plots
    if(not slot_list):
        slot_list = range(80)
    else:
        slot_list.sort()
    xx = [(str(slot),str(symbol)) for slot in slot_list for symbol in range(14)]
    common_x_range_symbol = FactorRange(*xx)

    try:
        xx = [(str(slot),str(symbol),str(cell)) for slot in slot_list for symbol in range(14) for cell in sorted(df_du_rx_timing_list[0].cell.unique())]
        common_x_range_symbol_cell = FactorRange(*xx)
    except:
        xx = [(str(slot),str(symbol),str(cell)) for slot in slot_list for symbol in range(14) for cell in sorted(df_ru_rx_timing_list[0].cell.unique())]
        common_x_range_symbol_cell = FactorRange(*xx)

    common_y_range_list = [Range1d(0,0) for ii in range(40)]

    HAS_DU_RX_TIMES = all([len(df_du_rx_timing) > 0 for df_du_rx_timing in df_du_rx_timing_list])
    HAS_RU_RX_TIMES = all([len(df_ru_rx_timing) > 0 for df_ru_rx_timing in df_ru_rx_timing_list])
    HAS_DU_RX_SRS_TIMES = all([len(df_du_rx_srs_timing) > 0 for df_du_rx_srs_timing in df_du_rx_srs_timing_list])
    HAS_RU_TX_TIMES = all([len(df_tx_timing) > 0 for df_tx_timing in df_tx_timing_list])
    SYMBOL_TIME = 500/14.

    if HAS_DU_RX_TIMES:
        ulu_tx_start_window = df_du_rx_timing_list[0].window_start_deadline.min()

    if HAS_DU_RX_SRS_TIMES:
        srs_tx_start_window = df_du_rx_srs_timing_list[0].window_start_deadline.min()

    #fig 1 - CCDF of DU RX
    ENABLE_DU_RX_CCDF = HAS_DU_RX_TIMES

    #fig 2 - CCDF of RU RX
    ENABLE_RU_RX_CCDF = HAS_RU_RX_TIMES

    # Initialize cross-file comparison data structures
    channel_file_data = {}  # channel -> list of (file_data, file_label)
    all_margin_values = []
    
    # Cross-file data for DU RX and RU RX timing analysis
    du_rx_timing_file_data = []  # list of (processed_df, label) for UL U Plane
    du_rx_srs_timing_file_data = []  # list of (processed_df, label) for SRS
    ru_rx_timing_file_data = {'DL U Plane': [], 'DL C Plane': [], 'UL C Plane': []}  # task -> list of (processed_df, label)

    for df_du_rx_timing,df_du_rx_srs_timing,df_ru_rx_timing,df_tx_timing,label in zip(df_du_rx_timing_list,df_du_rx_srs_timing_list,df_ru_rx_timing_list,df_tx_timing_list,labels):

        col_list = []
        current_y_index = 0

        div = Div(text="Latency summary for %s"%label,
                  style={'font-size':'150%'})
        col_list.append(div)

        if ENABLE_DU_RX_CCDF:
            # Add major subsection header for DU RX
            du_rx_header = section_generator.getSections("sub_section", "DU RX Analysis")
            col_list.append(du_rx_header)
            
            # Add sub-subsection header
            subsection_header = section_generator.getSections("sub_sub_section", "UL U Plane")
            col_list.append(subsection_header)
            
            summary_label = "%s [DU RX, UL U Plane]"%label

            #Add min start and max end to original dataframe, create a view that only contains symbol 0 points
            gb_slot = df_du_rx_timing.groupby(['t0_timestamp'])
            df_du_rx_timing = df_du_rx_timing.assign(start_min_slot=gb_slot['start_offset'].transform(min),end_max_slot=gb_slot['end_offset'].transform(max))
            gb_symbol = df_du_rx_timing.groupby(['t0_timestamp','symbol'])
            df_du_rx_timing = df_du_rx_timing.assign(start_min_symbol=gb_symbol['start_offset'].transform(min),end_max_symbol=gb_symbol['end_offset'].transform(max))
            gb_symbol = df_du_rx_timing.groupby(['t0_timestamp','symbol','cell'])
            df_du_rx_timing = df_du_rx_timing.assign(start_min_symbol_cell=gb_symbol['start_offset'].transform(min),end_max_symbol_cell=gb_symbol['end_offset'].transform(max))
            df_du_rx_timing['duration_symbol'] = df_du_rx_timing['end_max_symbol'] - df_du_rx_timing['start_min_symbol']
            df_du_rx_timing['duration_symbol_cell'] = df_du_rx_timing['end_max_symbol_cell'] - df_du_rx_timing['start_min_symbol_cell']
            
            # Collect processed data for cross-file analysis
            du_rx_timing_file_data.append((df_du_rx_timing.copy(), label))

            # Create DU RX plots
            color_list = ['green','red','lightgreen','salmon']
            fig_legend = comparison_legend(['Earliest Packet (min over slot)','Latest Packet (max over slot)',
                                            'Earliest Packet (per symbol)','Latest Packet (per symbol)'],
                                            height=200,width=400,color_list=color_list)
            fig_ccdf = ccdf_breakout(df_du_rx_timing,['start_min_slot','end_max_slot','start_offset','end_offset'],xdelta=1,xstart=-10,xstop=200,
                                     title="%s Earliest/Latest Packet"%summary_label,
                                     xlabel="Packet RX Times - Desired TX Time (T0 + %ius + sym offset)"%ulu_tx_start_window,height=800,width=800,color_list=color_list)
            col_list.append(fig_legend)
            col_list.append(fig_ccdf)

            # Create statistics on for symbol basis
            temp_gb = df_du_rx_timing.groupby(['slot','symbol'])
            temp_df = temp_gb.quantile(percentile, numeric_only=True).reset_index()
            fig_bar1 = bar_plot(temp_df,["slot","symbol"],"start_min_symbol",yfield_name="latency",
                                x_range=common_x_range_symbol,
                                y_range=common_y_range_list[current_y_index+0],
                                title='%s %.1f%% earliest per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='latency relative to window start start (usec)')
            fig_bar2 = bar_plot(temp_df,["slot","symbol"],"end_max_symbol",yfield_name="latency",
                                x_range=common_x_range_symbol,
                                y_range=common_y_range_list[current_y_index+1],
                                title='%s %.1f%% latest per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='latency relative to window start start (usec)')
            fig_bar3 = bar_plot(temp_df,["slot","symbol"],"duration_symbol",yfield_name="duration",
                                x_range=common_x_range_symbol,
                                y_range=common_y_range_list[current_y_index+2],
                                title='%s %.1f%% duration per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='duration (usec)')
            #Automatically scale the max
            common_y_range_list[current_y_index+0].end = max(common_y_range_list[current_y_index+0].end,temp_df.start_min_symbol.max()*1.1)
            common_y_range_list[current_y_index+1].end = max(common_y_range_list[current_y_index+1].end,temp_df.end_max_symbol.max()*1.1)
            common_y_range_list[current_y_index+2].end = max(common_y_range_list[current_y_index+2].end,temp_df.duration_symbol.max()*1.1)
            current_y_index += 3
            col_list.append(fig_bar1)
            col_list.append(fig_bar2)
            col_list.append(fig_bar3)

            # Create statistics on for symbol/cell basis
            temp_gb = df_du_rx_timing.groupby(['slot','symbol','cell'])
            temp_df = temp_gb.quantile(percentile, numeric_only=True).reset_index()
            fig_bar1 = bar_plot(temp_df,["slot","symbol","cell"],"start_min_symbol_cell",yfield_name="latency",
                                x_range=common_x_range_symbol_cell,
                                y_range=common_y_range_list[current_y_index+0],
                                title='%s %.1f%% earliest per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='latency relative to window start (usec)')
            fig_bar2 = bar_plot(temp_df,["slot","symbol","cell"],"end_max_symbol_cell",yfield_name="latency",
                                x_range=common_x_range_symbol_cell,
                                y_range=common_y_range_list[current_y_index+1],
                                title='%s %.1f%% latest per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='latency relative to window start (usec)')
            fig_bar3 = bar_plot(temp_df,["slot","symbol","cell"],"duration_symbol_cell",yfield_name="duration",
                                x_range=common_x_range_symbol_cell,
                                y_range=common_y_range_list[current_y_index+2],
                                title='%s %.1f%% duration per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='duration (usec)')
            #Automatically scale the max
            common_y_range_list[current_y_index+0].end = max(common_y_range_list[current_y_index+0].end,temp_df.start_min_symbol_cell.max()*1.1)
            common_y_range_list[current_y_index+1].end = max(common_y_range_list[current_y_index+1].end,temp_df.end_max_symbol_cell.max()*1.1)
            common_y_range_list[current_y_index+2].end = max(common_y_range_list[current_y_index+2].end,temp_df.duration_symbol_cell.max()*1.1)
            current_y_index += 3
            col_list.append(fig_bar1)
            col_list.append(fig_bar2)
            col_list.append(fig_bar3)

        if HAS_DU_RX_SRS_TIMES:
            # Add sub-subsection header for SRS (under DU RX Analysis)
            subsection_header = section_generator.getSections("sub_sub_section", "SRS")
            col_list.append(subsection_header)
            
            summary_label = "%s [DU RX, SRS]"%label

            #Add min start and max end to original dataframe, create a view that only contains symbol 0 points
            gb_slot = df_du_rx_srs_timing.groupby(['t0_timestamp'])
            df_du_rx_srs_timing = df_du_rx_srs_timing.assign(start_min_slot=gb_slot['start_offset'].transform(min),end_max_slot=gb_slot['end_offset'].transform(max))
            gb_symbol = df_du_rx_srs_timing.groupby(['t0_timestamp','symbol'])
            df_du_rx_srs_timing = df_du_rx_srs_timing.assign(start_min_symbol=gb_symbol['start_offset'].transform(min),end_max_symbol=gb_symbol['end_offset'].transform(max))
            gb_symbol = df_du_rx_srs_timing.groupby(['t0_timestamp','symbol','cell'])
            df_du_rx_srs_timing = df_du_rx_srs_timing.assign(start_min_symbol_cell=gb_symbol['start_offset'].transform(min),end_max_symbol_cell=gb_symbol['end_offset'].transform(max))
            df_du_rx_srs_timing['duration_symbol'] = df_du_rx_srs_timing['end_max_symbol'] - df_du_rx_srs_timing['start_min_symbol']
            df_du_rx_srs_timing['duration_symbol_cell'] = df_du_rx_srs_timing['end_max_symbol_cell'] - df_du_rx_srs_timing['start_min_symbol_cell']
            
            # Collect processed data for cross-file analysis
            du_rx_srs_timing_file_data.append((df_du_rx_srs_timing.copy(), label))

            # Create DU RX SRS plots
            color_list = ['green','red','lightgreen','salmon']
            fig_legend = comparison_legend(['Earliest Packet (min over slot)','Latest Packet (max over slot)',
                                            'Earliest Packet (per symbol)','Latest Packet (per symbol)'],
                                            height=200,width=400,color_list=color_list)
            fig_ccdf = ccdf_breakout(df_du_rx_srs_timing,['start_min_slot','end_max_slot','start_offset','end_offset'],xdelta=1,xstart=-10,xstop=200,
                                     title="%s Earliest/Latest Packet"%summary_label,
                                     xlabel="Packet RX Times - Desired TX Time (T0 + %ius + sym/eaxcid offset)"%srs_tx_start_window,height=800,width=800,color_list=color_list)
            col_list.append(fig_legend)
            col_list.append(fig_ccdf)

            # Create statistics on for symbol basis
            temp_gb = df_du_rx_srs_timing.groupby(['slot','symbol'])
            temp_df = temp_gb.quantile(percentile, numeric_only=True).reset_index()
            fig_bar1 = bar_plot(temp_df,["slot","symbol"],"start_min_symbol",yfield_name="latency",
                                x_range=common_x_range_symbol,
                                y_range=common_y_range_list[current_y_index+0],
                                title='%s %.1f%% earliest per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='latency relative to window start (usec)')
            fig_bar2 = bar_plot(temp_df,["slot","symbol"],"end_max_symbol",yfield_name="latency",
                                x_range=common_x_range_symbol,
                                y_range=common_y_range_list[current_y_index+1],
                                title='%s %.1f%% latest per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='latency relative to window start (usec)')
            fig_bar3 = bar_plot(temp_df,["slot","symbol"],"duration_symbol",yfield_name="duration",
                                x_range=common_x_range_symbol,
                                y_range=common_y_range_list[current_y_index+2],
                                title='%s %.1f%% duration per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='duration (usec)')
            #Automatically scale the max
            common_y_range_list[current_y_index+0].end = max(common_y_range_list[current_y_index+0].end,temp_df.start_min_symbol.max()*1.1)
            common_y_range_list[current_y_index+1].end = max(common_y_range_list[current_y_index+1].end,temp_df.end_max_symbol.max()*1.1)
            common_y_range_list[current_y_index+2].end = max(common_y_range_list[current_y_index+2].end,temp_df.duration_symbol.max()*1.1)
            current_y_index += 3
            col_list.append(fig_bar1)
            col_list.append(fig_bar2)
            col_list.append(fig_bar3)

            # Create statistics on for symbol/cell basis
            temp_gb = df_du_rx_srs_timing.groupby(['slot','symbol','cell'])
            temp_df = temp_gb.quantile(percentile, numeric_only=True).reset_index()
            fig_bar1 = bar_plot(temp_df,["slot","symbol","cell"],"start_min_symbol_cell",yfield_name="latency",
                                x_range=common_x_range_symbol_cell,
                                y_range=common_y_range_list[current_y_index+0],
                                title='%s %.1f%% earliest per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='latency relative to window start (usec)')
            fig_bar2 = bar_plot(temp_df,["slot","symbol","cell"],"end_max_symbol_cell",yfield_name="latency",
                                x_range=common_x_range_symbol_cell,
                                y_range=common_y_range_list[current_y_index+1],
                                title='%s %.1f%% latest per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='latency relative to window start (usec)')
            fig_bar3 = bar_plot(temp_df,["slot","symbol","cell"],"duration_symbol_cell",yfield_name="duration",
                                x_range=common_x_range_symbol_cell,
                                y_range=common_y_range_list[current_y_index+2],
                                title='%s %.1f%% duration per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='duration (usec)')
            #Automatically scale the max
            common_y_range_list[current_y_index+0].end = max(common_y_range_list[current_y_index+0].end,temp_df.start_min_symbol_cell.max()*1.1)
            common_y_range_list[current_y_index+1].end = max(common_y_range_list[current_y_index+1].end,temp_df.end_max_symbol_cell.max()*1.1)
            common_y_range_list[current_y_index+2].end = max(common_y_range_list[current_y_index+2].end,temp_df.duration_symbol_cell.max()*1.1)
            current_y_index += 3
            col_list.append(fig_bar1)
            col_list.append(fig_bar2)
            col_list.append(fig_bar3)


        if ENABLE_RU_RX_CCDF:
            # Add subsection header for RU RX analysis
            subsection_header = section_generator.getSections("sub_section", "RU RX Analysis")
            col_list.append(subsection_header)

            tc_type = getTCType(mmimo_enable)
            # Map tasks to their window offsets
            task_window_map = {
                'DL U Plane': getReceptionWindow(TrafficType.DTT_DLU, tc_type)[0],
                'DL C Plane': getReceptionWindow(TrafficType.DTT_DLC_BFW, tc_type)[0],
                'UL C Plane': getReceptionWindow(TrafficType.DTT_ULC_BFW, tc_type)[0]
            }
            
            # Maximum number of CCDF points for performance
            MAX_CCDF_POINTS = 5000
            
            for task in ['DL U Plane', 'DL C Plane', 'UL C Plane']:
                # Check if there's data for this task
                current_data = df_ru_rx_timing[df_ru_rx_timing.task==task]
                if len(current_data) == 0:
                    continue
                    
                window_offset = task_window_map[task]
                
                # Add task-specific subheader
                task_header = section_generator.getSections("sub_sub_section", task)
                col_list.append(task_header)
                
                summary_label = "%s [RU RX, %s]"%(label,task)

                #Add min start and max end to original dataframe, create a view that only contains symbol 0 points
                gb_slot = current_data.groupby(['t0_timestamp'])
                gb_symbol = current_data.groupby(['t0_timestamp','symbol'])
                gb_symbol_cell = current_data.groupby(['t0_timestamp','symbol','cell'])
                current_data = current_data.assign(start_min_slot=gb_slot['start_offset'].transform(min),end_max_slot=gb_slot['end_offset'].transform(max))
                current_data = current_data.assign(start_min_symbol=gb_symbol['start_offset'].transform(min),end_max_symbol=gb_symbol['end_offset'].transform(max))
                current_data = current_data.assign(start_min_symbol_cell=gb_symbol_cell['start_offset'].transform(min),end_max_symbol_cell=gb_symbol_cell['end_offset'].transform(max))
                current_data['duration_symbol'] = current_data['end_max_symbol'] - current_data['start_min_symbol']
                current_data['duration_symbol_cell'] = current_data['end_max_symbol_cell'] - current_data['start_min_symbol_cell']
                
                # Collect processed data for cross-file analysis
                ru_rx_timing_file_data[task].append((current_data.copy(), label))

                # Calculate data-driven CCDF range with adaptive xdelta
                data_min = min(
                    current_data['start_min_slot'].min(),
                    current_data['start_offset'].min()
                )
                data_max = max(
                    current_data['end_max_slot'].max(),
                    current_data['end_offset'].max()
                )
                ccdf_xstart = data_min - 10  # Add small margin
                ccdf_xstop = data_max + 10   # Add small margin
                
                # Adaptive xdelta: use 1us resolution unless range exceeds MAX_CCDF_POINTS
                data_range = ccdf_xstop - ccdf_xstart
                if data_range <= MAX_CCDF_POINTS:
                    ccdf_xdelta = 1
                else:
                    ccdf_xdelta = data_range / MAX_CCDF_POINTS

                # Create RU RX plots
                color_list = ['green','red','lightgreen','salmon']
                fig_legend = comparison_legend(['Earliest Packet (min over slot)','Latest Packet (max over slot)',
                                                'Earliest Packet (per symbol)','Latest Packet (per symbol)'],
                                                height=200,width=400,color_list=color_list)
                fig_ccdf = ccdf_breakout(current_data,['start_min_slot','end_max_slot','start_offset','end_offset'],xdelta=ccdf_xdelta,xstart=ccdf_xstart,xstop=ccdf_xstop,
                                        title="%s Earliest/Latest Packet"%summary_label,
                                        xlabel="Packet RX Times - Desired TX Time (T0 + %ius + sym offset)"%window_offset,height=800,width=800,color_list=color_list)
                col_list.append(fig_legend)
                col_list.append(fig_ccdf)

                # Create statistics on for symbol basis
                temp_gb = current_data.groupby(['slot','symbol'])
                temp_df = temp_gb.quantile(percentile, numeric_only=True).reset_index()
                fig_bar1 = bar_plot(temp_df,["slot","symbol"],"start_min_symbol",yfield_name="latency",
                                    x_range=common_x_range_symbol,
                                    y_range=common_y_range_list[current_y_index+0],
                                    title='%s %.1f%% earliest per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='latency relative to window start (usec)')
                fig_bar2 = bar_plot(temp_df,["slot","symbol"],"end_max_symbol",yfield_name="latency",
                                    x_range=common_x_range_symbol,
                                    y_range=common_y_range_list[current_y_index+1],
                                    title='%s %.1f%% latest per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='latency relative to window start (usec)')
                fig_bar3 = bar_plot(temp_df,["slot","symbol"],"duration_symbol",yfield_name="duration",
                                    x_range=common_x_range_symbol,
                                    y_range=common_y_range_list[current_y_index+2],
                                    title='%s %.1f%% duration per slot/symbol'%(summary_label, percentile*100),xlabel="slot,symbol",ylabel='duration (usec)')
                #Automatically scale the y-axis range (both min and max) to include negative values
                min_start_symbol = temp_df.start_min_symbol.min()
                common_y_range_list[current_y_index+0].start = min(common_y_range_list[current_y_index+0].start, min_start_symbol*1.1 if min_start_symbol < 0 else 0)
                common_y_range_list[current_y_index+0].end = max(common_y_range_list[current_y_index+0].end,temp_df.start_min_symbol.max()*1.1)
                common_y_range_list[current_y_index+1].end = max(common_y_range_list[current_y_index+1].end,temp_df.end_max_symbol.max()*1.1)
                common_y_range_list[current_y_index+2].end = max(common_y_range_list[current_y_index+2].end,temp_df.duration_symbol.max()*1.1)
                current_y_index += 3
                col_list.append(fig_bar1)
                col_list.append(fig_bar2)
                col_list.append(fig_bar3)

                # Create statistics on for symbol/cell basis
                temp_gb = current_data.groupby(['slot','symbol','cell'])
                temp_df = temp_gb.quantile(percentile, numeric_only=True).reset_index()
                fig_bar1 = bar_plot(temp_df,["slot","symbol","cell"],"start_min_symbol_cell",yfield_name="latency",
                                    x_range=common_x_range_symbol_cell,
                                    y_range=common_y_range_list[current_y_index+0],
                                    title='%s %.1f%% earliest per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='latency relative to window start (usec)')
                fig_bar2 = bar_plot(temp_df,["slot","symbol","cell"],"end_max_symbol_cell",yfield_name="latency",
                                    x_range=common_x_range_symbol_cell,
                                    y_range=common_y_range_list[current_y_index+1],
                                    title='%s %.1f%% latest per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='latency relative to window start (usec)')
                fig_bar3 = bar_plot(temp_df,["slot","symbol","cell"],"duration_symbol_cell",yfield_name="duration",
                                    x_range=common_x_range_symbol_cell,
                                    y_range=common_y_range_list[current_y_index+2],
                                    title='%s %.1f%% duration per slot/symbol/cell'%(summary_label, percentile*100),xlabel="slot,symbol,cell",ylabel='duration (usec)')
                #Automatically scale the y-axis range (both min and max) to include negative values
                min_start_symbol_cell = temp_df.start_min_symbol_cell.min()
                common_y_range_list[current_y_index+0].start = min(common_y_range_list[current_y_index+0].start, min_start_symbol_cell*1.1 if min_start_symbol_cell < 0 else 0)
                common_y_range_list[current_y_index+0].end = max(common_y_range_list[current_y_index+0].end,temp_df.start_min_symbol_cell.max()*1.1)
                common_y_range_list[current_y_index+1].end = max(common_y_range_list[current_y_index+1].end,temp_df.end_max_symbol_cell.max()*1.1)
                common_y_range_list[current_y_index+2].end = max(common_y_range_list[current_y_index+2].end,temp_df.duration_symbol_cell.max()*1.1)
                current_y_index += 3
                col_list.append(fig_bar1)
                col_list.append(fig_bar2)
                col_list.append(fig_bar3)

        # Create RU TX enqueue margin plots
        if HAS_RU_TX_TIMES:
            # Add subsection header
            subsection_header = section_generator.getSections("sub_section", "RU TX Enqueue Margin Analysis")
            col_list.append(subsection_header)
            
            summary_label = "%s [RU TX, Enqueue Margin]"%label

            # Calculate enqueue margin and group by t0_timestamp + symbol to get max
            df_tx_timing_copy = df_tx_timing.copy()
            df_tx_timing_copy['enqueue_margin'] = df_tx_timing_copy['tx_deadline'] - df_tx_timing_copy['enqueue_deadline']
            
            # Get unique channels found in the data and separate data by channel
            unique_channels = sorted(df_tx_timing_copy['channel'].unique())
            channel_data_list = []
            channel_labels = []
            
            for channel in unique_channels:
                channel_data = df_tx_timing_copy[df_tx_timing_copy['channel'] == channel].copy()
                if len(channel_data) > 0:
                    channel_data_list.append(channel_data)
                    channel_labels.append(channel)
                    
                    # Collect data for cross-file comparison
                    if channel not in channel_file_data:
                        channel_file_data[channel] = []
                    channel_file_data[channel].append((channel_data, label))
                    all_margin_values.extend(channel_data['enqueue_margin'].values)
                    
                    # Extract worst-case percentile values for summary table (consistent with other data collection)
                    margin_percentile = 1.0 - percentile
                    p_value = channel_data['enqueue_margin'].quantile(margin_percentile)
                    summary_table_data.append({
                        'file': label,
                        'metric': f'RU TX {channel} Channel Enqueue Margin ({margin_percentile*100:.1f}%)',
                        'value': p_value
                    })

            # Calculate min margins separately for each channel
            for data in channel_data_list:
                # Group by t0_timestamp + symbol and take min enqueue margin within this channel
                gb_symbol = data.groupby(['t0_timestamp','symbol'])
                data['enqueue_margin_min_symbol'] = gb_symbol['enqueue_margin'].transform(min)
                
                # Group by t0_timestamp + symbol + cell for cell-level analysis within this channel
                gb_symbol_cell = data.groupby(['t0_timestamp','symbol','cell'])
                data['enqueue_margin_min_symbol_cell'] = gb_symbol_cell['enqueue_margin'].transform(min)

            # Create margin statistics using 1-percentile (worst-case margins)
            margin_percentile = 1.0 - percentile
            
            # Create statistics for symbol basis - multi-series
            series_data_symbol = []
            
            for data in channel_data_list:
                temp_gb = data.groupby(['slot','symbol'])
                temp_df = temp_gb.quantile(margin_percentile, numeric_only=True).reset_index()
                series_data_symbol.append(temp_df)
            
            if series_data_symbol:
                fig_bar = bar_plot(series_data_symbol, ["slot","symbol"], "enqueue_margin_min_symbol", yfield_name="margin",
                                    x_range=common_x_range_symbol,
                                    y_range=common_y_range_list[current_y_index+0],
                                    title='%s %.1f%% enqueue margin per slot/symbol'%(summary_label, margin_percentile*100), xlabel="slot,symbol", ylabel='Enqueue Margin (usec)',
                                    series_labels=channel_labels)
                #Automatically scale the range to accommodate negative values
                max_vals = [df['enqueue_margin_min_symbol'].max() for df in series_data_symbol if len(df) > 0]
                min_vals = [df['enqueue_margin_min_symbol'].min() for df in series_data_symbol if len(df) > 0]
                if max_vals and min_vals:
                    common_y_range_list[current_y_index+0].end = max(common_y_range_list[current_y_index+0].end, max(max_vals)*1.1)
                    common_y_range_list[current_y_index+0].start = min(common_y_range_list[current_y_index+0].start, min(min_vals)*1.1)
                current_y_index += 1
                col_list.append(fig_bar)

            # Create statistics for symbol/cell basis - multi-series  
            series_data_symbol_cell = []
            
            for data in channel_data_list:
                temp_gb = data.groupby(['slot','symbol','cell'])
                temp_df = temp_gb.quantile(margin_percentile, numeric_only=True).reset_index()
                series_data_symbol_cell.append(temp_df)
            
            if series_data_symbol_cell:
                fig_bar_cell = bar_plot(series_data_symbol_cell, ["slot","symbol","cell"], "enqueue_margin_min_symbol_cell", yfield_name="margin",
                                    x_range=common_x_range_symbol_cell,
                                    y_range=common_y_range_list[current_y_index+0],
                                    title='%s %.1f%% enqueue margin per slot/symbol/cell'%(summary_label, margin_percentile*100), xlabel="slot,symbol,cell", ylabel='Enqueue Margin (usec)',
                                    series_labels=channel_labels)
                #Automatically scale the range to accommodate negative values
                max_vals_cell = [df['enqueue_margin_min_symbol_cell'].max() for df in series_data_symbol_cell if len(df) > 0]
                min_vals_cell = [df['enqueue_margin_min_symbol_cell'].min() for df in series_data_symbol_cell if len(df) > 0]
                if max_vals_cell and min_vals_cell:
                    common_y_range_list[current_y_index+0].end = max(common_y_range_list[current_y_index+0].end, max(max_vals_cell)*1.1)
                    common_y_range_list[current_y_index+0].start = min(common_y_range_list[current_y_index+0].start, min(min_vals_cell)*1.1)
                current_y_index += 1
                col_list.append(fig_bar_cell)

            # Add CDF plots for enqueue margin analysis
            # Create CDF 1: All data broken out by symbol (0-13)
            df_margin_by_symbol_list = []
            for sym in range(14):
                symbol_data = df_tx_timing_copy[df_tx_timing_copy.symbol==sym]
                if len(symbol_data) > 0:
                    df_margin_by_symbol_list.append(symbol_data)

            # Create CDF 2: All data broken out by channel name (reuse existing channel_data_list)
            df_margin_by_channel_list = [data for data in channel_data_list if len(data) > 0]

            if df_margin_by_symbol_list or df_margin_by_channel_list:
                # Calculate x-axis range based on data
                max_margin = df_tx_timing_copy['enqueue_margin'].max()
                min_margin = df_tx_timing_copy['enqueue_margin'].min()
                xstart = min_margin - 100
                xstop = max_margin + 100  # Add 100us extra margin
                
                if df_margin_by_symbol_list:
                    fig_margin_by_symbol_legend = comparison_legend(['Symbol %i'%sym for sym in range(14) if len(df_tx_timing_copy[df_tx_timing_copy.symbol==sym]) > 0],height=200,width=400)
                    fig_margin_by_symbol_cdf = ccdf_comparison(df_margin_by_symbol_list,'enqueue_margin',xstart=xstart,xstop=xstop,title="%s Enqueue Margin By Symbol CDF"%summary_label,xlabel="Enqueue Margin (usec)",height=600,width=800,no_complement=True)
                    col_list.append(fig_margin_by_symbol_legend)
                    col_list.append(fig_margin_by_symbol_cdf)
                
                if df_margin_by_channel_list:
                    fig_margin_by_channel_legend = comparison_legend(channel_labels,height=200,width=400)
                    fig_margin_by_channel_cdf = ccdf_comparison(df_margin_by_channel_list,'enqueue_margin',xstart=xstart,xstop=xstop,title="%s Enqueue Margin By Channel CDF"%summary_label,xlabel="Enqueue Margin (usec)",height=600,width=800,no_complement=True)
                    col_list.append(fig_margin_by_channel_legend)
                    col_list.append(fig_margin_by_channel_cdf)

        row_list.append(column(col_list))
    
    # Create the final layout with cross-file analysis at top, individual files below
    final_layout = []
    
    # Initialize cross_file_channel_plots
    cross_file_channel_plots = []
    
    # Top section: Cross-file channel analysis
    if channel_file_data:
        # Calculate x-axis range across all files and channels
        max_margin_all = max(all_margin_values)
        min_margin_all = min(all_margin_values)
        xstart_all = min_margin_all - 100
        xstop_all = max_margin_all + 100
        
        # Create separate CDF for each channel type - organize as row of plots
        for channel in sorted(channel_file_data.keys()):
            file_data_list = [data for data, label in channel_file_data[channel]]
            file_labels = [f"{channel} - {label}" for data, label in channel_file_data[channel]]
            
            # Create plot for any channel
            fig_legend = comparison_legend(file_labels, height=150, width=400)
            plot_title = f"{channel} Channel: Cross-File Enqueue Margin CDF" if len(labels) > 1 else f"{channel} Channel: Enqueue Margin CDF"
            fig_cdf = ccdf_comparison(file_data_list, 'enqueue_margin',
                                    xstart=xstart_all, xstop=xstop_all,
                                    title=plot_title,
                                    xlabel="Enqueue Margin (usec)",
                                    height=500, width=800, no_complement=True)
            
            # Each channel gets its own column
            channel_col = column([fig_legend, fig_cdf])
            cross_file_channel_plots.append(channel_col)
        
        # Prepare cross-file plots but don't add to layout yet
    cross_file_sections = []
    
    # DU RX Summaries cross-file section
    if du_rx_timing_file_data or du_rx_srs_timing_file_data:
        du_rx_cross_header = section_generator.getSections("sub_section", "DU RX Summaries")
        cross_file_sections.append(row([du_rx_cross_header]))
        
        du_rx_plots = []
        
        # UL U Plane latest times CCDF
        if du_rx_timing_file_data:
            ul_u_plane_data_list = [data for data, label in du_rx_timing_file_data]
            ul_u_plane_labels = [f"UL U Plane - {label}" for data, label in du_rx_timing_file_data]
            
            # Extract percentile values for summary table
            for data, label in du_rx_timing_file_data:
                p_value = data['end_max_slot'].quantile(percentile)
                summary_table_data.append({
                    'file': label,
                    'metric': f'DU RX UL U Plane Latest Packet ({percentile*100:.1f}%)',
                    'value': p_value
                })
            
            # Calculate data-driven x-axis range
            min_values = [data['end_max_slot'].min() for data in ul_u_plane_data_list]
            max_values = [data['end_max_slot'].max() for data in ul_u_plane_data_list]
            xstart_ul_u = min(0, min(min_values))
            xstop_ul_u = max(max_values) + 50
            
            fig_ul_u_legend = comparison_legend(ul_u_plane_labels, height=150, width=400)
            fig_ul_u_ccdf = ccdf_comparison(ul_u_plane_data_list, 'end_max_slot',
                                          xstart=xstart_ul_u, xstop=xstop_ul_u,
                                          title="UL U Plane: Cross-File Latest Packet Times CCDF",
                                          xlabel="Latest Packet RX Time - Desired TX Time (usec)",
                                          height=500, width=800)
            du_rx_plots.append(column([fig_ul_u_legend, fig_ul_u_ccdf]))
        
        # SRS latest times CCDF  
        if du_rx_srs_timing_file_data:
            srs_data_list = [data for data, label in du_rx_srs_timing_file_data]
            srs_labels = [f"SRS - {label}" for data, label in du_rx_srs_timing_file_data]
            
            # Extract percentile values for summary table
            for data, label in du_rx_srs_timing_file_data:
                p_value = data['end_max_slot'].quantile(percentile)
                summary_table_data.append({
                    'file': label,
                    'metric': f'DU RX SRS Latest Packet ({percentile*100:.1f}%)',
                    'value': p_value
                })
            
            # Calculate data-driven x-axis range
            min_values = [data['end_max_slot'].min() for data in srs_data_list]
            max_values = [data['end_max_slot'].max() for data in srs_data_list]
            xstart_srs = min(0, min(min_values))
            xstop_srs = max(max_values) + 50
            
            fig_srs_legend = comparison_legend(srs_labels, height=150, width=400)
            fig_srs_ccdf = ccdf_comparison(srs_data_list, 'end_max_slot',
                                         xstart=xstart_srs, xstop=xstop_srs,
                                         title="SRS: Cross-File Latest Packet Times CCDF",
                                         xlabel="Latest Packet RX Time - Desired TX Time (usec)",
                                         height=500, width=800)
            du_rx_plots.append(column([fig_srs_legend, fig_srs_ccdf]))
        
        if du_rx_plots:
            cross_file_sections.append(row(du_rx_plots))
    
    # RU RX Summaries cross-file section
    if any(ru_rx_timing_file_data.values()):
        ru_rx_cross_header = section_generator.getSections("sub_section", "RU RX Summaries")
        cross_file_sections.append(row([ru_rx_cross_header]))
        
        ru_rx_plots = []
        
        for task in ru_rx_timing_file_data.keys():
            if ru_rx_timing_file_data[task]:
                task_data_list = [data for data, label in ru_rx_timing_file_data[task]]
                
                # Skip this task if all dataframes are empty
                if not any(len(data) > 0 for data in task_data_list):
                    continue
                    
                # Filter to only non-empty dataframes for processing
                task_data_list = [data for data, label in ru_rx_timing_file_data[task] if len(data) > 0]
                task_labels = [f"{task} - {label}" for data, label in ru_rx_timing_file_data[task] if len(data) > 0]
                
                # Extract percentile values for summary table
                for data, label in ru_rx_timing_file_data[task]:
                    if len(data) == 0:
                        continue
                    p_value = data['end_max_slot'].quantile(percentile)
                    summary_table_data.append({
                        'file': label,
                        'metric': f'RU RX {task} Latest Packet ({percentile*100:.1f}%)',
                        'value': p_value
                    })
                
                # Calculate data-driven x-axis range
                min_values = [data['end_max_slot'].min() for data in task_data_list if len(data) > 0]
                max_values = [data['end_max_slot'].max() for data in task_data_list if len(data) > 0]
                xstart_task = min(0, min(min_values))
                xstop_task = max(max_values) + 50
                
                fig_task_legend = comparison_legend(task_labels, height=150, width=400)
                fig_task_ccdf = ccdf_comparison(task_data_list, 'end_max_slot',
                                              xstart=xstart_task, xstop=xstop_task,
                                              title=f"{task}: Cross-File Latest Packet Times CCDF",
                                              xlabel="Latest Packet RX Time - Desired TX Time (usec)",
                                              height=500, width=800)
                ru_rx_plots.append(column([fig_task_legend, fig_task_ccdf]))
        
        if ru_rx_plots:
            cross_file_sections.append(row(ru_rx_plots))
    
    # RU TX Summaries cross-file section
    if cross_file_channel_plots:
        rx_enqueue_header = section_generator.getSections("sub_section", "RX Enqueue Summaries")
        cross_file_sections.append(row([rx_enqueue_header]))
        cross_file_sections.append(row(cross_file_channel_plots))
    
    # Add cross-file section if we have any cross-file content
    if summary_table_data or cross_file_sections:
        # Add cross-file header
        cross_file_header = section_generator.getSections("section", "Cross-File Summaries")
        final_layout.append(cross_file_header)
        
        # Add summary table if we have data
        if summary_table_data:
            from aerial_postproc.table_utils import create_html_table
            
            # Convert to DataFrame for easier handling
            summary_df = pd.DataFrame(summary_table_data)
            
            # Round values to nearest microsecond (no decimals)
            summary_df['value'] = summary_df['value'].round(0).astype(int)
            
            # Pivot the table to have metrics as rows and files as columns
            pivot_df = summary_df.pivot(index='metric', columns='file', values='value').fillna('')
            
            # Reorder columns to match input label order for consistent display
            ordered_columns = []
            for label in labels:
                if label in pivot_df.columns:
                    ordered_columns.append(label)
            
            # Add any columns that weren't in the labels list
            for col in pivot_df.columns:
                if col not in ordered_columns:
                    ordered_columns.append(col)
            
            pivot_df = pivot_df[ordered_columns]
            
            # Reset index to make 'metric' a regular column
            pivot_df = pivot_df.reset_index()
            
            # Create subsection header for summary table
            summary_table_header = section_generator.getSections("sub_section", "Summary Table")
            
                        # Create HTML table using the generic utility
            html_table = create_html_table(
                pivot_df, 
                include_units_in_headers=True,
                unit_suffix="(μs)",
                exclude_unit_columns=['metric'],
                table_width="1400px"  # Wide table to accommodate multiple files
            )
            
            # Add summary table to layout
            final_layout.append(row([summary_table_header]))
            final_layout.append(row([html_table]))
        
        # Add all cross-file sections
        for section in cross_file_sections:
            final_layout.append(section)

    if row_list:
        
        # Bottom section: Individual file analysis
        individual_files_header = section_generator.getSections("section", "Individual File Analysis")
        final_layout.append(row([individual_files_header]))
        final_layout.append(row(row_list))
    
    return column(final_layout)


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
        "-l", "--labels", type=str, nargs="+", help="Names for each phy log"
    )
    parser.add_argument(
        "-o", "--out_filename", help="Save output directly to file, this parameter specifies filename"
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
        "-s", "--slot_selection", help="List of comma separate ranges for slot selection (0-79). Example \"1-4,7,9-11\" --> [1,2,3,4,7,9,10,11]"
    )
    parser.add_argument(
        "-y", "--symbol_selection", help="List of comma separate ranges for symbol selection (0-13). Example \"1-4,7,9-11\" --> [1,2,3,4,7,9,10,11]"
    )
    parser.add_argument(
        "-e", "--mmimo_enable", action="store_true", help="Modifies timeline requirement to mmimo settings"
    )
    parser.add_argument(
        "-p", "--percentile", default=0.99, type=float, help="Percentile for statistical analysis (0-1, default 0.99)"
    )
    args = parser.parse_args()

    main(args)
