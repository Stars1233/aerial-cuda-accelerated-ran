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
from aerial_postproc.logplot import comparison_legend, ccdf_breakout, timeline_plot_generic, ccdf_comparison, color_select

import argparse
import numpy as np
import os
import pandas as pd

from bokeh.plotting import save, show, figure
from bokeh.io import output_file
from bokeh.models import Range1d, ColumnDataSource, CrosshairTool
from bokeh.layouts import row, column, layout

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
    df_ti_list = []
    df_tx_timing_list = []
    df_tx_timing_sum_list = []
    df_ru_rx_timing_list = []
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
            df_du_rx_timing_list.append(io_class.data['df_du_rx_timing'])
            df_du_rx_srs_timing_list.append(io_class.data['df_du_rx_srs_timing'])
            df_ti_list.append(io_class.data['df_ti'])
            df_tx_timing_list.append(io_class.data['df_tx_timing'])
            df_tx_timing_sum_list.append(io_class.data['df_tx_timing_sum'])
            df_ru_rx_timing_list.append(io_class.data['df_ru_rx_timing'])
        else:
            print("Error:: Invalid input_data %s"%input_data)

    #Create plots
    fig = get_figs(df_du_rx_timing_list,df_du_rx_srs_timing_list,df_ti_list,df_tx_timing_list,df_tx_timing_sum_list,df_ru_rx_timing_list,labels,per_symbol=args.per_symbol,add_downlink=args.add_downlink,que_check=args.que_check,cell_symbol_index=args.cell_symbol_index)

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

def get_figs(df_du_rx_timing_list,df_du_rx_srs_timing_list,df_ti_list,df_tx_timing_list,df_tx_timing_sum_list,df_ru_rx_timing_list,labels,per_symbol=False,add_downlink=False,que_check=False,cell_symbol_index=False):

    col_list = []
    y_range = None
    x_range = None
    for df_du_rx_timing,df_du_rx_srs_timing,df_ti,df_tx_timing,df_tx_timing_sum,df_ru_rx_timing,label in zip(df_du_rx_timing_list,df_du_rx_srs_timing_list,df_ti_list,df_tx_timing_list,df_tx_timing_sum_list,df_ru_rx_timing_list,labels):

        #DU messages
        HAS_TI_TIMES = len(df_ti) > 0
        HAS_DU_RX_TIMES = len(df_du_rx_timing) > 0
        HAS_DU_RX_SRS_TIMES = len(df_du_rx_srs_timing) > 0
        SYMBOL_TIME = 500/14.
        if HAS_DU_RX_TIMES:
            ulu_rx_window_start = df_du_rx_timing.window_start_deadline.min()
            print("ULU rx window start: %s"%ulu_rx_window_start)

        if HAS_DU_RX_SRS_TIMES:
            srs_rx_window_start = df_du_rx_srs_timing.window_start_deadline.min()
            print("SRS rx window start: %s"%srs_rx_window_start)
            
        #RU messages
        HAS_RU_RX_TIMES = add_downlink and (len(df_ru_rx_timing) > 0)
        HAS_RU_TX_TIMES = per_symbol and len(df_tx_timing) > 0
        HAS_RU_TX_SUM_TIMES = len(df_tx_timing_sum) > 0

        #Show times relative to window
        ENABLE_FIG1 = HAS_DU_RX_TIMES or HAS_RU_TX_TIMES
        if(not ENABLE_FIG1):
            print("Warning :: No timing plot as there is no timing data :: HAS_DU_RX_TIMES=%s HAS_RU_TX_TIMES=%s"%(HAS_DU_RX_TIMES,HAS_RU_TX_TIMES))

        #Timeline view of transfers (transfer_fig)
        ENABLE_TRANSFER_FIG = HAS_DU_RX_TIMES and HAS_RU_RX_TIMES
        if(not ENABLE_TRANSFER_FIG):
            print("Warning :: No combined timing plot as we do not have both sides :: HAS_DU_RX_TIMES=%s HAS_RU_TX_TIMES=%s"%(HAS_DU_RX_TIMES,HAS_RU_TX_TIMES))

        #CCDF of DU RX (fig_legend+fig_ccdf)
        ENABLE_DU_CCDF = HAS_DU_RX_TIMES
        if(not ENABLE_DU_CCDF):
            print("Warning :: No DU CCDF as DU RX Times not available")

        #RU enqueue timeline overlay (fig2)
        RU_TX_RELATIVE_TO_SYMBOL_WINDOW = False
        ENABLE_RU_ENQUEUE_TIMELINE = HAS_RU_TX_TIMES or HAS_RU_TX_SUM_TIMES
        if(not ENABLE_RU_ENQUEUE_TIMELINE):
            print("Warning :: No RU enqueue timeline as we do not have any RU timing data :: HAS_RU_TX_TIMES=%s HAS_RU_TX_SUM_TIMES=%s"%(HAS_RU_TX_TIMES,HAS_RU_TX_SUM_TIMES))

        print("Plots enabled: ENABLE_FIG1=%s ENABLE_DU_CCDF=%s ENABLE_RU_ENQUEUE_TIMELINE=%s ENABLE_TRANSFER_FIG=%s HAS_DU_RX_TIMES=%s HAS_DU_RX_SRS_TIMES=%s"%(ENABLE_FIG1,ENABLE_DU_CCDF,ENABLE_RU_ENQUEUE_TIMELINE,ENABLE_TRANSFER_FIG,HAS_DU_RX_TIMES,HAS_DU_RX_SRS_TIMES))

        if HAS_DU_RX_TIMES:
            #Add tir of window
            df_du_rx_timing['window_tir'] = df_du_rx_timing['tir'] + df_du_rx_timing['window_start_deadline']*1e-6
            df_du_rx_timing['start_tir'] = df_du_rx_timing['window_tir'] + df_du_rx_timing['start_offset']*1e-6
            df_du_rx_timing['end_tir'] = df_du_rx_timing['window_tir'] + df_du_rx_timing['end_offset']*1e-6
            du_rx_timing_source = ColumnDataSource(df_du_rx_timing)

            #Add min start and max end to original dataframe, create a view that only contains symbol 0 points
            gb = df_du_rx_timing.groupby(['t0_timestamp','sfn','slot'])
            df_du_rx_timing = df_du_rx_timing.assign(start_min=gb['start_offset'].transform(min),end_max=gb['end_offset'].transform(max))
            df_du_rx_timing_per_slot = df_du_rx_timing[(df_du_rx_timing.symbol == 0) & (df_du_rx_timing.cell == df_du_rx_timing.cell.min())]
            source_slot = ColumnDataSource(df_du_rx_timing_per_slot)

        if HAS_DU_RX_SRS_TIMES:
            #Add tir of window for SRS
            df_du_rx_srs_timing['window_tir'] = df_du_rx_srs_timing['tir'] + df_du_rx_srs_timing['window_start_deadline']*1e-6
            df_du_rx_srs_timing['start_tir'] = df_du_rx_srs_timing['window_tir'] + df_du_rx_srs_timing['start_offset']*1e-6
            df_du_rx_srs_timing['end_tir'] = df_du_rx_srs_timing['window_tir'] + df_du_rx_srs_timing['end_offset']*1e-6
            du_rx_srs_timing_source = ColumnDataSource(df_du_rx_srs_timing)

            #Add min start and max end to original dataframe, create a view that only contains one point per slot
            gb = df_du_rx_srs_timing.groupby(['t0_timestamp','sfn','slot'])
            df_du_rx_srs_timing = df_du_rx_srs_timing.assign(start_min=gb['start_offset'].transform(min),end_max=gb['end_offset'].transform(max))
            
            # Take first occurrence for each slot instead of filtering by symbol 0
            df_du_rx_srs_timing_per_slot = df_du_rx_srs_timing.groupby(['t0_timestamp','sfn','slot']).first().reset_index()
            source_srs_slot = ColumnDataSource(df_du_rx_srs_timing_per_slot)

        if HAS_RU_RX_TIMES:
            #Add min start and max end to original dataframe
            df_ru_rx_timing['window_tir'] = df_ru_rx_timing['tir'] + df_ru_rx_timing['window_start_deadline']*1e-6
            df_ru_rx_timing['start_tir'] = df_ru_rx_timing['window_tir'] + df_ru_rx_timing['start_offset']*1e-6
            df_ru_rx_timing['end_tir'] = df_ru_rx_timing['window_tir'] + df_ru_rx_timing['end_offset']*1e-6
            ru_dlu_rx_timing_source = ColumnDataSource(df_ru_rx_timing[df_ru_rx_timing.task=='DL U Plane'])
            ru_dlc_rx_timing_source = ColumnDataSource(df_ru_rx_timing[df_ru_rx_timing.task=='DL C Plane'])
            ru_ulc_rx_timing_source = ColumnDataSource(df_ru_rx_timing[df_ru_rx_timing.task=='UL C Plane'])

        if HAS_TI_TIMES and HAS_DU_RX_TIMES:
            current_df_ti = df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Wait Order')].copy()
            current_df_ti['window_tir'] = current_df_ti['tir'] + (ulu_rx_window_start + 13.*500/14.)*1e-6
            current_df_ti['end_offset'] = current_df_ti['end_deadline'] - (ulu_rx_window_start + 13.*500/14.)
            source2 = ColumnDataSource(current_df_ti)

            current_df_ti = df_ti[(df_ti.task=='UL Task UL AGGR 3') & (df_ti.subtask=='Wait SRS Order')].copy()
            current_df_ti['window_tir'] = current_df_ti['tir'] + (ulu_rx_window_start + 13.*500/14.)*1e-6
            current_df_ti['end_offset'] = current_df_ti['end_deadline'] - (ulu_rx_window_start + 13.*500/14.)
            source2_srs = ColumnDataSource(current_df_ti)


        if ENABLE_FIG1:
            # Create figure, sharing same x and y axis
            TOOLTIPS = [('start_offset','@start_offset'),
                        ('end_offset','@end_offset'),
                        ('cell','@cell'),
                        ('sfn','@sfn'),
                        ('slot','@slot'),
                        ('symbol','@symbol'),]
            if(x_range is not None):
                fig1 = figure(title="[%s] DU RX Time Relative To Start Of Symbol Window"%label,tooltips=TOOLTIPS, height=600, width=1000,
                            x_range=x_range, y_range=y_range)
            else:
                fig1 = figure(title="[%s] DU RX Time Relative To Start Of Symbol Window"%label,tooltips=TOOLTIPS, height=600, width=1000)
                x_range = fig1.x_range
                y_range = fig1.y_range

            if HAS_DU_RX_TIMES:

                fig1.circle(x='window_tir',y='start_min',source=source_slot,legend_label="Earliest Packet For Slot",color='green')
                fig1.circle(x='window_tir',y='end_max',source=source_slot,legend_label="Latest Packet For Slot",color='red')

                if(per_symbol):
                    fig1.circle(x='window_tir',y='start_offset',source=du_rx_timing_source,legend_label="DU Earliest Packet (per Symbol/Cell)",color='lightgreen')
                    fig1.circle(x='window_tir',y='end_offset',source=du_rx_timing_source,legend_label="DU Latest Packet (per Symbol/Cell)",color='salmon')
            
            if HAS_RU_RX_TIMES and per_symbol:
                fig1.circle(x='window_tir',y='start_offset',source=ru_dlu_rx_timing_source,legend_label="RU DLU Earliest Packet (per Symbol/Cell)",color='blue')
                fig1.circle(x='window_tir',y='end_offset',source=ru_dlu_rx_timing_source,legend_label="RU DLU Latest Packet (per Symbol/Cell)",color='purple')
                fig1.circle(x='window_tir',y='start_offset',source=ru_dlc_rx_timing_source,legend_label="RU DLC Earliest Packet (per Symbol/Cell)",color='orange')
                fig1.circle(x='window_tir',y='end_offset',source=ru_dlc_rx_timing_source,legend_label="RU DLC Latest Packet (per Symbol/Cell)",color='black')
                fig1.circle(x='window_tir',y='start_offset',source=ru_ulc_rx_timing_source,legend_label="RU ULC Earliest Packet (per Symbol/Cell)",color='brown')
                fig1.circle(x='window_tir',y='end_offset',source=ru_ulc_rx_timing_source,legend_label="RU ULC Latest Packet (per Symbol/Cell)",color='darkgray')

            if HAS_DU_RX_SRS_TIMES:
                fig1.circle(x='window_tir',y='start_min',source=source_srs_slot,legend_label="SRS Earliest Packet For Slot",color='darkblue')
                fig1.circle(x='window_tir',y='end_max',source=source_srs_slot,legend_label="SRS Latest Packet For Slot",color='darkred')

                if(per_symbol):
                    fig1.circle(x='window_tir',y='start_offset',source=du_rx_srs_timing_source,legend_label="SRS Earliest Packet (per Symbol/Cell)",color='lightblue')
                    fig1.circle(x='window_tir',y='end_offset',source=du_rx_srs_timing_source,legend_label="SRS Latest Packet (per Symbol/Cell)",color='pink')

            if HAS_TI_TIMES and HAS_DU_RX_TIMES:
                fig1.circle(x='window_tir',y='end_offset',source=source2,legend_label="Order Kernel Complete",color='orange')
                fig1.circle(x='window_tir',y='end_offset',source=source2_srs,legend_label="SRS Order Kernel Complete",color='purple')
            fig1.legend.location = "top_left"
            fig1.legend.click_policy = "hide"
            fig1.xaxis.axis_label = "Symbol Window, Time In Run (sec)"
            fig1.yaxis.axis_label = "RX Time Relative To Window (usec)"

        if ENABLE_DU_CCDF:
            # Calculate xstop based on maximum timing value for ULU
            ulu_max = max(
                df_du_rx_timing['start_min'].max(),
                df_du_rx_timing['end_max'].max(),
                df_du_rx_timing['start_offset'].max(),
                df_du_rx_timing['end_offset'].max()
            )
            xstop = ulu_max

            # ULU CCDF (always create)
            color_list_ulu = ['green','red','lightgreen','salmon']
            fig_legend_ulu = comparison_legend(['ULU Earliest Packet (min over slot)','ULU Latest Packet (max over slot)',
                                              'ULU Earliest Packet (per symbol)','ULU Latest Packet (per symbol)'],
                                              height=200,width=400,color_list=color_list_ulu)
            
            fig_ccdf_ulu = ccdf_breakout(df_du_rx_timing,['start_min','end_max','start_offset','end_offset'],
                                       xdelta=1,xstart=-10,xstop=xstop,
                                       title="DU RX Earliest/Latest Packet (ULU)",
                                       xlabel="Packet RX Times - Desired TX Time (T0 + %ius)"%ulu_rx_window_start,
                                       height=800,width=800,color_list=color_list_ulu)

            if HAS_DU_RX_SRS_TIMES:
                # Update xstop to include SRS maximum values
                srs_max = max(
                    df_du_rx_srs_timing['start_min'].max(),
                    df_du_rx_srs_timing['end_max'].max(),
                    df_du_rx_srs_timing['start_offset'].max(),
                    df_du_rx_srs_timing['end_offset'].max()
                )
                xstop = max(xstop, srs_max)

                # SRS CCDF (only if we have SRS data)
                color_list_srs = ['darkblue','darkred','lightblue','pink']
                fig_legend_srs = comparison_legend(['SRS Earliest Packet (min over slot)','SRS Latest Packet (max over slot)',
                                                  'SRS Earliest Packet (per symbol)','SRS Latest Packet (per symbol)'],
                                                  height=200,width=400,color_list=color_list_srs)
                
                fig_ccdf_srs = ccdf_breakout(df_du_rx_srs_timing,['start_min','end_max','start_offset','end_offset'],
                                           xdelta=1,xstart=-10,xstop=xstop,
                                           title="DU RX Earliest/Latest Packet (SRS)",
                                           xlabel="Packet RX Times - Desired TX Time (T0 + %ius)"%srs_rx_window_start,
                                           height=800,width=800,color_list=color_list_srs)

                # Update ULU plot to use same xstop as SRS for consistency
                fig_ccdf_ulu.x_range.end = xstop

        #RU TX plots
        if ENABLE_RU_ENQUEUE_TIMELINE:
            if HAS_RU_TX_TIMES:
                df_tx_timing['tx_tir'] = df_tx_timing['tir'] + df_tx_timing['tx_deadline']*1e-6

                #Produce metrics based on symbol window
                sym_tx_field = "tx_sym_deadline"
                sym_enqueue_field = "enqueue_sym_deadline"
                sym_headroom_field = "enqueue_headroom"
                df_tx_timing[sym_tx_field] = df_tx_timing['tx_deadline'] - df_tx_timing['window_start_deadline']
                df_tx_timing[sym_enqueue_field] = df_tx_timing['enqueue_deadline'] - df_tx_timing['window_start_deadline']
                df_tx_timing[sym_headroom_field] = df_tx_timing['tx_deadline'] - df_tx_timing['enqueue_deadline']

                if RU_TX_RELATIVE_TO_SYMBOL_WINDOW:
                    tx_field = sym_tx_field
                    enqueue_field = sym_enqueue_field
                    fig_title = "RU TX Time Relative To Start Of Symbol Window"
                    fig_yaxis_label = "TX/Enqueue Time Relative To Start Of Symbol Window (usec)"
                else:
                    tx_field = "tx_deadline"
                    enqueue_field = "enqueue_deadline"
                    fig_title = "RU TX Time Relative To T0"
                    fig_yaxis_label = "TX/Enqueue Time Relative To T0 (usec)"

                # Get unique channels found in the data for dynamic sources and plotting
                unique_channels = sorted(df_tx_timing['channel'].unique())
                channel_sources = {}
                for channel in unique_channels:
                    channel_data = df_tx_timing[df_tx_timing.channel==channel]
                    channel_sources[channel] = ColumnDataSource(channel_data)

                # Create CCDF 1: All data broken out by symbol (0-13)
                df_enqueue_by_symbol_list = []
                for sym in range(14):
                    df_enqueue_by_symbol_list.append(df_tx_timing[df_tx_timing.symbol==sym])

                # Create CCDF 2: All data broken out by channel name
                df_enqueue_by_channel_list = []
                channel_labels = []
                for channel in unique_channels:
                    channel_data = df_tx_timing[df_tx_timing.channel==channel]
                    if len(channel_data) > 0:
                        df_enqueue_by_channel_list.append(channel_data)
                        channel_labels.append(channel)
                
                # Calculate x-axis range based on data
                max_headroom = df_tx_timing[sym_headroom_field].max()
                min_headroom = df_tx_timing[sym_headroom_field].min()
                xstart = min_headroom - 100
                xstop = max_headroom + 100  # Add 100us extra margin
                
                fig_enqueue_by_symbol_legend = comparison_legend(['Symbol %i'%sym for sym in range(14)],height=200,width=400)
                fig_enqueue_by_symbol_ccdf = ccdf_comparison(df_enqueue_by_symbol_list,sym_headroom_field,xstart=xstart,xstop=xstop,title="RU Enqueue Headroom By Symbol CDF",xlabel="Enqueue Headroom (usec)",height=600,width=800,no_complement=True)
                fig_enqueue_by_channel_legend = comparison_legend(channel_labels,height=200,width=400)
                fig_enqueue_by_channel_ccdf = ccdf_comparison(df_enqueue_by_channel_list,sym_headroom_field,xstart=xstart,xstop=xstop,title="RU Enqueue Headroom By Channel CDF",xlabel="Enqueue Headroom (usec)",height=600,width=800,no_complement=True)

            if HAS_RU_TX_SUM_TIMES:
                df_tx_timing_sum['tx_tir'] = df_tx_timing_sum['tir'] + df_tx_timing_sum['window_start_sym13_deadline']*1e-6
                df_tx_timing_sum['enqueue_sym0_margin'] = df_tx_timing_sum['enqueue_deadline'] - df_tx_timing_sum['window_start_sym0_deadline']
                df_tx_timing_sum['enqueue_sym13_margin'] = df_tx_timing_sum['enqueue_deadline'] - df_tx_timing_sum['window_start_sym13_deadline']
                
                # Create dynamic sources for TX_TIMINGS_SUM based on actual channels found
                unique_channels_sum = sorted(df_tx_timing_sum['channel'].unique())
                channel_sources_sum = {}
                for channel in unique_channels_sum:
                    channel_data_sum = df_tx_timing_sum[df_tx_timing_sum.channel==channel]
                    channel_sources_sum[channel] = ColumnDataSource(channel_data_sum)

            
            TOOLTIPS = [('tx_field','@%s'%tx_field),
                        ('enqueue_field','@%s'%enqueue_field),
                        ('cell','@cell'),
                        ('sfn','@sfn'),
                        ('slot','@slot'),
                        ('symbol','@symbol'),]
            
            fig2 = figure(title="[%s] %s"%(label,fig_title),tooltips=TOOLTIPS, height=600, width=1000)
                
            # Calculate total number of series across both TX_TIMINGS and TX_SUM for unique coloring
            tx_series_count = len(unique_channels) * 2 if HAS_RU_TX_TIMES else 0  # TX + Enqueue for each channel
            sum_series_count = len(unique_channels_sum) if HAS_RU_TX_SUM_TIMES else 0  # One color per channel (sym0/sym13 share color)
            total_series = tx_series_count + sum_series_count

            if HAS_RU_TX_TIMES:
                # Plot TX and Enqueue times dynamically for all channels found
                for i, channel in enumerate(unique_channels):
                    color_tx = color_select(i, total_series)
                    color_enqueue = color_select(i + len(unique_channels), total_series)
                    fig2.circle(x='tx_tir',y=tx_field,source=channel_sources[channel],legend_label=f"{channel} TX Time",color=color_tx)
                    fig2.circle(x='tx_tir',y=enqueue_field,source=channel_sources[channel],legend_label=f"{channel} Enqueue Time",color=color_enqueue)

            if HAS_RU_TX_SUM_TIMES:
                # Plot TX_TIMINGS_SUM data dynamically for all channels found
                # Offset indices to continue after TX_TIMINGS colors
                sum_color_offset = tx_series_count
                for i, channel in enumerate(unique_channels_sum):
                    if(not RU_TX_RELATIVE_TO_SYMBOL_WINDOW):
                        color = color_select(sum_color_offset + i, total_series)
                        fig2.circle(x='tx_tir',y=enqueue_field,source=channel_sources_sum[channel],legend_label=f"{channel} Cell Fully Enqueued",color=color)
                    else:
                        color = color_select(sum_color_offset + i, total_series)
                        fig2.circle(x='tx_tir',y='enqueue_sym0_margin',source=channel_sources_sum[channel],legend_label=f"{channel} Cell Fully Enqueued (sym0)",color=color)
                        fig2.circle(x='tx_tir',y='enqueue_sym13_margin',source=channel_sources_sum[channel],legend_label=f"{channel} Cell Fully Enqueued (sym13)",color=color)

            fig2.legend.location = "top_left"
            fig2.legend.click_policy = "hide"
            fig2.xaxis.axis_label = "Symbol Window, Time In Run (sec)"
            fig2.yaxis.axis_label = fig_yaxis_label

            if ENABLE_FIG1:
                fig2.x_range = fig1.x_range

        if ENABLE_TRANSFER_FIG:
            crosshair_tool = CrosshairTool()
            height = 600
            width = 1200
            
            # Conditionally create cell_symbol_index for ULC and DLC plots when enabled
            if cell_symbol_index:
                # Create cell_symbol_index field for better y-axis organization
                # This creates a unique index for each cell/symbol combination
                df_ru_rx_timing_copy = df_ru_rx_timing.copy()
                
                # Create a mapping of unique cell/symbol combinations to indices
                cell_symbol_combinations = df_ru_rx_timing_copy[['cell', 'symbol']].drop_duplicates().sort_values(['cell', 'symbol'])
                cell_symbol_combinations['cell_symbol_index'] = range(len(cell_symbol_combinations))
                
                # Merge back to get the index for each row
                df_ru_rx_timing_copy = df_ru_rx_timing_copy.merge(cell_symbol_combinations, on=['cell', 'symbol'], how='left')
                
                # Create a readable label for the cell_symbol_index
                df_ru_rx_timing_copy['cell_symbol_label'] = 'Cell ' + df_ru_rx_timing_copy['cell'].astype(str) + ', Symbol ' + df_ru_rx_timing_copy['symbol'].astype(str)
                
                # ULC plot with cell_symbol_index
                fig3 = timeline_plot_generic(df_ru_rx_timing_copy[df_ru_rx_timing_copy.task=='UL C Plane'],'cell_symbol_index',title="ULC Transfer Timeline View",height=height,width=width,x_range=None,
                                             tooltips=[('cell_symbol','@cell_symbol_label'),('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')],crosshair_tool=crosshair_tool)
                # DLC plot with cell_symbol_index
                fig6 = timeline_plot_generic(df_ru_rx_timing_copy[df_ru_rx_timing_copy.task=='DL C Plane'],'cell_symbol_index',title="DLC Transfer Timeline View",height=height,width=width,x_range=fig3.x_range,
                                             tooltips=[('cell_symbol','@cell_symbol_label'),('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')],crosshair_tool=crosshair_tool)
                # DLU plot uses regular cell (not affected by cell_symbol_index option)
                fig5 = timeline_plot_generic(df_ru_rx_timing[df_ru_rx_timing.task=='DL U Plane'],'cell',title="DLU Transfer Timeline View",height=height,width=width,x_range=fig3.x_range,
                                             tooltips=[('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')],crosshair_tool=crosshair_tool)
            else:
                # Default behavior - all plots use 'cell' for y-axis
                fig3 = timeline_plot_generic(df_ru_rx_timing[df_ru_rx_timing.task=='UL C Plane'],'cell',title="ULC Transfer Timeline View",height=height,width=width,x_range=None,
                                             tooltips=[('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')],crosshair_tool=crosshair_tool)
                fig6 = timeline_plot_generic(df_ru_rx_timing[df_ru_rx_timing.task=='DL C Plane'],'cell',title="DLC Transfer Timeline View",height=height,width=width,x_range=fig3.x_range,
                                             tooltips=[('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')],crosshair_tool=crosshair_tool)
                fig5 = timeline_plot_generic(df_ru_rx_timing[df_ru_rx_timing.task=='DL U Plane'],'cell',title="DLU Transfer Timeline View",height=height,width=width,x_range=fig3.x_range,
                                             tooltips=[('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')],crosshair_tool=crosshair_tool)
            
            # ULU plot always uses 'cell' (not affected by cell_symbol_index option)
            fig4 = timeline_plot_generic(df_du_rx_timing,'cell',title="ULU Transfer Timeline View",height=height,width=width,x_range=fig3.x_range,
                                         tooltips=[('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')],crosshair_tool=crosshair_tool)
            
            lateness_fig = figure(width=width,height=height//2,title="ULU/SRS/DLU Latencies",x_range=fig3.x_range,
                                  tooltips=[('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')])
            gb = df_du_rx_timing.groupby(['t0_timestamp','sfn','slot','symbol'])
            df_temp = df_du_rx_timing.assign(start_min=gb['start_offset'].transform(min),end_max=gb['end_offset'].transform(max))
            temp_source1 = ColumnDataSource(df_temp)
            df_dlu = df_ru_rx_timing[df_ru_rx_timing.task=='DL U Plane']
            gb = df_dlu.groupby(['t0_timestamp','sfn','slot','symbol'])
            df_temp = df_dlu.assign(start_min=gb['start_offset'].transform(min),end_max=gb['end_offset'].transform(max))
            temp_source2 = ColumnDataSource(df_temp)
            lateness_fig.circle(source=temp_source1,x='window_tir',y='start_min',legend_label='ULU earliest',color='green')
            lateness_fig.circle(source=temp_source1,x='window_tir',y='end_max',legend_label='ULU latest',color='red')
            lateness_fig.circle(source=temp_source2,x='window_tir',y='start_min',legend_label='DLU earliest',color='greenyellow')
            lateness_fig.circle(source=temp_source2,x='window_tir',y='end_max',legend_label='DLU latest',color='pink')
            lateness_fig.legend.click_policy='hide'
            lateness_fig.xaxis.axis_label='Time in Run'
            lateness_fig.yaxis.axis_label='Latency relative to symbol window (usec)'
            lateness_fig.add_tools(crosshair_tool)
            
            # Build list of figures for transfer_fig
            transfer_figs = [fig3, fig4]  # Start with ULC and ULU
            
            if HAS_DU_RX_SRS_TIMES:
                # Only create SRS figure if we have SRS data
                fig4_srs = timeline_plot_generic(df_du_rx_srs_timing,'cell',title="SRS Transfer Timeline View",height=height,width=width,x_range=fig3.x_range,
                                               tooltips=[('cell','@cell'),('sfn','@sfn'),('slot','@slot'),('symbol','@symbol')],crosshair_tool=crosshair_tool)
                transfer_figs.append(fig4_srs)  # Add SRS if we have the data
                
                # Add SRS points to lateness figure
                gb = df_du_rx_srs_timing.groupby(['t0_timestamp','sfn','slot','symbol'])
                df_temp_srs = df_du_rx_srs_timing.assign(start_min=gb['start_offset'].transform(min),end_max=gb['end_offset'].transform(max))
                temp_source_srs = ColumnDataSource(df_temp_srs)
                lateness_fig.circle(source=temp_source_srs,x='window_tir',y='start_min',legend_label='SRS earliest',color='blue')
                lateness_fig.circle(source=temp_source_srs,x='window_tir',y='end_max',legend_label='SRS latest',color='purple')
                
            transfer_figs.extend([lateness_fig, fig5, fig6])  # Add remaining figures
            
            transfer_fig = column(transfer_figs)


        inner_col_list = []

        if ENABLE_DU_CCDF:
            if HAS_DU_RX_SRS_TIMES:
                inner_col_list.append(row([column([fig_legend_ulu, fig_ccdf_ulu]), 
                                           column([fig_legend_srs, fig_ccdf_srs])]))
            else:
                inner_col_list.append(column([fig_legend_ulu, fig_ccdf_ulu]))

        if ENABLE_TRANSFER_FIG:
            inner_col_list.append(transfer_fig)

        if ENABLE_FIG1:
            inner_col_list.append(fig1)

        if ENABLE_RU_ENQUEUE_TIMELINE:
            inner_col_list.append(fig2)
            inner_col_list.append(column([fig_enqueue_by_symbol_legend, fig_enqueue_by_symbol_ccdf]))
            inner_col_list.append(column([fig_enqueue_by_channel_legend, fig_enqueue_by_channel_ccdf]))

        col_list.append(column(inner_col_list))

        #If enabled, make sure that all enqueued RU TX times are in order
        if que_check:
            if HAS_RU_TX_TIMES:
                que_list = list(set(df_tx_timing.que_index))
                que_list.sort()
                for que in que_list:
                    current_data = df_tx_timing[df_tx_timing.que_index==que].reset_index().sort_values(by='enqueue_time')
                    yy = current_data.tx_time.diff() / 1000.

                    if(any(yy<-0.005)):
                        print("Que %i.... FAILED!!!!!!!!!!!!!!"%que)
                    else:
                        print("Que %i.... Success."%que)
            else:
                print("No TX Times, not running verification step ")     
    
    return row(col_list)


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
        "-p", "--per_symbol", action="store_true", help="Enable per-symbol plot"
    )
    parser.add_argument(
        "-d", "--add_downlink", action="store_true", help="Enable downlink on timeline plot"
    )
    parser.add_argument(
        "-q", "--que_check", action="store_true", help="Perform que ordering check (RU TX)"
    )
    parser.add_argument(
        "-l", "--labels", type=str, nargs="+", help="Names for each phy log"
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
        "-e", "--mmimo_enable", action="store_true", help="Modifies timeline requirement to mmimo settings"
    )
    parser.add_argument(
        "-c", "--cell_symbol_index", action="store_true", help="Enable cell+symbol index for y-axis in ULC and DLC timeline plots"
    )
    args = parser.parse_args()

    main(args)
