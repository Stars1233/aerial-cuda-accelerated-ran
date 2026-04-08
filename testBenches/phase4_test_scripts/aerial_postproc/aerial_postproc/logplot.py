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

from cmath import nan
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.palettes import d3
from bokeh.models import Range1d, ColumnDataSource, DatetimeTickFormatter, Span, Label, CustomJS, LinearColorMapper, ColorBar, DataTable, TableColumn, NumberFormatter, Legend, FactorRange
from bokeh.layouts import column, row
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Label,
    LabelSet,
    LinearColorMapper,
    Range1d,
    Span,
)
from bokeh.palettes import Category10, d3, Category20
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.palettes import Category10
from bokeh.palettes import Set1
from bokeh.models import LabelSet, HoverTool
import pandas as pd

import datetime
import bisect
import copy

from aerial_postproc.logparse import NUM_20SLOT_GROUPS

def get_ref_t0(tir,t0_timestamp):
    """
    Utility function to back out reference time used by log parser
    """
    return int(t0_timestamp - tir * 1e9)

def plot_task_extract(p, glyph_color, data, task, width):

    filtered_data = data[data.task == task]
    source = ColumnDataSource(filtered_data)

    p.vbar(x='t0_datetime', width=width, bottom='start_deadline', top='end_deadline', source=source, color=glyph_color, legend_label=task, alpha=0.5)

def add_start_time_label(p):
    start_time_label = Label(
        x=0,
        y=0,
        x_units="screen",
        y_units="screen",
        text="Unknown Start Time",
        render_mode="css",
        border_line_color="black",
        border_line_alpha=1.0,
        background_fill_color="white",
        background_fill_alpha=1.0,
    )
    p.add_layout(start_time_label)

    p.x_range.js_on_change(
        "start",
        CustomJS(
            args=dict(other=start_time_label),
            code="""
                      var start_date = new Date(cb_obj.start)
                      other.text = start_date.toISOString()
                      """,
        ),
    )


def plot_all_tasks(df, df_ru=None):
    TOOLTIPS = [
        ("task", "@task"),
        ("subtask", "@subtask"),
        ("(sfn,slot)", "(@sfn, @slot)"),
        ("log timestamp", "@log_timestamp"),
        ("t0 timestamp", "@t0_formatted"),
        ("start deadline", "@start_deadline"),
        ("end deadline", "@end_deadline"),
        ("duration (ns)", "@duration"),
    ]

    p = figure(
        title="Profiling Results - Deadline view",
        x_axis_type="datetime",
        tooltips=TOOLTIPS,
        tools="tap,crosshair,hover,xwheel_zoom,ywheel_zoom,xwheel_pan,ywheel_pan,box_zoom,reset",
        plot_width=900,
        plot_height=700,
    )
    p.add_layout(Legend(), 'right')

    tasks = df.task.unique()
    num_colors = len(tasks)
    num_colors = max(3, len(tasks))
    num_colors = min(20, num_colors)
    for task_idx, task in enumerate(tasks):
        task_color = d3["Category20b"][num_colors][task_idx % num_colors]
        plot_task_extract(p, task_color, df, task, 0.3)

    if df_ru is not None and df_ru.empty is False:
        plot_task_extract(p, "red", df_ru, "RU DL U Plane", 0.2)
        u_plane_early_limit = Span(
            location=-345,
            dimension="width",
            line_color="black",
            line_dash="dashed",
            line_width=3,
        )
        p.add_layout(u_plane_early_limit)
        u_plane_late_limit = Span(
            location=-294,
            dimension="width",
            line_color="black",
            line_dash="dashed",
            line_width=3,
        )
        p.add_layout(u_plane_late_limit)

    add_start_time_label(p)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.xaxis.axis_label = "Slot T0 Time"
    p.yaxis.axis_label = "Time from T0 (us)"

    return p


def plot_task_extract_timeline(p, glyph_color, data, task, y_loc):

    filtered_data = data[data.task == task]
    source = ColumnDataSource(filtered_data)

    if y_loc >= 0:
        y_loc = "cpu"

    p.hbar(
        y=y_loc,
        height=0.8,
        left="start_datetime",
        right="end_datetime",
        source=source,
        color=glyph_color,
        legend_label=task,
    )


def plot_all_tasks_timeline(df, df_ru=None, x_range=None):
    TOOLTIPS = [
        ("task", "@task"),
        ("subtask", "@subtask"),
        ("(sfn,slot)", "(@sfn, @slot)"),
        ("log_timestamp", "@log_timestamp"),
        ("t0 timestamp", "@t0_formatted"),
        ("duration (us)", "@duration"),
        ("start_deadline (us)", "@start_deadline"),
        ("end_deadline (us)", "@end_deadline"),
        ("cpu", "@cpu"),
    ]

    p = figure(
        title="Profiling Results - Timeline view",
        x_axis_type="datetime",
        tooltips=TOOLTIPS,
        tools="tap,crosshair,hover,xwheel_zoom,ywheel_zoom,xwheel_pan,ywheel_pan,box_zoom,reset",
        plot_width=900,
        plot_height=700,
        x_range=x_range,
    )
    p.add_layout(Legend(), 'right')

    tasks = df.task.unique()
    num_colors = len(tasks)
    num_colors = max(3, len(tasks))
    num_colors = min(20, num_colors)
    for task_idx, task in enumerate(tasks):
        task_color = d3["Category20b"][num_colors][task_idx % num_colors]
        plot_task_extract_timeline(p, task_color, df, task, task_idx)

    if df_ru is not None and df_ru.empty is False:
        plot_task_extract_timeline(p, "red", df_ru, "RU DL U Plane", -1)
        plot_task_extract_timeline(p, "orange", df_ru, "RU DL C Plane", -2)
        plot_task_extract_timeline(p, "blue", df_ru, "RU UL C Plane", -3)

    add_start_time_label(p)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = "Task ID"

    return p


def timeline_plot_generic(input_df,y_key,title="",height=600,width=1600,x_range=None,tooltips=None,crosshair_tool=None):
    """
    Generates a timeline plot, assuming the presence of start_datetime and end_datetime keywords
    """
    cols = input_df.columns

    DATETIME_PLOT = (y_key in cols) and ('start_datetime' in cols) and ('end_datetime' in cols)
    TIR_PLOT = (y_key in cols) and ('start_tir' in cols) and ('end_tir' in cols)

    if(DATETIME_PLOT):
        fig1 = figure(title=title,
                    x_axis_type="datetime",
                    tooltips=tooltips,
                    tools="tap,crosshair,hover,xwheel_zoom,ywheel_zoom,xwheel_pan,ywheel_pan,box_zoom,reset",
                    plot_width=width, plot_height=height,
                    x_range=x_range)
    elif(TIR_PLOT):
        fig1 = figure(title=title,
                    tooltips=tooltips,
                    plot_width=width, plot_height=height,
                    x_range=x_range)
    else:
        print("Unknown keys seen in timeline_plot_generic: %s"%input_df.columns)
        return None
    
    y_vals = list(input_df[y_key].unique())
    y_vals.sort()
    temp_df = input_df.copy()
    temp_df['yy'] = temp_df[y_key].apply(y_vals.index)

    if(DATETIME_PLOT):
        for ii, y_val in enumerate(y_vals):
            source = ColumnDataSource(temp_df[temp_df[y_key]==y_val])
            color = color_select(ii,10)
            fig1.hbar(y='yy', height=0.8, left='start_datetime', right='end_datetime', source=source, color=color, line_width=1, line_color='black', legend_label='(%i) %s'%(ii,y_val))

        fig1.legend.location = "top_right"
        fig1.legend.click_policy = "hide"
        fig1.xaxis.axis_label = 'Time'
        fig1.yaxis.axis_label = '%s'%y_key

    elif(TIR_PLOT):
        for ii, y_val in enumerate(y_vals):
            source = ColumnDataSource(temp_df[temp_df[y_key]==y_val])
            color = color_select(ii,10)
            fig1.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source, color=color, line_width=1, line_color='black', legend_label='(%i) %s'%(ii,y_val))

        fig1.legend.location = "top_right"
        fig1.legend.click_policy = "hide"
        fig1.xaxis.axis_label = 'Time'
        fig1.yaxis.axis_label = '%s'%y_key

    if(crosshair_tool is not None):
        fig1.add_tools(crosshair_tool)

    return fig1

def plot_all_tasks_cpu_timeline(ti_subtask_df, l2_df, tick_df, testmac_df, x_range=None, traces_only=False, cpu_only=False, crosshair_tool=None, ul_only=False):
    """
    Generates a timeline plot that includes CPU utilization estimates

    trace_only - Generates only traces
                Note: traces can lead to large/slow html files
    cpu_only - Generates only CPU utilization graphs
                Note: cpu utilization calculation is not very fast (needs work)

    Note: if both traces_only and cpu_only are set, traces_only overrides

    Returns a bokeh column containing the plots
    """
    # For timeline plot
    TOOLTIPS = [
        ("task", "@task"),
        ("subtask", "@subtask"),
        ("(sfn,slot)", "(@sfn, @slot)"),
        ("log_timestamp", "@log_timestamp"),
        ("t0 timestamp", "@t0_formatted"),
        ("duration (us)", "@duration"),
        ("cpu", "@cpu"),
    ]

    if(not cpu_only):
        fig1 = figure(title="CPU Timeline view",
                    x_axis_type="datetime",
                    tooltips=TOOLTIPS,
                    tools="tap,crosshair,hover,xwheel_zoom,ywheel_zoom,xwheel_pan,ywheel_pan,box_zoom,reset",
                    plot_width=1600, plot_height=600,
                    x_range=x_range)
        
    if(ul_only):
        temp_df = ti_subtask_df[ti_subtask_df.task.str.startswith("UL Task")]
    else:
        temp_df = ti_subtask_df

    # Make copies so we can mess with the dataframe
    task_df = temp_df[temp_df.task.str.startswith("DL Task") | temp_df.task.str.startswith("UL Task") | temp_df.task.str.startswith("Debug Task")].copy(deep=True)
    dl_comms_df = temp_df[temp_df.task.str.startswith("DL Task GPU Comms")].copy(deep=True)
    dl_comp_df = temp_df[temp_df.task.str.startswith("DL Task Compression")].copy(deep=True)
    enable_l2 = len(l2_df) > 0
    if(enable_l2):
        l2_copy_df = l2_df.copy(deep=True)
    enable_tick = len(tick_df) > 0
    if(enable_tick):
        tick_copy_df = tick_df.copy(deep=True)
    enable_testmac = len(testmac_df) > 0
    if(enable_testmac):
        testmac_copy_df = testmac_df.copy(deep=True)



    # Determine cpu options
    cpus = list(temp_df.cpu.unique())
    cpus.sort()

    # Determine number lines per cpu based on max sequence number
    sequence_nums = list(dl_comms_df.sequence.unique())
    sequence_nums.extend(dl_comp_df.sequence.unique())
    sequence_nums = list(set(sequence_nums))
    sequence_nums.sort()
    lines_per_cpu = 1+len(sequence_nums)

    DL_TASK_COLOR = 'red'
    UL_TASK_COLOR = 'green'
    DL_COMMS_COLOR = 'blue'
    DL_COMP_COLOR = 'yellow'
    DL_DEBUG_TASK_COLOR = 'pink'
    L2_COLOR = 'orange'
    TICK_COLOR = 'black'
    TESTMAC_COLOR = 'brown'

    # Set yy values for tasks
    task_df["yy"] = [cpus.index(aa) for aa in task_df.cpu]

    # Set yy values for l2
    if(enable_l2):
        l2_copy_df.loc[l2_copy_df.slot%2 == 0,"yy"] = len(cpus)
        l2_copy_df.loc[l2_copy_df.slot%2 == 1,"yy"] = len(cpus)+1

    if(enable_tick):
        tick_copy_df["yy"] = len(cpus)+3

    if(enable_testmac):
        testmac_copy_df["yy"] = len(cpus)+2

    # Split out DL vs UL
    dl_task_df = task_df[task_df.task.str.startswith("DL Task")]
    ul_task_df = task_df[task_df.task.str.startswith("UL Task")]
    debug_task_df = task_df[task_df.task.str.startswith("Debug Task")]

    # Set yy value for comms tasks
    dl_comms_df['yy'] = [(cpus.index(aa)*lines_per_cpu + 1 + sequence_nums.index(bb))/lines_per_cpu for aa,bb in zip(dl_comms_df.cpu, dl_comms_df.sequence)]
    dl_comp_df['yy'] = [(cpus.index(aa)*lines_per_cpu + 1 + sequence_nums.index(bb))/lines_per_cpu for aa,bb in zip(dl_comp_df.cpu, dl_comp_df.sequence)]

    if(not cpu_only):
        # Create the figure with both tasks and dl comms
        source1 = ColumnDataSource(dl_task_df)
        source2 = ColumnDataSource(ul_task_df)
        source3 = ColumnDataSource(dl_comms_df)
        source4 = ColumnDataSource(dl_comp_df)
        source5 = ColumnDataSource(debug_task_df)
        fig1.hbar(y='yy', height=0.8/lines_per_cpu, left='start_datetime', right='end_datetime', source=source1, color=DL_TASK_COLOR, line_width=1, line_color='black', legend_label='DL CPU Tasks')
        fig1.hbar(y='yy', height=0.8/lines_per_cpu, left='start_datetime', right='end_datetime', source=source2, color=UL_TASK_COLOR, line_width=1, line_color='black', legend_label='UL CPU Tasks')
        fig1.hbar(y='yy', height=0.8/lines_per_cpu, left='start_datetime', right='end_datetime', source=source3, color=DL_COMMS_COLOR, line_width=1, line_color='black', legend_label='DL Comm Task Breakout')
        fig1.hbar(y='yy', height=0.8/lines_per_cpu, left='start_datetime', right='end_datetime', source=source4, color=DL_COMP_COLOR, line_width=1, line_color='black', legend_label='DL Compression Breakout')
        fig1.hbar(y='yy', height=0.8/lines_per_cpu, left='start_datetime', right='end_datetime', source=source5, color=DL_DEBUG_TASK_COLOR, line_width=1, line_color='black', legend_label='Debug Task')
        if(enable_l2):
            source6 = ColumnDataSource(l2_copy_df)
            fig1.hbar(y='yy', height=0.8/lines_per_cpu, left='start_datetime', right='end_datetime', source=source6, color=L2_COLOR, line_width=1, line_color='black', legend_label='L2A Processing')
        if(enable_tick):
            source7 = ColumnDataSource(tick_copy_df)
            fig1.hbar(y='yy', height=0.8/lines_per_cpu, left='cpu_datetime', right='cpu_datetime', source=source7, color=TICK_COLOR, line_width=1, line_color='black', legend_label='Tick')
        if(enable_testmac):
            source8 = ColumnDataSource(testmac_copy_df)
            fig1.hbar(y='yy', height=0.8/lines_per_cpu, left='start_datetime', right='end_datetime', source=source8, color=TESTMAC_COLOR, line_width=1, line_color='black', legend_label='Testmac FAPI Send')

        # Configure figure
        add_start_time_label(fig1)
        fig1.legend.location = "top_right"
        fig1.legend.click_policy = "hide"
        fig1.xaxis.axis_label = 'Time'
        fig1.yaxis.axis_label = 'CPU Index'
        if(crosshair_tool is not None):
            fig1.add_tools(crosshair_tool)

    if(traces_only):
        return column([fig1])


    # For utilization plot
    TOOLTIPS2 = [
        ("slot", "@slot"),
        ("(x,y)", "($x, $y)"),
    ]

    # Set x range appropriately
    if(not cpu_only):
        fig2_xrange = fig1.x_range
    else:
        fig2_xrange = x_range

    fig2 = figure(title="CPU Utilization",
                x_axis_type="datetime",
                tooltips=TOOLTIPS2,
                tools="tap,crosshair,hover,xwheel_zoom,ywheel_zoom,xwheel_pan,ywheel_pan,box_zoom,reset",
                plot_width=1600, plot_height=300,
                x_range=fig2_xrange)

    # Determines the total duration of overlap of the df row to the start/end datetimes
    # Value returned in nanoseconds
    def sum_overlaps_ns(df, start_timestamp, end_timestamp):
        total_accum_ns = 0

        # Check partial overlaps on right side (or completely inside)
        condition1 = (df.start_timestamp > start_timestamp) & (
            df.start_timestamp < end_timestamp
        )
        data1 = df[condition1]
        total_accum_ns += np.sum(
            np.minimum(data1.end_timestamp, end_timestamp) - data1.start_timestamp
        )

        # Check partial overlaps on left side (and not completely inside)
        condition2 = (
            (df.end_timestamp > start_timestamp)
            & (df.end_timestamp < end_timestamp)
            & ~(condition1)
        )
        data2 = df[condition2]
        total_accum_ns += np.sum(data2.end_timestamp - start_timestamp)

        # Check full overlaps
        condition3 = (df.start_timestamp < start_timestamp) & (
            df.end_timestamp > end_timestamp
        )
        data3 = df[condition3]
        total_accum_ns += np.sum(
            np.minimum(data3.end_timestamp, end_timestamp)
            - np.maximum(data3.start_timestamp, start_timestamp)
        )

        return total_accum_ns

    from aerial_postproc.logparse import GH_4TR_DLU_WINDOW_OFFSET1
    SLOT_NS = 500000
    DLU_WINDOW_OFFSET1_NS = GH_4TR_DLU_WINDOW_OFFSET1*1e3
    DL_WINDOW_NS = 2*SLOT_NS + DLU_WINDOW_OFFSET1_NS
    UL_WINDOW_NS = 2*SLOT_NS
    min_t0 = task_df.t0_timestamp.min()
    max_t0 = task_df.t0_timestamp.max()
    start_slot = task_df[task_df.t0_timestamp == min_t0].slot.iloc[0]
    num_slots = int(np.ceil((max_t0 - min_t0) / SLOT_NS + 1))

    print("Calculating CPU percentages...")
    t0_datetimes = []
    dlu_window_datetimes = []
    slots = []
    dl_task_percentages = []
    ul_task_percentages = []
    dl_comms_percentages = []
    dl_comp_percentages = []
    dl_task_nowait_percentages = []
    ul_task_nowait_percentages = []
    current_t0 = min_t0
    current_slot = start_slot
    dl_task_df_nowait = dl_task_df[(~dl_task_df.subtask.str.startswith("Wait Slot End Task")) & (~dl_task_df.subtask.str.startswith("Wait Slot Channel End")) & (~dl_task_df.subtask.str.startswith("Compression Wait"))]
    ul_task_df_nowait = ul_task_df[(~ul_task_df.subtask.str.startswith("Wait Channel End Task")) & (~ul_task_df.subtask.str.startswith("Wait Order")) & (~ul_task_df.subtask.str.startswith("Wait PUSCH")) & (~ul_task_df.subtask.str.startswith("Wait PUCCH")) & (~ul_task_df.subtask.str.startswith("Wait PRACH"))]
    for ii in range(num_slots):
        if ii % 1000 == 0:
            print("%i of %i slots..." % (ii, num_slots))
        t0_datetimes.append(pd.to_datetime(current_t0,unit='ns'))
        dlu_window_datetimes.append(pd.to_datetime(current_t0+DLU_WINDOW_OFFSET1_NS,unit='ns'))
        slots.append(current_slot)
        dl_task_percentages.append(sum_overlaps_ns(dl_task_df, current_t0-2*SLOT_NS, current_t0+DLU_WINDOW_OFFSET1_NS) / DL_WINDOW_NS * 100)
        ul_task_percentages.append(sum_overlaps_ns(ul_task_df, current_t0-2*SLOT_NS, current_t0) / UL_WINDOW_NS * 100)
        dl_comms_percentages.append(sum_overlaps_ns(dl_comms_df, current_t0-2*SLOT_NS, current_t0+DLU_WINDOW_OFFSET1_NS) / DL_WINDOW_NS * 100)
        dl_comp_percentages.append(sum_overlaps_ns(dl_comp_df, current_t0-2*SLOT_NS, current_t0+DLU_WINDOW_OFFSET1_NS) / DL_WINDOW_NS * 100)

        dl_task_nowait_percentages.append(sum_overlaps_ns(dl_task_df_nowait, current_t0-2*SLOT_NS, current_t0+DLU_WINDOW_OFFSET1_NS) / DL_WINDOW_NS * 100)
        ul_task_nowait_percentages.append(sum_overlaps_ns(ul_task_df_nowait, current_t0-2*SLOT_NS, current_t0) / UL_WINDOW_NS * 100)

        current_t0 += SLOT_NS
        current_slot += 1
        current_slot %= NUM_20SLOT_GROUPS*20

    source4 = ColumnDataSource({
        't0_datetime': t0_datetimes,
        'dlu_window_datetime': dlu_window_datetimes,
        'slot': slots,
        'dl_task_percentage':dl_task_percentages,
        'ul_task_percentage': ul_task_percentages,
        'dl_comms_percentage': dl_comms_percentages,
        'dl_comp_percentage': dl_comp_percentages,
        'dl_task_nowait_percentages': dl_task_nowait_percentages,
        'ul_task_nowait_percentages': ul_task_nowait_percentages
        })

    fig2.line(y='dl_task_percentage', x='dlu_window_datetime', source=source4, color=DL_TASK_COLOR, legend_label="DL Task CPU %")
    fig2.circle(y='dl_task_percentage', x='dlu_window_datetime', source=source4, color=DL_TASK_COLOR, legend_label="DL Task CPU %")
    fig2.line(y='ul_task_percentage', x='t0_datetime', source=source4, color=UL_TASK_COLOR, legend_label="UL Task CPU %")
    fig2.line(y='dl_comms_percentage', x='dlu_window_datetime', source=source4, color=DL_COMMS_COLOR, legend_label="DL Comms CPU %")
    fig2.line(y='dl_comp_percentage', x='dlu_window_datetime', source=source4, color=DL_COMP_COLOR, legend_label="DL Comp CPU %")
    fig2.line(y='dl_task_nowait_percentages', x='dlu_window_datetime', source=source4, color='orange', legend_label="DL Task NoWait CPU %")
    fig2.line(y='ul_task_nowait_percentages', x='t0_datetime', source=source4, color='purple', legend_label="UL Task NoWait CPU %")

    # # Code for adding CPU percentages for smaller intervals in a contiguous manner
    # print("Calculating fine time percentages...")
    # FINE_INTERVAL_NS = 100000
    # start_timestamp = min(dl_task_df.t0_timestamp)
    # end_timestamp = max(dl_task_df.t0_timestamp)
    # timestamp_list = []
    # percentage_list = []
    # current_timestamp = start_timestamp
    # while(current_timestamp <= end_timestamp):
    #     percentage_list.append(sum_overlaps_ns(dl_task_df,current_timestamp-FINE_INTERVAL_NS,current_timestamp) / FINE_INTERVAL_NS * 100)
    #     timestamp_list.append(current_timestamp)
    #     current_timestamp += FINE_INTERVAL_NS
    # datetime_list = [datetime.datetime.fromtimestamp(aa/1e9) for aa in timestamp_list]
    # source5 = ColumnDataSource({
    #     'fine_datetimes': datetime_list,
    #     'fine_percentages': percentage_list
    # })
    # fig2.line(y='fine_percentages', x='fine_datetimes', source=source5, color='black', legend_label="Fine DL CPU Percent (interval=%ius)"%(FINE_INTERVAL_NS/1000))

    #Create a data table with summary statistics
    utilization_data = [dl_task_percentages,ul_task_percentages,dl_comms_percentages,dl_comp_percentages,dl_task_nowait_percentages,ul_task_nowait_percentages]
    utilization_names = ["All DL", "All UL", "DL Comms", "DL Compression", "All DL No Wait", "All UL No Wait"]

    q99_list = []
    q50_list = []
    avg_list = []
    for aa,bb in zip(utilization_data,utilization_names):
        q99_list.append(np.percentile(aa,99))
        q50_list.append(np.percentile(aa,50))
        avg_list.append(np.mean(aa))


    table_dict = {'name':utilization_names,
                  'q99':q99_list,
                  'q50':q50_list,
                  'avg':avg_list}
    table_source = ColumnDataSource(table_dict)
    columns = [
        TableColumn(field='name', title='name'),
        TableColumn(field='q99', title='q99', formatter=NumberFormatter(format="0.0")),
        TableColumn(field='q50', title='q50', formatter=NumberFormatter(format="0.0")),
        TableColumn(field='avg', title='avg', formatter=NumberFormatter(format="0.0")),
    ]
    table1 = DataTable(source=table_source, columns=columns)

    add_start_time_label(fig2)
    fig2.legend.location = "top_right"
    fig2.legend.click_policy = "hide"
    fig2.xaxis.axis_label = 'Time'
    fig2.yaxis.axis_label = 'CPU Percentage'

    if(crosshair_tool is not None):
        fig2.add_tools(crosshair_tool)

    if(not cpu_only):
        return column([fig1,row(fig2,table1)])
    else:
        return column([row(fig2,table1)])


# Generates a vertical bar plot that contains GPU runtimes
def plot_gpu_data(df_gpu, field="duration", title="GPU Task Durations"):
    # Group by slot+task, take average across all fields
    groupings = df_gpu.groupby(["slot", "task"])[
        [
            "cuda_setup_duration",
            "cuda_run_duration",
            "max_gpu_duration",
            "wait_gpu_duration",
            "gpu_setup1_duration",
            "gpu_setup2_duration",
            "gpu_run_duration",
            "gpu_total_duration",
            "execution_diff",
        ]
    ]
    gpu_perf_data = groupings.mean().reset_index()

    # Add several fields
    # gpu_perf_data = gpu_perf_data.assign(
    #     duration_q01=groupings.quantile(0.01).reset_index().max_gpu_duration,
    #     duration_q50=groupings.quantile(0.50).reset_index().max_gpu_duration,
    #     duration_q99=groupings.quantile(0.99).reset_index().max_gpu_duration,
    #     duration_max=groupings.max().reset_index().max_gpu_duration,
    #     delta_start_max=groupings.max().reset_index().delta_start,
    #     proc_deadline=-1000.0,
    #     slot_time=0
    # )

    fig = figure(title=title)
    fig.x_range = Range1d(gpu_perf_data.slot.min() - 1, gpu_perf_data.slot.max() + 1)
    fig.y_range = Range1d(0, 1500)
    fig.plot_height = 400
    fig.plot_width = 400

    task_list = list(set(gpu_perf_data.task))
    task_list.sort()

    for ii, task in enumerate(task_list):
        source = ColumnDataSource(gpu_perf_data[gpu_perf_data.task == task])
        spacing = 0.6 / len(task_list)
        center = -0.3 + ii * spacing
        fig.vbar(
            x=dodge("slot", center, range=fig.x_range),
            top=field,
            width=spacing,
            source=source,
            color=Category10[max(3, len(task_list))][ii],
            legend_label=task,
        )

    fig.yaxis.axis_label = "duration (us)"
    fig.xaxis.axis_label = "slot"
    fig.legend.click_policy = "hide"

    return fig


def plot_pdsch_deadlines(df_gpu):
    TOOLTIPS = [("sfn", "@sfn"), ("slot", "@slot")]

    fig = figure(title="PDSCH Deadlines", tooltips=TOOLTIPS)
    fig.y_range = Range1d(-1500, 500)
    fig.plot_height = 400
    fig.plot_width = 1200

    # Prepare sfn/slot sorted PDSCH data and place in column data source
    pdsch_df = df_gpu[df_gpu.task == "PDSCH Aggr"]
    pdsch_df.sort_values(by=["sfn", "slot"])
    pdsch_df.reset_index()
    source1 = ColumnDataSource(pdsch_df)

    # Add field representing max_gpu_end[ii] - gpu_setup1_duration - gpu_setup2_duration - gpu_run_duration
    source1.data["deadline1"] = (
        source1.data["max_gpu_end_deadline"]
        - source1.data["gpu_setup1_duration"]
        - source1.data["gpu_setup2_duration"]
        - source1.data["gpu_run_duration"]
    )

    # Add field representing max(max_gpu_end[ii-1],max_gpu_start[ii])
    temp = [
        (
            max(
                source1.data["max_gpu_start_timestamp"][ii],
                source1.data["max_gpu_end_timestamp"][ii - 1],
            )
            - source1.data["t0_timestamp"][ii]
        )
        / 1000.0
        for ii in range(1, len(source1.data["max_gpu_start_timestamp"]))
    ]
    temp.insert(0, np.nan)
    source1.data["deadline2"] = temp

    # Prepare sfn/slot sorted COMPRESSION data
    compression_df = df_gpu[df_gpu.task == "COMPRESSION DL"]
    compression_df.sort_values(by=["sfn", "slot"])
    compression_df.reset_index()
    source2 = ColumnDataSource(compression_df)

    # Add field representing previous compression timeline
    temp = [
        (
            source2.data["max_gpu_end_timestamp"][ii - 1]
            - source2.data["t0_timestamp"][ii]
        )
        / 1000.0
        for ii in range(1, len(source2.data["max_gpu_end_deadline"]))
    ]
    temp.insert(0, np.nan)
    source2.data["deadline2"] = temp

    # Add line representing t0
    fig.line(
        x=[min(df_gpu.tir), max(df_gpu.tir)],
        y=[0, 0],
        line_dash="dashed",
        color="black",
    )

    # These are the PDSCH fields
    field_list1 = [
        "max_gpu_end_deadline",
        "deadline1",
        "deadline2",
        "max_gpu_start_deadline",
    ]
    label_list1 = [
        "max_gpu_end[i]",
        "max_gpu_end[i] - gpu_setup[i] - gpu_run[i]",
        "max(max_gpu_end[i-1],max_gpu_start[i])",
        "max_gpu_start[i]",
    ]
    num_fields1 = len(field_list1)

    # These are the compression fields
    field_list2 = ["max_gpu_end_deadline", "deadline2"]
    label_list2 = ["compression_max_gpu_end[ii]", "compression_max_gpu_end[ii-1]"]
    num_fields2 = len(field_list2)

    num_colors = min(10, max(3, num_fields1 + num_fields2))

    # Add PDSCH data
    for ii in range(num_fields1):
        fig.circle(
            x="tir",
            y=field_list1[ii],
            color=Category10[num_colors][ii % 10],
            source=source1,
            legend_label=label_list1[ii],
        )

    # Add compression data
    for ii in range(num_fields2):
        fig.circle(
            x="tir",
            y=field_list2[ii],
            color=Category10[num_colors][(ii + num_fields1) % 10],
            source=source2,
            legend_label=label_list2[ii],
        )

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    fig.xaxis.axis_label = "Time in run (sec)"
    fig.yaxis.axis_label = "Time relative to T0"

    return fig


# Generates a 2D colored plot indicating late packet distributions
# x axis is symbol, y axis is slot
def plot_dl_late_packet_matrices(df_late_packet):
    dlc_df = df_late_packet[df_late_packet.task == "DL C Plane"]
    dlu_df = df_late_packet[df_late_packet.task == "DL U Plane"]
    ulc_df = df_late_packet[df_late_packet.task == "UL C Plane"]

    fig_list = []

    for current_df, current_title_postfix in zip(
        [dlc_df, dlu_df, ulc_df], ["DLC", "DLU", "ULC"]
    ):

        # Group by slot+symbol, summing in each bin
        groupings = current_df.groupby(["slot", "symbol"])["late_packet_count"]
        lp_df = groupings.sum().reset_index()
        lp_df["lp_percentage"] = 100.0 * (
            lp_df["late_packet_count"] / max(lp_df["late_packet_count"].sum(), 1)
        )

        # Create 2D box plot
        source1 = ColumnDataSource(lp_df)
        TOOLTIPS = [
            ("late_packet_count", "@late_packet_count"),
            ("lp_percentage", "@lp_percentage"),
        ]
        fig = figure(
            title="Percent Contribution to Late Packets, %s" % current_title_postfix,
            tooltips=TOOLTIPS,
        )
        lcm = LinearColorMapper(palette="Magma256", low=0, high=5)
        cb = ColorBar(color_mapper=lcm)
        fig.add_layout(cb, "right")
        fig.rect(
            x="symbol",
            y="slot",
            height=1,
            width=1,
            source=source1,
            fill_color={"field": "lp_percentage", "transform": lcm},
        )
        fig.xaxis.axis_label = "Symbol"
        fig.yaxis.axis_label = "Slot"
        fig.plot_height = 400
        fig.plot_width = 400
        fig.y_range = Range1d(-1,80)
        fig.x_range = Range1d(-1,14)
        fig_list.append(fig)

    return row(fig_list)


def plot_hist(p, line_color, data, legend_label):
    bins = np.linspace(-2000, 500, 501)
    bins[0] = -np.inf
    bins[-1] = np.inf
    hist, edges = np.histogram(data, bins=bins)
    hist = hist / np.sum(hist)
    edges = edges[0:-1]
    p.line(x=edges, y=hist, line_color=line_color, legend_label=legend_label)


import holoviews as hv
from holoviews import dim
hv.extension('bokeh')

def ti_boxplot(ti_df):

    TOOLTIPS = [
        ("subtask", "@subtask"),
        ("slot", "@slot"),
        ("duration", "@duration"),
    ]

    fig_list = []
    task_list = list(set(ti_df.task))
    task_list.sort()
    for task in task_list:
        title = "CPU Subtask Durations (%s)"%task
        current_df = ti_df[ti_df.task == task].sort_values(by='sequence')
        boxwhisker = hv.BoxWhisker(current_df, ['sequence','subtask','slot'], 'duration', label=title)
        boxwhisker.opts(show_legend=False, height=800, width=1600, box_fill_color='red')
        hover=HoverTool(tooltips=TOOLTIPS)
        boxwhisker.opts(tools=[hover])
        fig = hv.render(boxwhisker)
        fig.yaxis.axis_label = 'duration (usec)'
        fig.title.render_mode = 'css'
        fig_list.append(fig)

    title = "All CPU Subtasks"
    boxwhisker = hv.BoxWhisker(ti_df, ['task','subtask','slot'], 'duration', label=title)
    boxwhisker.opts(show_legend=False, height=800, width=len(task_list)*1600, box_fill_color='task', cmap='Set1')
    hover=HoverTool(tooltips=TOOLTIPS)
    boxwhisker.opts(tools=[hover])
    fig = hv.render(boxwhisker)
    fig.yaxis.axis_label = 'duration (usec)'
    fig.title.render_mode = 'css'
    fig_list.append(fig)

    return column(fig_list)

def ti_start_time_plot_slotted(ti_df):
    TOOLTIPS = [
        ("task", "@task"),
        ("subtask", "@subtask"),
        ("slot", "@slot"),
    ]

    filtered_df = ti_df[ti_df.task.str.startswith("DL Task") & (ti_df.subtask=="Start Task")]

    boxwhisker = hv.BoxWhisker(filtered_df, ['task','slot'], 'start_deadline', label="Task Start Times (Distributions)")
    boxwhisker.opts(show_legend=False, height=800, width=1600, box_fill_color='task', cmap='Set1')
    hover=HoverTool(tooltips=TOOLTIPS)
    boxwhisker.opts(tools=[hover])
    fig = hv.render(boxwhisker)
    fig.yaxis.axis_label = 'Deadline Time (us)'

    return fig

def l2times_plot_slotted(l2_df):
    TOOLTIPS = [
        ("slot", "@slot"),
    ]

    # This makes sure holoviews sorts the slot properly
    temp_df = l2_df.sort_values(by='slot')

    boxwhisker = hv.BoxWhisker(temp_df, ['slot'], 'start_deadline', label="TestMAC Completion Times")
    boxwhisker.opts(show_legend=False, height=400, width=1600, box_fill_color='red', cmap='Set1')
    hover=HoverTool(tooltips=TOOLTIPS)
    boxwhisker.opts(tools=[hover])
    fig1 = hv.render(boxwhisker)
    fig1.yaxis.axis_label = 'Deadline Time (us)'

    boxwhisker = hv.BoxWhisker(temp_df, ['slot'], 'end_deadline', label="L2A Completion Times")
    boxwhisker.opts(show_legend=False, height=400, width=1600, box_fill_color='red', cmap='Set1')
    hover=HoverTool(tooltips=TOOLTIPS)
    boxwhisker.opts(tools=[hover])
    fig2 = hv.render(boxwhisker)
    fig2.yaxis.axis_label = 'Deadline Time (us)'

    return column([fig1,fig2])

def ti_deadline_plot_raw(ti_df,l2_df):
    TOOLTIPS = [
        ("task", "@task"),
        ("subtask", "@subtask"),
        ("sfn", "@sfn"),
        ("frame", "@frame"),
        ("slot", "@slot"),
        ("duration", "@duration"),
    ]

    fig = figure(title="Task Deadline Plot (Raw)", tooltips=TOOLTIPS, height=800, width=1600)

    # Plot the completion times based on
    debug_task_name = 'Debug Task'
    trigger_complete_subtask_name = 'Trigger synchronize'
    current_df = ti_df[(ti_df.task == debug_task_name) & (ti_df.subtask == trigger_complete_subtask_name)]
    if(len(current_df) > 0):
        current_source = ColumnDataSource(current_df)
        fig.circle(x='tir',y='end_deadline',source=current_source, color='red', line_color='yellow', legend_label='Trigger Kernel Completed')

    # Plot the raw deadline values for all "Start Task" subtasks (DL only)
    filtered_df = ti_df[ti_df.task.str.startswith("DL Task")]
    task_list = list(set(filtered_df.task))
    num_colors = min(max(3,len(task_list)),10)
    for ii,task in enumerate(task_list):
        current_df = filtered_df[(filtered_df.task==task) & (filtered_df.subtask=="Start Task")]
        current_source = ColumnDataSource(current_df)
        fig.circle(x='tir',y='start_deadline',source=current_source,color=Category10[num_colors][ii%num_colors],legend_label=task)

    #Add in compression completion time
    current_df = filtered_df[(filtered_df.task=="DL Task GPU Comms") & (filtered_df.subtask=="Compression Wait") & (filtered_df.slot%10 == 9)]
    current_source = ColumnDataSource(current_df)
    fig.circle(x='tir',y='end_deadline',source=current_source,color='purple',line_color='black',size=9,legend_label='Compression Completed, *9')
    current_df = filtered_df[(filtered_df.task=="DL Task GPU Comms") & (filtered_df.subtask=="Compression Wait") & (filtered_df.slot%10 == 6)]
    current_source = ColumnDataSource(current_df)
    fig.circle(x='tir',y='end_deadline',source=current_source,color='yellow',line_color='black',size=9,legend_label='Compression Completed, *6')
    current_df = filtered_df[(filtered_df.task=="DL Task GPU Comms") & (filtered_df.subtask=="Compression Wait") & (filtered_df.slot%10 != 9) & (filtered_df.slot%10 != 6)]
    current_source = ColumnDataSource(current_df)
    fig.circle(x='tir',y='end_deadline',source=current_source,color='green',line_color='black',size=9,legend_label='Compression Completed, other')


    # Plot the raw deadline values for start/end of L2 processing (DL only)
    if(len(l2_df) > 0):
        current_source = ColumnDataSource(l2_df[l2_df.dl_slot==True])
        fig.circle(x='tir',y='start_deadline',source=current_source,color='black',legend_label='TestMAC Finished')
        fig.circle(x='tir',y='end_deadline',source=current_source,color='blue',legend_label='L2A Finished')

    fig.legend.click_policy = "hide"
    fig.yaxis.axis_label = "Deadline Time (usec)"
    fig.xaxis.axis_label = "Time In Run (sec)"

    return fig

def late_slot_l2_plot(l2_df,lp_slot_df):
    import pandas as pd

    TOOLTIPS = [
            ("slot", "@slot"),
            ("end_deadline", "@end_deadline"),
            ("late_packet_count", "@late_packet_count"),
        ]

    lp_dlu_df = lp_slot_df[lp_slot_df.task=='DL U Plane']

    cells = list(set(lp_dlu_df.cell))
    cells.sort()
    fig1 = figure(title="Late Packets vs L2A Completion", tooltips=TOOLTIPS, height=800, width=800)
    fig2 = figure(title="Late Packets vs Slot", tooltips=TOOLTIPS, height=800, width=800)
    num_colors = min(max(3,len(cells)),10)
    for ii,cell in enumerate(cells):
        merged_df = pd.merge(l2_df,lp_dlu_df[lp_dlu_df.cell==cell][['t0_timestamp','late_packet_count']],on=["t0_timestamp"],validate="one_to_one")
        source = ColumnDataSource(merged_df)
        fig1.circle(x='end_deadline',y='late_packet_count',source=source,color=Category10[num_colors][ii%num_colors],legend_label="Cell%i"%cell)
        fig2.circle(x='slot',y='late_packet_count',source=source,color=Category10[num_colors][ii%num_colors],legend_label="Cell%i"%cell)


    source = ColumnDataSource(lp_slot_df)
    fig1.legend.click_policy = "hide"
    fig1.yaxis.axis_label = "Late Packet Count"
    fig1.xaxis.axis_label = "L2A Completion Deadline Time (usec)"
    fig2.legend.click_policy = "hide"
    fig2.yaxis.axis_label = "Late Packet Count"
    fig2.xaxis.axis_label = "Slot"

    return row([fig1,fig2])

def plot_late_slot_debug2(lp_slot_df,ti_df,symbol_timing_df, packet_timing_df):
    TOOLTIPS1 = [
            ("frame", "@frame"),
            ("slot", "@slot"),
            ("cell", "@cell"),
            ("late_packet_count", "@late_packet_count"),
        ]

    # Calculate reference t0 of phy log
    ref_t0 = ti_df.iloc[0].t0_timestamp - int(ti_df.iloc[0].tir*1e9)

    # Add plot that contain DLU late slots
    dlu_df = lp_slot_df[lp_slot_df.task=="DL U Plane"].copy()
    dlu_df['tir'] = (dlu_df.t0_timestamp - ref_t0)/1e9
    source = ColumnDataSource(dlu_df)
    fig1 = figure(title="Late DLU Packets vs Time", tooltips=TOOLTIPS1, height=400, width=1600)
    fig1.circle(x='tir', y='late_packet_count',source=source,color='black')
    fig1.yaxis.axis_label = "Late Packets"
    fig1.xaxis.axis_label = "Time In Run"

    # Add plot that contains end of trigger kernel time
    trigger_kernel_df = ti_df[(ti_df.task=="Debug Task") & (ti_df.subtask=="Trigger synchronize")]
    TOOLTIPS2 = [
        ("slot", "@slot"),
        ("sfn", "@sfn"),
    ]
    fig2 = figure(title="Trigger Kernel Deadlines", tooltips=TOOLTIPS2, height=400, width=1600)
    current_df = trigger_kernel_df[(trigger_kernel_df.slot != 39) & (trigger_kernel_df.slot%10 != 6)]
    current_source = ColumnDataSource(current_df)
    fig2.circle(x='tir',y='end_deadline',source=current_source,color='green',line_color='black',size=9,legend_label='Trigger Kernel Completed')
    current_df = trigger_kernel_df[(trigger_kernel_df.slot == 39)]
    current_source = ColumnDataSource(current_df)
    fig2.circle(x='tir',y='end_deadline',source=current_source,color='red',line_color='black',size=9,legend_label='Trigger Kernel Completed (slot 39)')

    fig2.legend.click_policy = "hide"
    fig2.yaxis.axis_label = "Deadline Time"
    fig2.xaxis.axis_label = "Time In Run"
    fig2.x_range = fig1.x_range

    TOOLTIPS3 = [
        ("frame", "@frame"),
        ("slot", "@slot"),
        ("cell", "@cell"),
        ("symbol", "@symbol"),
        ("duration", "@duration"),
    ]
    fig3 = figure(title="Symbol Timeline (RU)", tooltips=TOOLTIPS3, height=400, width=1600)
    temp_df = symbol_timing_df.copy()
    cell_list = list(set(temp_df.cell))
    num_cells = max(cell_list) + 1
    task_list = list(set(temp_df.task))
    task_list.sort()
    temp_df['yy'] = [task_list.index(aa)*num_cells + bb for aa,bb in zip(temp_df.task,temp_df.cell)]
    temp_df['color'] = 'orange'
    temp_df.loc[temp_df.task=='DL C Plane','color'] = 'blue'
    temp_df.loc[temp_df.task=='DL U Plane','color'] = 'red'
    temp_df.loc[temp_df.task=='UL C Plane','color'] = 'green'
    temp_df.loc[temp_df.is_late==True,'color'] = 'black'
    source1 = ColumnDataSource(temp_df[temp_df.task=='DL C Plane'])
    source2 = ColumnDataSource(temp_df[temp_df.task=='DL U Plane'])
    source3 = ColumnDataSource(temp_df[temp_df.task=='UL C Plane'])
    fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source1, color='color', line_width=1, line_color='black', legend_label='DL-C')
    fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source2, color='color', line_width=1, line_color='black', legend_label='DL-U')
    fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source3, color='color', line_width=1, line_color='black', legend_label='UL-C')
    fig3.yaxis.axis_label = "Cell/Task"
    fig3.xaxis.axis_label = "Time In Run"
    fig3.x_range = fig1.x_range

    # TOOLTIPS4 = [
    #     ("frame", "@frame"),
    #     ("slot", "@slot"),
    #     ("cell", "@cell"),
    #     ("symbol", "@symbol"),
    # ]
    # fig4 = figure(title="Symbol Timeline (DU)", tooltips=TOOLTIPS3, height=400, width=1600)

    # if(len(packet_timing_df) > 0):
    #     data1 = packet_timing_df[packet_timing_df.seq_num == 1][['t0_timestamp','symbol','cell','actual_start_time','frame','slot']].copy() #First packet
    #     data2 = packet_timing_df[packet_timing_df.seq_num == 2][['t0_timestamp','symbol','cell','actual_start_time']].copy()
    #     data2['actual_end_time'] = data2['actual_start_time']
    #     merged_df = pd.merge(data1,data2[['t0_timestamp','symbol','cell','actual_end_time']],on=["t0_timestamp","symbol","cell"],validate="one_to_one")
    #     merged_df['start_tir'] = (merged_df['actual_start_time'] - ref_t0) / 1e9
    #     merged_df['end_tir'] = (merged_df['actual_end_time'] - ref_t0) / 1e9
    #     merged_df['duration'] = (merged_df['end_tir'] - merged_df['start_tir'])*1e6
    #     source = ColumnDataSource(merged_df)
    #     fig4.hbar(y='cell', height=0.8, left='start_tir', right='end_tir', source=source, color='red',line_width=1, line_color='black', legend_label='DL-U')

    # fig4.yaxis.axis_label = "Cell"
    # fig4.xaxis.axis_label = "Time In Run"
    # fig4.x_range = fig1.x_range

    # cell_list = list(set(temp_df.cell))
    # num_cells = max(cell_list) + 1
    # task_list = list(set(temp_df.task))
    # task_list.sort()
    # temp_df['yy'] = [task_list.index(aa)*num_cells + bb for aa,bb in zip(temp_df.task,temp_df.cell)]
    # temp_df['color'] = 'orange'
    # temp_df.loc[temp_df.task=='DL C Plane','color'] = 'blue'
    # temp_df.loc[temp_df.task=='DL U Plane','color'] = 'red'
    # temp_df.loc[temp_df.task=='UL C Plane','color'] = 'green'
    # temp_df.loc[temp_df.is_late==True,'color'] = 'black'
    # source1 = ColumnDataSource(temp_df[temp_df.task=='DL C Plane'])
    # source2 = ColumnDataSource(temp_df[temp_df.task=='DL U Plane'])
    # source3 = ColumnDataSource(temp_df[temp_df.task=='UL C Plane'])
    # fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source1, color='color', line_width=1, line_color='black', legend_label='DL-C')
    # fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source2, color='color', line_width=1, line_color='black', legend_label='DL-U')
    # fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source3, color='color', line_width=1, line_color='black', legend_label='UL-C')
    # fig3.yaxis.axis_label = "Cell/Task"
    # fig3.xaxis.axis_label = "Time In Run"
    # fig3.x_range = fig1.x_range

    # TOOLTIPS5 = [
    #     ("frame", "@frame"),
    #     ("slot", "@slot"),
    #     ("cell", "@cell"),
    #     ("symbol", "@symbol"),
    # ]
    # fig5 = figure(title="DU TX Times (Relative to Symbol Window Time)", tooltips=TOOLTIPS4, height=400, width=1600)
    # if(len(packet_timing_df) > 0):
    #     windows_df = packet_timing_df[['seq_num','frame','slot','cell','symbol','desired_offset_usec','actual_offset_usec','symbol_window_timestamp']].copy()
    #     windows_df['tir'] = (windows_df.symbol_window_timestamp - ref_t0)/1e9
    #     data1 = windows_df[windows_df.seq_num == 0] #Wait WQE
    #     data2 = windows_df[windows_df.seq_num == 1] #First Packet
    #     data3 = windows_df[windows_df.seq_num == 2] #Last Packet
    #     source1 = ColumnDataSource(data1)
    #     source2 = ColumnDataSource(data2)
    #     source3 = ColumnDataSource(data3)
    #     fig5.circle(source=source1,x='tir',y='desired_offset_usec',color=Category20[6][0],legend_label='Requested Offset')
    #     fig5.circle(source=source1,x='tir',y='actual_offset_usec',color=Category20[6][1],legend_label='Actual Offset (Wait WQE)')
    #     #fig5.circle(source=source2,x='tir',y='desired_offset_usec',color=Category20[6][2],legend_label='Requested Offset (first packet)')
    #     fig5.circle(source=source2,x='tir',y='actual_offset_usec',color=Category20[6][3],legend_label='Actual Offset (first packet)')
    #     #fig5.circle(source=source3,x='tir',y='desired_offset_usec',color=Category20[6][4],legend_label='Requested Offset (last packet)')
    #     fig5.circle(source=source3,x='tir',y='actual_offset_usec',color=Category20[6][5],legend_label='Actual Offset (last packet)')
    # fig5.legend.click_policy = "hide"
    # fig5.yaxis.axis_label = "DU TX Times (usec)"
    # fig5.xaxis.axis_label = "Time In Run"
    # fig5.x_range = fig1.x_range

    # return column([fig1,fig2,fig3,fig4,fig5])
    return column([fig1,fig2,fig3])

def plot_late_slot_debug(lp_slot_df,ti_df,compression_df,symbol_timing_df,disable_timeline=False):
    TOOLTIPS1 = [
            ("frame", "@frame"),
            ("slot", "@slot"),
            ("cell", "@cell"),
            ("late_packet_count", "@late_packet_count"),
        ]

    pdsch_callback_df = ti_df[(ti_df.task=="DL Task GPU Comms") & (ti_df.subtask=="PDSCH Callback")]
    compression_durations_df = compression_df[(compression_df.event=='compression_kernel')][['t0_timestamp','duration']]
    gap_durations_df = compression_df[(compression_df.event=='channel_to_compression_gap')][['t0_timestamp','duration']]
    prepare_kernel_df = compression_df[(compression_df.event=='prepare_kernel')][['t0_timestamp','duration']]
    prepare_memsets_df = compression_df[(compression_df.event=='prepare_memsets')][['t0_timestamp','duration']]

    merge1 = pd.merge(pdsch_callback_df,compression_durations_df,on=["t0_timestamp"],validate="one_to_one",suffixes=("","_compression_kernel"))
    merge1 = pd.merge(merge1,gap_durations_df,on=["t0_timestamp"],validate="one_to_one",suffixes=("","_channel_gap"))
    merge1 = pd.merge(merge1,prepare_kernel_df,on=["t0_timestamp"],validate="one_to_one",suffixes=("","_prepare_kernel"))
    merge1 = pd.merge(merge1,prepare_memsets_df,on=["t0_timestamp"],validate="one_to_one",suffixes=("","_prepare_memsets"))
    merge1['compression_start'] = ((merge1['start_timestamp'] - merge1['duration_compression_kernel']*1000) - merge1['t0_timestamp'])/1000.
    merge1['channel_stop'] = ((merge1['start_timestamp'] - merge1['duration_compression_kernel']*1000 - merge1['duration_channel_gap']*1000) - merge1['t0_timestamp'])/1000.
    merge1['prepare_kernel_start'] = ((merge1['start_timestamp'] - merge1['duration_compression_kernel']*1000 - merge1['duration_prepare_kernel']*1000) - merge1['t0_timestamp'])/1000.
    merge1['prepare_memsets_start'] = ((merge1['start_timestamp'] - merge1['duration_compression_kernel']*1000 - merge1['duration_prepare_kernel']*1000 - merge1['duration_prepare_memsets']*1000) - merge1['t0_timestamp'])/1000.

    # Calculate reference t0 of phy log
    ref_t0 = ti_df.iloc[0].t0_timestamp - int(ti_df.iloc[0].tir*1e9)

    # Add plot that contain DLU late slots
    dlu_df = lp_slot_df[lp_slot_df.task=="DL U Plane"].copy()
    dlu_df['tir'] = (dlu_df.t0_timestamp - ref_t0)/1e9
    source = ColumnDataSource(dlu_df)
    fig1 = figure(title="Late DLU Packets vs Time", tooltips=TOOLTIPS1, height=400, width=1600)
    fig1.circle(x='tir', y='late_packet_count',source=source,color='black')
    fig1.yaxis.axis_label = "Late Packets"
    fig1.xaxis.axis_label = "Time In Run"

    # Add plot that contains end of compression time
    TOOLTIPS2 = [
        ("slot", "@slot"),
        ("sfn", "@sfn"),
    ]
    fig2 = figure(title="Slot Deadlines", tooltips=TOOLTIPS2, height=400, width=1600)
    current_df = ti_df[(ti_df.task=="DL Task GPU Comms") & (ti_df.subtask=="PDSCH Callback") & (ti_df.slot%10 != 9) & (ti_df.slot%10 != 6)]
    current_source = ColumnDataSource(current_df)
    fig2.circle(x='tir',y='end_deadline',source=current_source,color='green',line_color='black',size=9,legend_label='Compression Completed')
    current_df = ti_df[(ti_df.task=="DL Task GPU Comms") & (ti_df.subtask=="PDSCH Callback") & (ti_df.slot%10 == 9)]
    current_source = ColumnDataSource(current_df)
    fig2.circle(x='tir',y='end_deadline',source=current_source,color='red',line_color='black',size=9,legend_label='Compression Completed (*9 slot)')
    current_df = ti_df[(ti_df.task=="DL Task GPU Comms") & (ti_df.subtask=="PDSCH Callback") & (ti_df.slot%10 == 6)]
    current_source = ColumnDataSource(current_df)
    fig2.circle(x='tir',y='end_deadline',source=current_source,color='yellow',line_color='black',size=9,legend_label='Compression Completed (*6 slot)')

    current_source = ColumnDataSource(merge1)
    fig2.circle(x='tir',y='compression_start',source=current_source,color='orange',line_color='black',size=9,legend_label='Compression Start')
    current_source = ColumnDataSource(merge1)
    fig2.circle(x='tir',y='channel_stop',source=current_source,color='blue',line_color='black',size=9,legend_label='Channel Stop')
    current_source = ColumnDataSource(merge1)
    fig2.circle(x='tir',y='prepare_kernel_start',source=current_source,color='purple',line_color='black',size=9,legend_label='Prepare Kernel Start')
    current_source = ColumnDataSource(merge1)
    fig2.circle(x='tir',y='prepare_memsets_start',source=current_source,color='pink',line_color='black',size=9,legend_label='Prepare Memset Start')

    fig2.legend.click_policy = "hide"
    fig2.yaxis.axis_label = "Deadline Time"
    fig2.xaxis.axis_label = "Time In Run"
    fig2.x_range = fig1.x_range


    fig4 = figure(title="Compression Completion vs Late Packet Count (cell0)", tooltips=TOOLTIPS2, height=400, width=800)
    merge2 = pd.merge(pdsch_callback_df,lp_slot_df[(lp_slot_df.cell==0) & (lp_slot_df.task == 'DL U Plane')][['t0_timestamp','late_packet_count']],on=["t0_timestamp"],validate="one_to_one")
    current_df = merge2[(merge2.slot%10 == 9)]
    current_source = ColumnDataSource(current_df)
    fig4.circle(x='late_packet_count',y='end_deadline',source=current_source,color='red',line_color='black',size=9,legend_label='*9 slot')
    current_df = merge2[(merge2.slot%10 == 6)]
    current_source = ColumnDataSource(current_df)
    fig4.circle(x='late_packet_count',y='end_deadline',source=current_source,color='yellow',line_color='black',size=9,legend_label='*6 slot')

    fig4.legend.click_policy = "hide"
    fig4.yaxis.axis_label = "Compression Completion"
    fig4.xaxis.axis_label = "Late Packet Count"

    if( not disable_timeline):
        TOOLTIPS3 = [
            ("frame", "@frame"),
            ("slot", "@slot"),
            ("cell", "@cell"),
            ("symbol", "@symbol"),
            ("duration", "@duration"),
        ]
        fig3 = figure(title="Symbol Timeline", tooltips=TOOLTIPS3, height=400, width=1600)
        temp_df = symbol_timing_df.copy()
        cell_list = list(set(temp_df.cell))
        num_cells = max(cell_list) + 1
        task_list = list(set(temp_df.task))
        task_list.sort()
        temp_df['yy'] = [task_list.index(aa)*num_cells + bb for aa,bb in zip(temp_df.task,temp_df.cell)]
        temp_df['color'] = 'orange'
        temp_df.loc[temp_df.task=='DL C Plane','color'] = 'blue'
        temp_df.loc[temp_df.task=='DL U Plane','color'] = 'red'
        temp_df.loc[temp_df.task=='UL C Plane','color'] = 'green'
        temp_df.loc[temp_df.is_late==True,'color'] = 'black'
        source1 = ColumnDataSource(temp_df[temp_df.task=='DL C Plane'])
        source2 = ColumnDataSource(temp_df[temp_df.task=='DL U Plane'])
        source3 = ColumnDataSource(temp_df[temp_df.task=='UL C Plane'])
        fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source1, color='color', line_width=1, line_color='black', legend_label='DL-C')
        fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source2, color='color', line_width=1, line_color='black', legend_label='DL-U')
        fig3.hbar(y='yy', height=0.8, left='start_tir', right='end_tir', source=source3, color='color', line_width=1, line_color='black', legend_label='UL-C')
        fig3.yaxis.axis_label = "Cell/Task"
        fig3.xaxis.axis_label = "Time In Run"
        fig3.x_range = fig1.x_range

    if(not disable_timeline):
        return column([fig1,row(fig2,fig4),fig3])
    else:
        return column([fig1,row(fig2,fig4)])


def plot_late_slot_distributions(lp_slot_df):
    import pandas as pd

    TOOLTIPS = [
            ("slot", "@slot"),
            ("late_count", "@late_count"),
            ("total_count", "@total_count"),
        ]

    min_slot = min(lp_slot_df.slot)
    max_slot = max(lp_slot_df.slot)

    zeros = []
    slots = []
    late_counts = []
    total_counts = []
    late_percentages = []
    for ii in range(min_slot,max_slot+1):
        df1 = lp_slot_df[(lp_slot_df.slot==ii) & (lp_slot_df.late_packet_count > 0)]
        df2 = lp_slot_df[(lp_slot_df.slot==ii)]
        zeros.append(0.0)
        slots.append(ii)
        late_counts.append(len(df1))
        total_counts.append(len(df2))
        if(len(df2) > 0):
            late_percentages.append(float(len(df1)) / float(len(df2)) * 100.0)
        else:
            late_percentages.append(0.0)

    source = ColumnDataSource({
        'zero':zeros,
        'slot':slots,
        'late_count':late_counts,
        'total_count':total_counts,
        'late_slot_percentages':late_percentages})
    fig1 = figure(title="Late Slot Percentage vs Slot", tooltips=TOOLTIPS, height=800, width=800)
    fig1.vbar(x='slot', width=0.8, bottom='zero', top='late_slot_percentages', source=source, color='red', alpha=0.5)
    fig1.yaxis.axis_label = "Late Slot Percentage"
    fig1.xaxis.axis_label = "Slot"
    fig1.x_range = Range1d(min_slot - 1, max_slot + 1)
    fig1.y_range = Range1d(-2, 5)


    return fig1


def compression_boxplot(compression_df):

    TOOLTIPS = [
        ("event", "@event"),
        ("slot", "@slot"),
        ("duration", "@duration"),
    ]
    title = "Compression Event Durations"
    boxwhisker = hv.BoxWhisker(compression_df, ['event','slot'], 'duration', label=title)
    boxwhisker.opts(show_legend=False, height=800, width=1600, box_fill_color='event', cmap='Set1')
    hover=HoverTool(tooltips=TOOLTIPS)
    boxwhisker.opts(tools=[hover])
    fig = hv.render(boxwhisker)
    fig.yaxis.axis_label = 'duration (us)'

    return fig



def whisker_comparison(df_list,yfield,groupby_list,title="",ylabel="",height=900,width=1600):
    import holoviews as hv
    from holoviews import dim
    hv.extension('bokeh')

    # Create the tooltips
    all_fields = copy.deepcopy(groupby_list)
    all_fields.append(yfield)
    TOOLTIPS = [ (aa,"@"+aa) for aa in all_fields ]

    # Concatenate datasets
    truncated_cmap = []
    current_color_index = 0
    for ii,df in enumerate(df_list):
        temp_df = df.copy()
        temp_df['file_index'] = current_color_index
        if(ii == 0):
            concat_df = temp_df
        else:
            concat_df = pd.concat([concat_df,temp_df])

        if(len(temp_df) > 0):
            truncated_cmap.append(color_select(ii,len(df_list)))
            current_color_index += 1

    boxwhisker = hv.BoxWhisker(concat_df, groupby_list, yfield, label=title)
    #Note: Using color mapping, so colors must be one to one with file_index and file_index must be evenly spaced
    # This means that file_index is not in line with the index of original df_list, but just increments for each valid set of data
    boxwhisker.opts(show_legend=False, height=height, width=width, box_fill_color=dim('file_index').str(),cmap=truncated_cmap)
    hover=HoverTool(tooltips=TOOLTIPS)
    boxwhisker.opts(tools=[hover])
    fig = hv.render(boxwhisker)
    fig.title.render_mode = 'css'

    fig.yaxis.axis_label = ylabel
    fig.yaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label_text_font_size = '16pt'

    return fig

def whisker_comparison_legend(df_list,filename_list,disable_completion_line=True):

    box_width = 10
    box_height = 1

    fig = figure(height=200,width=400,title='Legend:')
    num_colors = min(max(len(df_list),3),9)
    for ii,df in enumerate(df_list):
        xx = 0
        yy = -2*ii
        fig.rect(x=xx,y=yy,width=box_width,height=box_height,color=Set1[num_colors][ii%num_colors])
        source = ColumnDataSource(dict(x=[xx], y=[yy], text=[filename_list[ii]]))
        fig.text(source=source,text='text',x='x',y='y',text_font_size='16pt',text_align='center',text_baseline='middle')

    if(not disable_completion_line):
        xx = 0
        yy = -2*len(df_list)
        fig.line(x=[xx-box_width/2, xx+box_width/2],y=[yy-box_height/2, yy-box_height/2],color='black')
        source = ColumnDataSource(dict(x=[xx], y=[yy], text=['Completion Deadline']))
        fig.text(source=source,text='text',x='x',y='y',text_font_size='16pt',text_align='center',text_baseline='middle')

    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.axis.visible = False
    fig.toolbar.logo = None
    fig.toolbar_location = None

    return fig

def comparison_legend(label_list,height=200,width=400,color_list=None,text_font_size='16pt'):
    box_width = 10
    box_height = 1

    fig = figure(height=height,width=width,title='Legend:')
    for ii,df in enumerate(label_list):
        xx = 0
        yy = -2*ii

        if(color_list):
            current_color = color_list[ii%len(color_list)]
        else:
            current_color = color_select(ii,len(label_list))

        fig.rect(x=xx,y=yy,width=box_width,height=box_height,color=current_color)
        source = ColumnDataSource(dict(x=[xx], y=[yy], text=[label_list[ii]]))
        fig.text(source=source,text='text',x='x',y='y',text_font_size=text_font_size,text_align='center',text_baseline='middle')

    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.axis.visible = False
    fig.toolbar.logo = None
    fig.toolbar_location = None

    return fig

def ccdf_breakout(df,yfields,xdelta=1,xstart=-1500,xstop=0,title="",xlabel="",height=800,width=800,color_list=None):
    nbins = int(round((xstop - xstart)/xdelta)) + 1;
    bin_edges = [xstart + xdelta*(ii-0.5) for ii in range(nbins+1)]
    bins = [xstart + xdelta*ii for ii in range(nbins)]

    # Calculate ccdf
    ccdf_list = []
    for yfield in yfields:
        if(len(df) > 0):
            hh,_ = np.histogram(df[yfield],bin_edges)
            aa = np.cumsum(hh) #Calculate non-normalized cdf
            bb = aa/np.max(aa) #Normalize the cdf
            cc = 1.0 - bb #Complement the cdf
            ccdf_list.append(cc)
        else:
            ccdf_list.append(None)

    # Produce plots
    fig = figure(title=title, height=height, width=width, y_axis_type="log", y_range=[0.5e-6, 2])
    line_list = []
    for ii,ccdf in enumerate(ccdf_list):
        if(ccdf is not None):
            if(color_list):
                current_color = color_list[ii%len(color_list)]
            else:
                current_color = color_select(ii,len(yfields))
            line_list.append(fig.line(x=bins,y=ccdf,color=current_color,line_width=5))

    fig.add_tools(HoverTool(tooltips="y: @y, x: @x", renderers=line_list, mode="hline"))

    fig.title.render_mode = 'css'
    fig.yaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = 'CCDF (Fraction > Value)'

    return fig

def ccdf_comparison(df_list,yfield,xdelta=1,xstart=-1500,xstop=0,ystart=0.5e-6,ystop=2,title="",xlabel="",height=400,width=800,color_list=None,no_complement=False):

    nbins = int(round((xstop - xstart)/xdelta)) + 1;
    bin_edges = [xstart + xdelta*(ii-0.5) for ii in range(nbins+1)]
    bins = [xstart + xdelta*ii for ii in range(nbins)]

    # Calculate ccdf
    ccdf_list = []
    for df in df_list:
        if(len(df) > 0):
            hh,_ = np.histogram(df[yfield],bin_edges)
            aa = np.cumsum(hh) #Calculate non-normalized cdf
            bb = aa/np.max(aa) #Normalize the cdf
            if(not no_complement):
                cc = 1.0 - bb #Complement the cdf
            else:
                cc = bb
            ccdf_list.append(cc)
        else:
            ccdf_list.append(None)

    # Produce plots
    fig = figure(title=title, height=height, width=width, y_axis_type="log", y_range=[ystart, ystop])
    line_list = []
    for ii,ccdf in enumerate(ccdf_list):
        if(ccdf is not None):
            if(color_list):
                current_color = color_list[ii%len(color_list)]
            else:
                current_color = color_select(ii,len(df_list))
            line_list.append(fig.line(x=bins,y=ccdf,color=current_color,line_width=5))

    fig.add_tools(HoverTool(tooltips="y: @y, x: @x", renderers=line_list, mode="hline"))

    fig.title.render_mode = 'css'
    fig.yaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label = xlabel
    if(not no_complement):
        fig.yaxis.axis_label = 'CCDF (Fraction > Value)'
    else:
        fig.yaxis.axis_label = 'CDF (Fraction <= Value)'

    return fig

from bokeh.transform import factor_cmap
def bar_plot(input_df,xfield_list,yfield,yfield_name=None,title="",xlabel="",ylabel="",
             height=800,width=800,color_list=None,x_range=None,y_range=None,enable_colors=False,
             series_labels=None):

    # Check if input is a list of dataframes (multiple series) or single dataframe
    if isinstance(input_df, list):
        return _bar_plot_multi_series(input_df, xfield_list, yfield, yfield_name, title, xlabel, ylabel,
                                    height, width, color_list, x_range, y_range, series_labels)
    
    # Original single series implementation
    #Create input data
    if(len(xfield_list) > 1):
        xx = list(input_df[xfield_list].astype('str').itertuples(index=False, name=None))
    else:
        xx = list(input_df[xfield_list[0]].astype('str'))
    yy = list(input_df[yfield])
    source = ColumnDataSource(data=dict(xx=xx, yy=yy))

    if(enable_colors and len(xfield_list) > 1):
        #Use all but first field to color entries
        unique_values = list(set([aa[1:] for aa in xx]))
        unique_values.sort()
        # palette = [color_select(ii,len(unique_values)) for ii in range(len(unique_values))]
        palette = [color_select(unique_values.index(aa[1:]),len(unique_values)) for aa in list(set(xx))]
        fill_color = factor_cmap('xx', palette=palette, factors=list(set(xx)))
        line_color = factor_cmap('xx', palette=palette, factors=list(set(xx)))
    else:
        fill_color = color_select(0,1)
        line_color = fill_color

    #Generate figure
    if(yfield_name):
        tooltips=[(yfield_name,"@yy")]
    else:
        tooltips=None
    if(x_range and y_range):
        fig = figure(x_range=x_range, y_range=y_range, height=height, width=width, title=title, tooltips=tooltips)
    elif(x_range and not y_range):
        fig = figure(x_range=x_range, height=height, width=width, title=title, tooltips=tooltips)
    elif(not x_range and y_range):
        fig = figure(x_range=FactorRange(*xx), y_range=y_range, height=height, width=width, title=title, tooltips=tooltips)
    else:
        fig = figure(x_range=FactorRange(*xx), height=height, width=width, title=title, tooltips=tooltips)

    vbar = fig.vbar(x='xx', top='yy', width=0.9, source=source, fill_color=fill_color, line_color=line_color)

    #Make it so only glyph that uses HoverTool is the vertical bars (allows annotations)
    ht = [aa for aa in fig.tools if type(aa)==HoverTool][0]
    ht.renderers = [vbar]

    # fig.add_tools(HoverTool(tooltips="y: @yy, x: @xx", renderers=vbar, mode="hline"))
    # fig.title.render_mode = 'css'
    fig.yaxis.axis_label_text_font_size = '12pt'
    fig.xaxis.axis_label_text_font_size = '12pt'
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = ylabel

    return fig

def _bar_plot_multi_series(input_df_list, xfield_list, yfield, yfield_name=None, title="", xlabel="", ylabel="",
                         height=800, width=800, color_list=None, x_range=None, y_range=None, series_labels=None):
    """
    Internal function to handle multiple data series in bar plots
    """
    from bokeh.transform import dodge
    
    # Default colors if not provided
    if color_list is None:
        color_list = [color_select(i, len(input_df_list)) for i in range(len(input_df_list))]
    
    # Default labels if not provided
    if series_labels is None:
        series_labels = [f"Series {i+1}" for i in range(len(input_df_list))]
    
    # Collect all unique x values across all series
    all_xx = set()
    for df in input_df_list:
        if len(df) > 0:
            if len(xfield_list) > 1:
                xx = list(df[xfield_list].astype('str').itertuples(index=False, name=None))
            else:
                xx = list(df[xfield_list[0]].astype('str'))
            all_xx.update(xx)
    
    all_xx = sorted(list(all_xx))
    
    # Generate figure
    if(yfield_name):
        tooltips=[(yfield_name,"@yy"), ("series", "@series")]
    else:
        tooltips=[("series", "@series")]
        
    if(x_range and y_range):
        fig = figure(x_range=x_range, y_range=y_range, height=height, width=width, title=title, tooltips=tooltips)
    elif(x_range and not y_range):
        fig = figure(x_range=x_range, height=height, width=width, title=title, tooltips=tooltips)
    elif(not x_range and y_range):
        fig = figure(x_range=FactorRange(*all_xx), y_range=y_range, height=height, width=width, title=title, tooltips=tooltips)
    else:
        fig = figure(x_range=FactorRange(*all_xx), height=height, width=width, title=title, tooltips=tooltips)
    
    # Calculate bar width and spacing
    num_series = len(input_df_list)
    bar_width = 0.8 / num_series
    
    vbar_renderers = []
    
    # Plot each series
    for i, (df, label) in enumerate(zip(input_df_list, series_labels)):
        if len(df) == 0:
            continue
            
        # Create input data for this series
        if len(xfield_list) > 1:
            xx = list(df[xfield_list].astype('str').itertuples(index=False, name=None))
        else:
            xx = list(df[xfield_list[0]].astype('str'))
        yy = list(df[yfield])
        
        # Add series info for tooltips
        series_info = [label] * len(xx)
        source = ColumnDataSource(data=dict(xx=xx, yy=yy, series=series_info))
        
        # Calculate dodge offset
        dodge_offset = (i - (num_series - 1) / 2) * bar_width
        
        # Create bars
        current_color = color_list[i % len(color_list)]
        vbar = fig.vbar(x=dodge('xx', dodge_offset, range=fig.x_range), 
                       top='yy', 
                       width=bar_width, 
                       source=source, 
                       fill_color=current_color,
                       line_color=current_color,
                       legend_label=label)
        vbar_renderers.append(vbar)
    
    # Configure hover tool to work with all bars
    ht = [aa for aa in fig.tools if type(aa)==HoverTool][0]
    ht.renderers = vbar_renderers
    
    # Configure legend
    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    
    # Configure axes
    fig.yaxis.axis_label_text_font_size = '12pt'
    fig.xaxis.axis_label_text_font_size = '12pt'
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = ylabel
    
    return fig

#Computes running max, min, median, avg, q99, and q01
# Bins are based on specified xfield/xdelta
def compute_running_stats(input_df,xfield,yfield,xdelta=0.0):
    xdelta_div2 = xdelta/2.0

    #Read in data and sort
    xx = np.array(input_df[xfield])
    yy = np.array(input_df[yfield])
    sorted_indices = np.argsort(xx)
    xx = xx[sorted_indices]
    yy = yy[sorted_indices]

    output_data = {'xx_bin':[],
                   'yy_min':[],
                   'yy_max':[],
                   'yy_med':[],
                   'yy_avg':[],
                   'yy_q99':[],
                   'yy_q01':[]}

    if(xdelta > 0.0):
        #Create bins
        start_xx = input_df[xfield].min()
        end_xx = input_df[xfield].max()
        num_bins = int((end_xx - start_xx + xdelta) / xdelta)
        xx_bins = [start_xx + ii*xdelta + xdelta_div2 for ii in range(num_bins)]

        
        for ii,xx_bin in enumerate(xx_bins):

            #Note: this is equivalent to saying xx_bin-xdelta/2.0 < xx <= xx+xdelta/2.0
            if(ii==0):
                #Use nominal bin split
                left_half_bin = xdelta_div2
            else:
                #Use estimate based on bins (avoids rounding error)
                left_half_bin = (xx_bins[ii] - xx_bins[ii-1]) / 2.0

            if(ii==len(xx_bins)-1):
                #Use nominal bin split
                right_half_bin = xdelta_div2
            else:
                #Use estimate based on bins (avoids rounding error)
                right_half_bin = (xx_bins[ii+1] - xx_bins[ii]) / 2.0

            idx1 = bisect.bisect_left(xx,xx_bin-left_half_bin)
            idx2 = bisect.bisect_left(xx,xx_bin+right_half_bin)
            current_xx = xx[idx1:idx2]
            current_yy = yy[idx1:idx2]

            if(len(current_xx) > 0):
                output_data['xx_bin'].append(xx_bin)
                output_data['yy_min'].append(np.min(current_yy))
                output_data['yy_max'].append(np.max(current_yy))
                output_data['yy_med'].append(np.median(current_yy))
                output_data['yy_avg'].append(np.mean(current_yy))
                output_data['yy_q99'].append(np.quantile(current_yy,0.99))
                output_data['yy_q01'].append(np.quantile(current_yy,0.01))

    else:
        output_data['xx_bin'] = xx
        output_data['yy_min'] = yy
        output_data['yy_max'] = yy
        output_data['yy_med'] = yy
        output_data['yy_avg'] = yy
        output_data['yy_q99'] = yy
        output_data['yy_q01'] = yy

    return output_data

#Given a dataframe containing 'xfield' (typically time in run, but can be anything) and 'yfield', calculate and plot the statistics
# vs xdelta bins
#
#Computes/displays max, min, median, avg
#
#Note: defaults to xdelta=0.0, which just plots the raw points
def stats_plot(input_df,xfield,yfield,xdelta=0.0,yfield_name=None,title="",xlabel="",ylabel="",height=800,width=800,x_range=None,enable_q99=False,enable_q01=False,crosshair_tool=None,disable_lines=False):

    COMPUTE_STATS = xdelta is not None

    #Compute stats and create data source
    if(COMPUTE_STATS):
        stats = compute_running_stats(input_df,xfield,yfield,xdelta=xdelta)
    else:
        stats = {'xx':[aa for aa in input_df[xfield]],
                 'yy':[aa for aa in input_df[yfield]]}
    source = ColumnDataSource(data=stats)

    #Generate figure
    if(x_range):
        fig = figure(x_range=x_range, height=height, width=width, title=title)
    else:
        fig = figure(height=height, width=width, title=title)

    line_list = []
    if(COMPUTE_STATS):
        cc1 = fig.circle(x='xx_bin', y='yy_min', width=0.9, source=source,color='green',legend_label='min')
        cc2 = fig.circle(x='xx_bin', y='yy_max', width=0.9, source=source,color='red',legend_label='max')
        cc3 = fig.circle(x='xx_bin', y='yy_med', width=0.9, source=source,color='blue',legend_label='median')
        cc4 = fig.circle(x='xx_bin', y='yy_avg', width=0.9, source=source,color='orange',legend_label='average')
        if(not disable_lines):
            lin1 = fig.line(x='xx_bin', y='yy_med', width=0.9, source=source,color='blue')
            line_list.append(lin1)

        if(enable_q01):
            q01_cc1 = fig.circle(x='xx_bin', y='yy_q01', width=0.9, source=source,color='lightgreen',legend_label='1%')
            if(not disable_lines):
                q01_lin1 = fig.line(x='xx_bin', y='yy_q01', width=0.9, source=source,color='lightgreen')

        if(enable_q99):
            q99_cc1 = fig.circle(x='xx_bin', y='yy_q99', width=0.9, source=source,color='lightcoral',legend_label='99%')
            if(not disable_lines):
                q99_lin1 = fig.line(x='xx_bin', y='yy_q99', width=0.9, source=source,color='lightcoral')
    else:
        cc4 = fig.circle(x='xx', y='yy', width=0.9, source=source,color='blue',legend_label='val')
        if(not disable_lines):
            lin1 = fig.line(x='xx', y='yy', width=0.9, source=source,color='blue')
            line_list.append(lin1)

    fig.yaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = ylabel

    if(COMPUTE_STATS):
        fig.add_tools(HoverTool(tooltips=[('99%','@yy_q99'),
                                          ('50%','@yy_med'),
                                          (' 1%','@yy_q01')], renderers=line_list, mode="vline"))
    else:
        fig.add_tools(HoverTool(tooltips=[('yval','@yy'),
                                          ('xval','@xx')], renderers=line_list, mode="vline"))
        

    if(crosshair_tool is not None):
        fig.add_tools(crosshair_tool)

    return fig


def color_select(current_index,num_colors=10):
    if(num_colors <= 10):
        DEFAULT_PALETTE = Category10
    else:
        DEFAULT_PALETTE = Category20

    min_colors = min(list(DEFAULT_PALETTE.keys()))
    max_colors = max(list(DEFAULT_PALETTE.keys()))
    color_count = min(max(min_colors,num_colors),max_colors)

    return DEFAULT_PALETTE[color_count][current_index%color_count]


from bokeh.plotting import show, save
from bokeh.io import output_file
import os
import time

def output_plot(fig,out_filename=None):
    time1 = time.time()

    if(not out_filename):
        out_filename = "result.html"
    output_file(filename=out_filename, title=os.path.split(__file__)[1])
    print("Writing output to %s"%out_filename)
    if(out_filename):
        time1 = time.time()
        save(fig)
        time2 = time.time()

        print("Plot saved in %fsec"%(time2-time1))
    else:
        show(fig)

def format_labels(labels,num_files):
    if not labels:
        labels = ["Unlabeled File %02i"%ii for ii in range(num_files)]
    while(len(labels) < num_files):
        labels.append("Unlabeled File %02i"%(len(labels)-1))

    return labels
