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

import pandas as pd
import argparse
import os

from bokeh.plotting import show, save, figure
from bokeh.io import output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, TableColumn, DataTable, StringFormatter, NumberFormatter, Div, HTMLTemplateFormatter

# Imported color_select function directly to remove dependency
from bokeh.palettes import Category10, Category20

def color_select(current_index, num_colors=10):
    if(num_colors <= 10):
        DEFAULT_PALETTE = Category10
    else:
        DEFAULT_PALETTE = Category20

    min_colors = min(list(DEFAULT_PALETTE.keys()))
    max_colors = max(list(DEFAULT_PALETTE.keys()))
    color_count = min(max(min_colors,num_colors),max_colors)

    return DEFAULT_PALETTE[color_count][current_index%color_count]

def format_figure(fig,xlabel,ylabel):
    fig.title.render_mode = 'css'
    fig.xaxis.axis_label_text_font_size = '16pt'
    fig.yaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = ylabel
    fig.legend.click_policy = 'hide'

def main(args):
    data = pd.read_csv(args.power_csv)
    
    # Check if any PDU data is available
    has_pdu_data = any('[PDU]' in col for col in data.columns)
    
    if has_pdu_data:
        print("PDU data detected")
        # Only calculate PDU power if PDU fields are available
        if '[PDU]total_current' in data.columns and '[PDU]voltage' in data.columns:
            data['[PDU]power=I*V'] = data['[PDU]total_current']*data['[PDU]voltage']
    else:
        print("No PDU data found - proceeding with available power fields only")
    
    source = ColumnDataSource(data)
    col_list = []

    if(args.external_overlay):
        outlet_data = []
        for ii,ff in enumerate(args.external_overlay):
            # temp_data = pd.read_csv(ff,skiprows=1,delim_whitespace=True,names=['sample_index','timestamp','pdu_ip','nam '])
            temp_data = pd.read_fwf(ff,widths=[4,10,15,35,35])
            temp_data["ii"] = ii
            rename_dict = {}
            for watt_col in [aa for aa in temp_data.columns if aa.find("Total Power")>=0]:
                rename_dict[watt_col] = "power"
            temp_data = temp_data.rename(columns=rename_dict)
            temp_data['tir'] = temp_data['timestamp'] - data.iloc[0]['collection_start_time'] + data.iloc[0]['collection_start_tir']
            outlet_data.append(temp_data)

        perflab_data = pd.concat(outlet_data)
        perflab_data = perflab_data.assign(total_power=perflab_data.groupby("timestamp").power.transform(sum))[['tir','total_power']]
        perflab_data = perflab_data.drop_duplicates()



    height = 400
    width = 1200
    tooltips = [("(x,y)", "($x, $y)")]

    TRACK_ACTIVITY = args.gpu_threshold is not None and "[GPU]power.draw.average" in data.columns
    if(TRACK_ACTIVITY):

        #Find all data above threshold
        active_data = data[data["[GPU]power.draw.average"]>args.gpu_threshold]

        #Must have at least two available data points
        if(len(active_data)<2):
            print("ERROR :: insufficient activity detected after power thresholding (data points=%i).  Disabling power calculation."%len(active_data))
            TRACK_ACTIVITY = False

    if(TRACK_ACTIVITY):
        #Define active window as 1sec after point at which data crossed threshold the first time to 1sec after where it went under threshold the last time
        active_start_tir = active_data.iloc[0].collection_end_tir + 1.0 + args.ignore_duration
        active_end_tir = active_data.iloc[-1].collection_end_tir - 1.0
        active_data = data[(data.collection_end_tir >= active_start_tir) & (data.collection_end_tir <= active_end_tir)]

        #Must have at least one available data point
        if(len(active_data)<1):
            print("ERROR :: insufficient activity detected after window cut (data points=%i, start=%f, end=%f).  Disabling power calculation."%(len(active_data),active_start_tir,active_end_tir))
            TRACK_ACTIVITY = False

    if(TRACK_ACTIVITY):

        summary_dict = {
            'source': [],
            'start_tir': [],
            'end_tir': [],
            'min_power': [],
            'avg_power': [],
            'med_power': [],
            'max_power': [],
            }

        # Build power fields list - include PDU fields only if PDU data is available
        power_fields_and_names = []
        if has_pdu_data:
            power_fields_and_names.append(('[PDU]total_power', 'PDU'))
            # Dynamically find all outlet power fields
            pdu_outlet_power_fields = [col for col in data.columns if col.startswith('[PDU]outlet_') and col.endswith('_power')]
            for field in sorted(pdu_outlet_power_fields):
                # Extract outlet number from field name like '[PDU]outlet_32_power'
                outlet_num = field.split('_')[1]
                power_fields_and_names.append((field, f'PDU Outlet {outlet_num}'))
        
        power_fields_and_names.extend([
            ('[MOD]Module Power Socket 0 Power', 'GH Module'),
            ('[GPU]power.draw.average', 'GPU'),
            ('[MOD]Grace Power Socket 0 Power', 'CPU+SysIO'),
            ('[MOD]CPU Power Socket 0 Power', 'CPU'),
            ('[MOD]SysIO Power Socket 0 Power', 'SysIO')
        ])

        for pow_col, pow_name in power_fields_and_names:
            if pow_col in active_data.columns:
                summary_dict['source'].append(pow_name)
                summary_dict['start_tir'].append(active_start_tir)
                summary_dict['end_tir'].append(active_end_tir)
                summary_dict['min_power'].append(active_data[pow_col].min())
                summary_dict['avg_power'].append(active_data[pow_col].mean())
                summary_dict['med_power'].append(active_data[pow_col].median())
                summary_dict['max_power'].append(active_data[pow_col].max())
            else:
                print("Unable to summarize field %s"%pow_col)
        
        #Output result to stdout
        for key in summary_dict.keys():
            if(key not in ['source']):
                print("%s: %f"%(key,summary_dict[key][0]))

        #If enabled output summary to csv output
        if(args.summary_csv):
            summary_df = pd.DataFrame(summary_dict)
            summary_df.to_csv(args.summary_csv,float_format="%.3f",index=False)
        
        data_table_source = ColumnDataSource(summary_dict)
        float_formatter = NumberFormatter(format="0,0.0",text_align='right')

        def get_html_formatter(my_col):
            template = """
                <div style="background:<%= 
                    (function colorfromint(){
                        if(result_col == 'Positive'){
                            return('#f14e08')}
                        else if (result_col == 'Negative')
                            {return('#8a9f42')}
                        else if (result_col == 'Invalid')
                            {return('#8f6b31')}
                        }()) %>; 
                    color: white"> 
                <%= value %>
                </div>
            """.replace('result_col',my_col)
            
            return HTMLTemplateFormatter(template=template)

        columns = [
            TableColumn(field="source", title="Power Data Source", formatter=StringFormatter()),
            TableColumn(field="start_tir", title="Start Time (sec)", formatter=float_formatter),
            TableColumn(field="end_tir", title="End Time (sec)", formatter=float_formatter),
            TableColumn(field="min_power", title="Min Power (W)", formatter=float_formatter),
            TableColumn(field="med_power", title="Med Power (W)", formatter=float_formatter),
            TableColumn(field="avg_power", title="Avg Power (W)", formatter=float_formatter),
            TableColumn(field="max_power", title="Max Power (W)", formatter=float_formatter),
        ]

        # columns = [
        #     TableColumn(field="start_tir", title="Start Time (sec)", formatter=get_html_formatter('start_tir')),
        #     TableColumn(field="end_tir", title="End Time (sec)", formatter=get_html_formatter('end_tir')),
        #     TableColumn(field="min_power", title="Min Power (W)", formatter=get_html_formatter('min_power')),
        #     TableColumn(field="med_power", title="Med Power (W)", formatter=get_html_formatter('med_power')),
        #     TableColumn(field="avg_power", title="Avg Power (W)", formatter=get_html_formatter('avg_power')),
        #     TableColumn(field="max_power", title="Max Power (W)", formatter=get_html_formatter('max_power')),
        # ]

        data_table = DataTable(source=data_table_source, columns=columns, index_position=None, height=240, width=width, editable=False,css_classes=["style1","style2"])
        data_table_style = Div(text="""<style>
                                        .style1{
                                        border: 1px solid !important;
                                        font-weight: bold !important;
                                        font-size: 14px !important; 
                                        font-color: black !important;
                                        }
                                        </style>
                                        """)
        col_list.append(column(data_table,data_table_style))

    #Power vs Time plots
    fields = []
    if has_pdu_data:
        # Add total power first
        fields.append("[PDU]total_power")
        # Dynamically find all outlet power fields
        pdu_outlet_power_fields = [col for col in data.columns if col.startswith('[PDU]outlet_') and col.endswith('_power')]
        fields.extend(sorted(pdu_outlet_power_fields))
    
    fields.extend(["[GPU]power.draw.average","[GPU]power.draw.instant","[GPU]power.limit","[MOD]Module Power Socket 0 Power","[MOD]Grace Power Socket 0 Power","[MOD]CPU Power Socket 0 Power","[MOD]SysIO Power Socket 0 Power"])
    
    num_colors = len(fields)
    if(args.external_overlay):
        num_colors += 1+len(outlet_data)
    fig1 = figure(title="Power vs Time",tooltips=tooltips,plot_width=width, plot_height=height)
    for ii,field in enumerate(fields):
        if(field in data.columns):
            fig1.circle(x='collection_end_tir',y=field,source=source,legend_label=field,color=color_select(ii,num_colors))
        else:
            print("Unable to find field %s in power data"%field)

    if(args.external_overlay):
        external_source = ColumnDataSource(perflab_data)
        fig1.circle(x='tir',y="total_power",source=external_source,legend_label="Perflab Total Power",color=color_select(ii+1,num_colors))
        for jj,od in enumerate(outlet_data):
            external_source = ColumnDataSource(od)
            fig1.circle(x='tir',y="power",source=external_source,legend_label="Perflab Outlet %i Power"%(jj+1),color=color_select(ii+jj+2,num_colors))


    if(TRACK_ACTIVITY):
        x1 = active_start_tir
        x2 = active_end_tir
        y1 = -50
        
        # Find max power for scaling - use PDU total power if available, otherwise fallback to other fields
        if has_pdu_data and '[PDU]total_power' in data.columns:
            y2 = 1.1*data["[PDU]total_power"].max()
        else:
            # Find max from available non-PDU power fields
            max_power_value = 0
            fallback_fields = ['[GPU]power.draw.average', '[MOD]Module Power Socket 0 Power']
            for power_field in fallback_fields:
                if power_field in data.columns:
                    field_max = data[power_field].max()
                    if field_max > max_power_value:
                        max_power_value = field_max
            y2 = 1.1 * max_power_value if max_power_value > 0 else 1100  # Default to 1100W if no power fields found
        
        # fig1.line(x=[x1,x1],y=[y1,y2],color='green',line_dash='dashed')
        # fig1.line(x=[x2,x2],y=[y1,y2],color='red',line_dash='dashed')
        fig1.rect(x=(x1+x2)/2.0,y=(y1+y2)/2.0,width=x2-x1,height=y2-y1,line_dash='dotted',line_color='green',line_alpha=1.0,fill_color='green',fill_alpha=0.05)

    format_figure(fig1,"Time in run (sec)","Power (W)")
    col_list.append(fig1)

    #Temps vs Time plots
    fields = ["[GPU]temperature.memory","[GPU]temperature.gpu","[GPU]temperature.gpu.tlimit"]
    fields.extend([aa for aa in data.columns if aa.find("TJMax")>=0])
    fig2 = figure(title="Temperature vs Time",tooltips=tooltips,plot_width=width, plot_height=height, x_range=fig1.x_range)
    for ii,field in enumerate(fields):
        if(field in data.columns):
            fig2.circle(x='collection_end_tir',y=field,source=source,legend_label=field,color=color_select(ii,len(fields)))
        else:
            print("Unable to find field %s in power data"%field)

    format_figure(fig2,"Time in run (sec)","Temperature (C)")
    col_list.append(fig2)

    #Clocks vs Time plots
    fields = ["[GPU]clocks.current.graphics","[GPU]clocks.current.sm","[GPU]clocks.current.memory","[GPU]clocks.current.video"]
    fig3 = figure(title="Clock vs Time",tooltips=tooltips,plot_width=width, plot_height=height, x_range=fig1.x_range)
    for ii,field in enumerate(fields):
        if(field in data.columns):
            fig3.circle(x='collection_end_tir',y=field,source=source,legend_label=field,color=color_select(ii,len(fields)))
        else:
            print("Unable to find field %s in power data"%field)

    format_figure(fig3,"Time in run (sec)","Clock Frequency (MHz)")
    col_list.append(fig3)

    #Throttle Reasons Bitmask vs Time
    bitmask_col = "[GPU]clocks_event_reasons.active"
    if bitmask_col in data.columns:
        bitmask_int_col = bitmask_col + "_int"
        def _parse_bitmask(x):
            s = str(x).strip()
            try:
                return int(s, 16) if s.startswith("0x") else int(s)
            except (ValueError, TypeError):
                return float('nan')
        data[bitmask_int_col] = data[bitmask_col].apply(_parse_bitmask)
        source_with_bitmask = ColumnDataSource(data)
        fig4 = figure(title="Clock Throttle Reasons vs Time",tooltips=tooltips,plot_width=width, plot_height=height, x_range=fig1.x_range)
        fig4.circle(x='collection_end_tir',y=bitmask_int_col,source=source_with_bitmask,legend_label="clocks_event_reasons.active",color=color_select(0,1))
        format_figure(fig4,"Time in run (sec)","Throttle Reasons (bitmask)")
        col_list.append(fig4)

    #Throttle Counter Deltas vs Time
    counter_fields = [
        "[GPU]clocks_event_reasons_counters.sw_power_cap",
        "[GPU]clocks_event_reasons_counters.hw_thermal_slowdown",
        "[GPU]clocks_event_reasons_counters.hw_power_brake_slowdown",
        "[GPU]clocks_event_reasons_counters.sw_thermal_slowdown",
        "[GPU]clocks_event_reasons_counters.sync_boost",
    ]
    available_counters = [f for f in counter_fields if f in data.columns]
    if available_counters:
        delta_cols = {}
        for field in available_counters:
            delta_col = field + "_delta"
            data[delta_col] = pd.to_numeric(data[field], errors='coerce').diff().fillna(0)
            delta_cols[field] = delta_col
        source_with_deltas = ColumnDataSource(data)
        fig5 = figure(title="Clock Throttle Duration vs Time",tooltips=tooltips,plot_width=width, plot_height=height, x_range=fig1.x_range)
        for ii,(orig_field,delta_col) in enumerate(delta_cols.items()):
            label = orig_field.replace("[GPU]clocks_event_reasons_counters.","")
            fig5.circle(x='collection_end_tir',y=delta_col,source=source_with_deltas,legend_label=label,color=color_select(ii,len(delta_cols)))
        format_figure(fig5,"Time in run (sec)","Throttle Duration (us per interval)")
        col_list.append(fig5)

    #Output plots
    if(args.out_filename):
        out_filename = args.out_filename
    else:
        out_filename = "result.html"
    output_file(filename=out_filename, title=os.path.split(__file__)[1])
    print("Writing output to %s"%out_filename)
    if(args.out_filename):
        save(column(col_list))
    else:
        show(column(col_list))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plots summary of collect power data"
    )
    parser.add_argument(
        "power_csv", help="Input power csv"
    )
    parser.add_argument(
        "-g", "--gpu_threshold", type=float, help="If specified attempts to determine time period of activity based on GPU wattage value > gpu_threshold.  Summarizes and plots power over that interval"
    )
    parser.add_argument(
        "-i", "--ignore_duration", type=float, default=0.0, help="Ignore this many seconds after the start of the activity period"
    )
    parser.add_argument(
        "-s", "--summary_csv", help="Output csv summarizing results"
    )
    parser.add_argument(
        "-o", "--out_filename", help="Filename for the output"
    )
    parser.add_argument(
        "-e", "--external_overlay", nargs="+", help="Specify a list of external perflab power dataset to overlay"
    )
    args = parser.parse_args()

    main(args)
