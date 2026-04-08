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
import pandas as pd
import numpy as np
import os
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10, Category20

#Note - goal was to make netperftest_compare standalone, so copying comparison_legend, ccdf_comparison, and color_select into this file
# from aerial_postproc.logplot import comparison_legend, ccdf_comparison


def color_select(current_index,num_colors=10):
    if(num_colors <= 10):
        DEFAULT_PALETTE = Category10
    else:
        DEFAULT_PALETTE = Category20

    min_colors = min(list(DEFAULT_PALETTE.keys()))
    max_colors = max(list(DEFAULT_PALETTE.keys()))
    color_count = min(max(min_colors,num_colors),max_colors)

    return DEFAULT_PALETTE[color_count][current_index%color_count]


def comparison_legend(label_list,height=200,width=400,color_list=None):
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
        fig.text(source=source,text='text',x='x',y='y',text_font_size='16pt',text_align='center',text_baseline='middle')

    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.axis.visible = False
    fig.toolbar.logo = None
    fig.toolbar_location = None

    return fig

def threshold_legend(height=50,width=400,color_list=None):
    box_width = width
    box_height = height

    fig = figure(height=height,width=width)
    fig.rect(x=[box_width//2],y=[box_height//2],width=[box_width],height=[box_height],color='white')
    fig.circle(x=[box_width//8],y=[box_height//2],color='black',size=6)
    fig.line(x=[box_width//8,box_width//8],y=[(box_height//2)+(box_height//4),(box_height//2)-(box_height//4)],color='black',line_dash='dashed')
    fig.text(text=['Thresholds'],x=[box_width/2],y=[box_height//2],text_font_size='16pt',text_align='center',text_baseline='middle')

    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.axis.visible = False
    fig.toolbar.logo = None
    fig.toolbar_location = None

    return fig


def ccdf_comparison(df_list,yfield,xdelta=1,xstart=-1500,xstop=0,title="",xlabel="",height=800,width=800,color_list=None,no_complement=False):

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
    fig = figure(title=title, height=height, width=width, y_axis_type="log", y_range=[0.5e-6, 2])
    line_list = []
    for ii,ccdf in enumerate(ccdf_list):
        if(ccdf is not None):
            if(color_list):
                current_color = color_list[ii%len(color_list)]
            else:
                current_color = color_select(ii,len(df_list))
            line_list.append(fig.line(x=bins,y=ccdf,color=current_color,line_width=5))

    fig.add_tools(HoverTool(tooltips="y: @y, x: @x", renderers=line_list, mode="hline"))

    # fig.title.render_mode = 'css'
    fig.yaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label_text_font_size = '16pt'
    fig.xaxis.axis_label = xlabel
    if(not no_complement):
        fig.yaxis.axis_label = 'CCDF (Fraction > Value)'
    else:
        fig.yaxis.axis_label = 'CDF (Fraction <= Value)'

    return fig

def add_thresholds(fig,threshold_data,prefix):
    threshold_xx = []
    threshold_yy = []
    for key in [aa for aa in threshold_data.columns if aa.startswith(prefix)]:
        val = threshold_data[key].iloc[0]
        if(not np.isnan(val)):
            probability = float(key.lstrip(prefix))/100.0
            yy = 1.0 - probability
            xx = val
            fig.circle(x=[xx],y=[yy],color='black',size=6)

            #Add vertical line denoting this as a threshold
            # ccdf_fig1.line(x=[xx,xx],y=[yy-line_height_decades*yy,yy+line_height_decades*yy],color='black')

            #Keep track of threshold points
            threshold_xx.append(xx)
            threshold_yy.append(yy)

    num_thresholds = len(threshold_xx)
    if(num_thresholds > 1):
        line_xx = []
        line_yy = []
        for ii in range(num_thresholds-1):
            #Point representing value
            line_xx.append(threshold_xx[ii])
            line_yy.append(threshold_yy[ii])

            #Point keeping yy the same but moving to next xx
            line_xx.append(threshold_xx[ii+1])
            line_yy.append(threshold_yy[ii])

        line_xx.append(threshold_xx[num_thresholds-1])
        line_yy.append(threshold_yy[num_thresholds-1])
        fig.line(x=line_xx,y=line_yy,color='black',line_dash='dashed')

def main(args):

    #Read in threshold file if enabled
    if(args.threshold_csv):
        threshold_data = pd.read_csv(args.threshold_csv,na_values="     None")
        line_height_decades = 0.25
        threshold_legend_fig = threshold_legend()

    #Initialize to generic file index if not specified
    labels = args.labels
    if not labels:
        labels = ["Unlabeled File %02i"%ii for ii in range(len(args.input_csvs))]
    while(len(labels) < len(args.input_csvs)):
        labels.append("Unlabeled File %02i"%(len(labels)-1))

    df_list = []
    for input_csv in args.input_csvs:
        data = pd.read_csv(input_csv,na_values="     None")

        #Apply time filter to start_tir
        data['start_tir'] = data['desired_tir'] + data['rx_start_deadline']*1e-6
        data['end_tir'] = data['desired_tir'] + data['rx_end_deadline']*1e-6
        data = data[(data.start_tir<args.max_duration)&(data.start_tir>=args.ignore_duration)]

        df_list.append(data)

    fig_list = []
    width = 1200
    height = 600
    legend_fig = comparison_legend(labels)
    ccdf_fig1 = ccdf_comparison(df_list,'estimated_sustained_throughput_gbps',
                                no_complement=True,
                                height=height,
                                width=width,
                                xlabel="Estimated Sustained Throughput (Gbps)",
                                xdelta=1,
                                xstart=0,
                                xstop=200)

    fig_list.append(column([row([ccdf_fig1,legend_fig])]))

    ccdf_fig1 = ccdf_comparison(df_list,'rx_start_deadline',
                                height=height,
                                width=width,
                                xlabel="First Packet Time Relative To Desired (usec)",
                                xdelta=0.1,
                                xstart=0,
                                xstop=60)
    if(args.threshold_csv):
        add_thresholds(ccdf_fig1,threshold_data,"rx_start_q")

        fig_list.append(column([row([ccdf_fig1,column([legend_fig,threshold_legend_fig])])]))
    else:
        fig_list.append(column([row([ccdf_fig1,column([legend_fig])])]))

    ccdf_fig1 = ccdf_comparison(df_list,'rx_end_deadline',
                                height=height,
                                width=width,
                                xlabel="Last Packet Time Relative To Desired (usec)",
                                xdelta=0.1,
                                xstart=0,
                                xstop=60)
    if(args.threshold_csv):
        add_thresholds(ccdf_fig1,threshold_data,"rx_end_q")

        fig_list.append(column([row([ccdf_fig1,column([legend_fig,threshold_legend_fig])])]))
    else:
        fig_list.append(column([row([ccdf_fig1,column([legend_fig])])]))

    ccdf_fig1 = ccdf_comparison(df_list,'rx_duration',
                                height=height,
                                width=width,
                                xlabel="Transfer duration (usec)",
                                xdelta=0.1,
                                xstart=0,
                                xstop=60)
    fig_list.append(column([row([ccdf_fig1,column([legend_fig])])]))

    #Output plots
    if(args.out_filename):
        out_filename = args.out_filename
    else:
        out_filename = "result.html"
    output_file(filename=out_filename, title=os.path.split(__file__)[1])
    print("Writing output to %s"%out_filename)
    if(args.out_filename):
        save(column(fig_list))
    else:
        show(column(fig_list))


if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0

    parser = argparse.ArgumentParser(
        description="Plots CCDFs of throughputs and packet timelines.  Overlays threshold data points if specified."
    )
    parser.add_argument(
        "input_csvs", type=str, nargs="+", help="List of CSV file parsed by rxlogs2csv"
    )
    parser.add_argument(
        "-m", "--max_duration", default=default_max_duration, type=float, help="Maximum run duration to process"
    )
    parser.add_argument(
        "-i", "--ignore_duration", default=default_ignore_duration, type=float, help="Number of seconds at beginning of file to ignore"
    )
    parser.add_argument(
        "-o", "--out_filename", type=str, help="Output html file"
    )
    parser.add_argument(
        "-l", "--labels", type=str, nargs="+", help="Names for each dataset"
    )
    parser.add_argument(
        "-t", "--threshold_csv", help="Overlays data from threshold file on CCDFs"
    )
    args = parser.parse_args()

    main(args)
