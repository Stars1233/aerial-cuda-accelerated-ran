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
import os
from bokeh.io import output_file, save, show
from bokeh.layouts import column
from aerial_postproc.logplot import stats_plot
from bokeh.models import CrosshairTool

def main(args):
    data = pd.read_csv(args.input_csv,na_values="     None")

    #Apply time filter to start_tir
    data['start_tir'] = data['desired_tir'] + data['rx_start_deadline']*1e-6
    data['end_tir'] = data['desired_tir'] + data['rx_end_deadline']*1e-6
    data = data[(data.start_tir<args.max_duration)&(data.start_tir>=args.ignore_duration)]

    fig_list = []    

    crosshair_tool = CrosshairTool()
    width = 1200
    height = 600
    fig1 = stats_plot(data,'desired_tir','estimated_sustained_throughput_gbps',xdelta=args.time_delta,yfield_name='Estimated DL (Gbps)',
                      title="Estimated DL Throughput vs Time",xlabel="Time In Run (sec)",ylabel="Estimated DL Throughput (Gbps)",
                      height=height,width=width,crosshair_tool=crosshair_tool,enable_q01=True)
    fig2 = stats_plot(data,'desired_tir','rx_duration',xdelta=args.time_delta,yfield_name='Transfer Duration (usec)',
                      title="Transfer Duration vs Time",xlabel="Time In Run (sec)",ylabel="Transfer Duration (usec)",
                      height=height,width=width,crosshair_tool=crosshair_tool,enable_q99=True,x_range=fig1.x_range)
    fig3 = stats_plot(data,'desired_tir','rx_start_deadline',xdelta=args.time_delta,yfield_name='Earliest Latency (usec)',
                      title="Earliest Packet Latency vs Time",xlabel="Time In Run (sec)",ylabel="Earliest Packet Latency (usec)",
                      height=height,width=width,crosshair_tool=crosshair_tool,enable_q99=True,x_range=fig1.x_range)
    fig4 = stats_plot(data,'desired_tir','rx_end_deadline',xdelta=args.time_delta,yfield_name='Latest Latency (usec)',
                      title="Latest Packet Latency vs Time",xlabel="Time In Run (sec)",ylabel="Latest Packet Latency (usec)",
                      height=height,width=width,crosshair_tool=crosshair_tool,enable_q99=True,x_range=fig1.x_range)
    fig_list.extend([fig1,fig2,fig3,fig4])

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

    print(data)

if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0
    default_time_delta = 100*(500e-6/14)

    parser = argparse.ArgumentParser(
        description="Produces a timeline plot using data collected by netperftest"
    )
    parser.add_argument(
        "input_csv", type=str, help="CSV file parsed by rxlogs2csv"
    )
    parser.add_argument(
        "-m", "--max_duration", default=default_max_duration, type=float, help="Maximum run duration to process"
    )
    parser.add_argument(
        "-i", "--ignore_duration", default=default_ignore_duration, type=float, help="Number of seconds at beginning of file to ignore"
    )
    parser.add_argument(
        "-t", "--time_delta", type=float, default=default_time_delta, help="Time delta used when computing each statistic (default=%e)"%default_time_delta
    )
    parser.add_argument(
        "-o", "--out_filename", type=str, help="Output html file"
    )
    args = parser.parse_args()

    main(args)
