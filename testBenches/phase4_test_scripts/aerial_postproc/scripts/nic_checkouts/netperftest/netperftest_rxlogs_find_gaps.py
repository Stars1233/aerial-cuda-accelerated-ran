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

from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row

def find_gaps(first_packet_rx_log,last_packet_rx_log):
    df1 = pd.read_csv(first_packet_rx_log)
    df2 = pd.read_csv(last_packet_rx_log)

    df1_gaps = df1[~df1.burst_id.isin(df2.burst_id)]
    df2_gaps = df2[~df2.burst_id.isin(df1.burst_id)]

    ref_t0 = min(df1.desired_tx_timestamp.min(),df2.desired_tx_timestamp.min())
    df1_gaps = df1_gaps.assign(tir=(df1_gaps.desired_tx_timestamp - ref_t0)/1e9)
    df2_gaps = df2_gaps.assign(tir=(df2_gaps.desired_tx_timestamp - ref_t0)/1e9)

    fig = figure(height=400,width=1000,title='Gap Locations')
    source1 = ColumnDataSource(df1_gaps)
    source2 = ColumnDataSource(df2_gaps)
    fig.circle(x='tir',y='txq',source=source1, color='green',legend_label='First Packet Gap')
    fig.circle(x='tir',y='txq',source=source2, color='red', legend_label='Last Packet Gap')

    fig.xaxis.axis_label = 'Time In Run (sec)'
    fig.yaxis.axis_label = 'TXQ'
    
    output_file(filename="result.html", title="Find Gaps Results")
    show(fig)



def main(args):
    df_gaps = find_gaps(args.first_packet_rx_log,args.last_packet_rx_log)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Finds location of gaps in netperftest RX logs"
    )
    parser.add_argument(
        "first_packet_rx_log", type=str, help="Log containing first packet information (typically log_rxq0.csv)"
    )
    parser.add_argument(
        "last_packet_rx_log", type=str, help="Log containing last packet information (for 2 RX ques - log_rxq1.csv)"
    )
    args = parser.parse_args()

    main(args)


    
