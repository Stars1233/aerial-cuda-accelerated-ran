# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from bokeh.plotting import figure, show, output_file, save
from bokeh.models import BoxAnnotation, Span, Label
from bokeh.layouts import column
from bokeh.io import export_png
import numpy as np
import json
import argparse
import time

def plot(filename, data, cells, bandwidth=273):

    # start_time = time.time()

    types = data["types"]

    color_mapping = {}
    color_mapping["ssb"] = "#FFC000"
    color_mapping["pdsch"] = "#9933FF"
    color_mapping["pdcch"] = "#70AD47"
    color_mapping["csirs"] = "#8EA9DB"
    color_mapping["csirs_flex_0"] = "#8EA9DB" 
    color_mapping["csirs_flex_1"] = "#8EA9DB"
    color_mapping["pucch"] = "#e404ba"
    color_mapping["prach"] = "#ddd5be"
    color_mapping["pusch"] = "#3d85c6"
    color_mapping["srs"] = "#FF5733"
    color_mapping["w/ bfw"] = "#A50026" 

    bfw_x_coords = []
    bfw_y_coord = 270      # Adjust the y-coordinate as needed

    # populate data
    nSlots = len(data["sequence"])
    nFrame = nSlots // 20
    bars = {}
    for cell in range(cells):
        bars[cell] = {}
        for slot, type in enumerate(data["sequence"]):
            frame = slot // 20
            slotInFrame = slot % 20
            bars[cell].setdefault(frame, {})
            for channels in types[type]:                
                if 'comment' in channels:
                    continue                
                
                if ("csirs_flex" in channels) and (int(channels[-1]) != cell%2):
                    continue

                bars[cell][frame].setdefault(channels, {"x": [], "bottom": [], "top": []})
                for channel in types[type][channels]:
                    x = np.arange(channel[0],channel[1]) + 0.5 + slotInFrame*14 
                    bars[cell][frame][channels]["x"] = bars[cell][frame][channels]["x"] + list(x)
                    bars[cell][frame][channels]["bottom"] = bars[cell][frame][channels]["bottom"] + [channel[2]]*len(x)
                    bars[cell][frame][channels]["top"] = bars[cell][frame][channels]["top"] + [channel[3]]*len(x)
            
            # if slotInFrame in data["bfw_slots"]:
            #     bfw_x_coords.append(slotInFrame*14//2)

            # if slotInFrame in data["bfw_slots"]:
            #     x = np.arange(3, 11) + 0.5 + slotInFrame*14 
            #     bars[cell][frame].setdefault("w/ bfw", {"x": [], "bottom": [], "top": []})
            #     bars[cell][frame]["w/ bfw"]["x"] = bars[cell][frame]["w/ bfw"]["x"] + list(x)
            #     bars[cell][frame]["w/ bfw"]["bottom"] = bars[cell][frame]["w/ bfw"]["bottom"] + [bandwidth+5]*len(x)
            #     bars[cell][frame]["w/ bfw"]["top"] = bars[cell][frame]["w/ bfw"]["top"] + [bandwidth+15]*len(x)
                    
    # create a new plot with a title and axis labels
    p = [figure(title=f"Pattern cell {i}, frame {j}", x_axis_label='Symbols', y_axis_label='PRBs', sizing_mode="stretch_width",
    height=350, x_range=(0, 14*20), y_range=(0, bandwidth+18)) for j in range(nFrame) for i in range(cells)]

    for frame in range(nFrame):
        for cell in range(cells):        
            for channels in bars[cell][frame].keys():
                p[frame*cells + cell].vbar(x=bars[cell][frame][channels]["x"], 
                            bottom=bars[cell][frame][channels]["bottom"], 
                            top=bars[cell][frame][channels]["top"], 
                            legend_label=channels, width=0.9, color=color_mapping[channels])

            p[frame*cells + cell].xaxis.ticker = np.arange(0,14*20,7)
            p[frame*cells + cell].xaxis.major_label_overrides = {7+i*14: f' \n slot {i}' for i in range(20)}

            xs = [[i,i] for i in range(0,14*20,14)] 
            ys = [[0,bandwidth]] * len(xs)    
            p[frame*cells + cell].multi_line(xs, ys, line_color='black', line_dash='dashed', line_width=1.5)

            # Add labels to the plot
            # for x in bfw_x_coords:
            #     label = Label(x=x, y=bfw_y_coord, text="BFW", text_align='center', text_baseline='bottom')
            #     p[frame*cells + cell].add_layout(label)

    # end_time = time.time()
    # print(end_time-start_time)

    show(column(p,sizing_mode='stretch_both'))

    filename=filename.replace('json','html')
    filename=filename.replace('POC2','pattern_plot_POC2')
    output_file(filename)
    save(p)


base = argparse.ArgumentParser()
base.add_argument(
    "--config",
    type=str,
    nargs="+",
    dest="config",
    help="Specifies the configuration file",
    required=True,
)
base.add_argument(
    "--prb",
    type=int,
    dest="nPrb",
    help="Specifies the bandwidth in PRB number",
    required=False,
)
base.add_argument(
    "--cells",
    type=int,
    dest="cells",
    default=2,
    help="Specifies the number of cells (default: 2)",
    required=False,
)
args = base.parse_args()


for config in args.config:

    ifile = open(config, "r")
    data = json.load(ifile)
    ifile.close()

    cells = args.cells

    if args.nPrb is not None:
        plot(config, data, cells, args.nPrb)
    else:
        plot(config, data, cells, 273)