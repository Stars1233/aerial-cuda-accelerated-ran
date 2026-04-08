#!/usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re

# Constants
C_PLANE_TX = "C-plane"
CALLBACK = "Callback"
CPU_CUDA_TIME = "CPU CUDA time"
CPU_CUDA_SETUP = "CPU CUDA Setup"
CPU_CUDA_RUN = "CPU CUDA Run"
CPU_FREE_U_PLANE = "CPU Free U-plane"
CPU_RX_U_PLANE = "CPU RX U-plane"
CPU_WAIT_GPU_TIME = "CPU wait GPU Time"
DL_FH_COMPRESSION = "DL FH Compression"
DL = "DL"
GPU_ORDER_IDLE = "GPU Order Idle"
GPU_ORDER_RUN = "GPU Order Run"
GPU_RUN = "GPU Run"
GPU_SETUP = "GPU Setup"
GPU_SETUP_PH1 = "GPU Setup Ph1"
GPU_SETUP_PH2 = "GPU Setup Ph2"
MAX_GPU_TIME = "Max GPU Time"
ORDER_KERNEL = "Order Kernel"
PBCH = "PBCH"
PDCCH_DL = "PDCCH DL"
PDCCH_UL = "PDCCH UL"
PDSCH = "PDSCH"
PUSCH = "PUSCH"
PUCCH = "PUCCH"
CSIRS = "CSI-RS"
PRACH = "PRACH"
U_PLANE_PREPARE =  "U-plane prepare"
U_PLANE_TX = "U-plane tx"
UL = "UL"

LATENCY_CATEGORIES_NEVER_CELL_AGGREGATED = {DL, UL, ORDER_KERNEL, DL_FH_COMPRESSION}

# Structure for tracking latencies across different channels and CPU/GPU activities for each channel
LATENCIES = { 
    DL : {
        C_PLANE_TX                : None,
        U_PLANE_PREPARE           : None,
        U_PLANE_TX                : None,
    },
    UL : {
        C_PLANE_TX                : None,
        CPU_FREE_U_PLANE          : None,
        CPU_RX_U_PLANE            : None,
    },
    PDSCH : {
        CPU_CUDA_SETUP            : None,
        CPU_CUDA_RUN              : None,
        CPU_WAIT_GPU_TIME         : None,
        GPU_RUN                   : None,
        GPU_SETUP                 : None,
        # MAX_GPU_TIME              : None,
    },
    PBCH : {
        CPU_CUDA_SETUP            : None,
        CPU_CUDA_RUN              : None,
        CPU_WAIT_GPU_TIME         : None,
        GPU_RUN                   : None,
        GPU_SETUP                 : None,
        # MAX_GPU_TIME              : None,
    },
    PDCCH_DL : {
        CPU_CUDA_SETUP            : None,
        CPU_CUDA_RUN              : None,
        CPU_WAIT_GPU_TIME         : None,
        GPU_RUN                   : None,
        GPU_SETUP                 : None,
        # MAX_GPU_TIME              : None,
    },
    PDCCH_UL : {
        CPU_CUDA_SETUP            : None,
        CPU_CUDA_RUN              : None,
        CPU_WAIT_GPU_TIME         : None,
        GPU_RUN                   : None,
        GPU_SETUP                 : None,
        # MAX_GPU_TIME              : None,
    },
    CSIRS : {
        CPU_CUDA_SETUP            : None,
        CPU_CUDA_RUN              : None,
        CPU_WAIT_GPU_TIME         : None,
        GPU_RUN                   : None,
        GPU_SETUP                 : None,
        # MAX_GPU_TIME              : None,
    },
    PUSCH : {
        CPU_CUDA_SETUP            : None,
        CPU_CUDA_RUN              : None,
        CPU_WAIT_GPU_TIME         : None,
        GPU_RUN                   : None,
        GPU_SETUP_PH1             : None,
        GPU_SETUP_PH2             : None,
        # MAX_GPU_TIME              : None,
        CALLBACK                  : None,
    },
    PUCCH : {
        CPU_CUDA_SETUP            : None,
        CPU_CUDA_RUN              : None,
        CPU_WAIT_GPU_TIME         : None,
        GPU_RUN                   : None,
        GPU_SETUP                 : None,
        # MAX_GPU_TIME              : None,
        CALLBACK                  : None,
    },
    PRACH : {
        CPU_CUDA_TIME             : None,
        CPU_WAIT_GPU_TIME         : None,
        GPU_RUN                   : None,
        GPU_SETUP                 : None,
        # MAX_GPU_TIME              : None,
        CALLBACK                  : None,
    },
    ORDER_KERNEL : {
        CPU_CUDA_TIME             : None,
        GPU_ORDER_IDLE            : None,
        GPU_ORDER_RUN             : None,
    },
    DL_FH_COMPRESSION : {
        CPU_CUDA_TIME             : None,
        CPU_WAIT_GPU_TIME         : None,
    },
}


def print_delimiter(name):
    print('\n#####', (name + " (us) ").ljust(45, '#'))


# Collect all unique Cell IDs from the PHY log
def find_all_unique_cell_ids(lines):
    cell_ids = set()

    for line in lines:
        found = re.search(r"Cell \d+ with MU", line)
        if found:
            cell_ids.add(int(found.group().split(" ")[1]))

    return cell_ids


# Prepare LATENCIES structure for tracking latencies across channels and activities within the channel.
# Optionally, track each Cell's latencies separately
def prepare_for_collecting_latencies(cell_ids, aggregate):
    for category in LATENCIES:
        for op in LATENCIES[category]:
            if category in LATENCY_CATEGORIES_NEVER_CELL_AGGREGATED or not aggregate:
                LATENCIES[category][op] = {}
                for cell_id in cell_ids:
                    LATENCIES[category][op][cell_id] = []
            else:
                LATENCIES[category][op] = []
                

# Optionally, track each Cell's latencies separately
def parse_latency(line, name):
    regex = r"\[" + name + r"\].*?DURATION: \d+" 
    found = re.search(regex, line)
    if found:
        value = found.group().split(" ")[-1]
        return int(value)
    

# Find Cell ID in current line
def find_cell_id(line):
    found = re.search(r"Cell \d+", line)
    if found:
        value = found.group().split(" ")[-1]
        return int(value)
    
    return -1


# Find latency value in the current line given the 'name' of latency
def collect_latency(line, cell_id, category, name):
    lat = parse_latency(line, name)
    if (lat != None) and (lat > 0):
        if cell_id != -1 and isinstance(LATENCIES[category][name], dict):
            LATENCIES[category][name][cell_id].append(lat)
        else:
            LATENCIES[category][name].append(lat)


# Populate the LATENCIES structure with all latencies found in PHY log
def collect_all_latencies(lines):
    for line in lines:
        cell_id = find_cell_id(line)

        if re.search(r"\[PHYDRV\] SFN .* DL Communication Tasks", line):
            collect_latency(line, cell_id, DL, C_PLANE_TX)
            collect_latency(line, cell_id, DL, U_PLANE_PREPARE)
            collect_latency(line, cell_id, DL, U_PLANE_TX)

        elif re.search(r"\[PHYDRV\] SFN .* COMPRESSION DL", line):
            collect_latency(line, cell_id, DL_FH_COMPRESSION, CPU_CUDA_TIME)
            collect_latency(line, cell_id, DL_FH_COMPRESSION, CPU_WAIT_GPU_TIME)

        elif re.search(r"\[PHYDRV\] SFN .* UL Communication Tasks", line):
            category = UL
            collect_latency(line, cell_id, UL, C_PLANE_TX)

        elif re.search(r"\[PHYDRV\] SFN .* PDSCH", line):
            collect_latency(line, cell_id, PDSCH, CPU_CUDA_SETUP)
            collect_latency(line, cell_id, PDSCH, CPU_CUDA_RUN)
            collect_latency(line, cell_id, PDSCH, GPU_SETUP)
            collect_latency(line, cell_id, PDSCH, GPU_RUN)
            # collect_latency(line, cell_id, PDSCH, MAX_GPU_TIME)
            collect_latency(line, cell_id, PDSCH, CPU_WAIT_GPU_TIME)

        elif re.search(r"\[PHYDRV\] SFN .* PBCH", line):
            collect_latency(line, cell_id, PBCH, CPU_CUDA_SETUP)
            collect_latency(line, cell_id, PBCH, CPU_CUDA_RUN)
            collect_latency(line, cell_id, PBCH, GPU_SETUP)
            collect_latency(line, cell_id, PBCH, GPU_RUN)
            # collect_latency(line, cell_id, PBCH, MAX_GPU_TIME)
            collect_latency(line, cell_id, PBCH, CPU_WAIT_GPU_TIME)

        elif re.search(r"\[PHYDRV\] SFN .* PDCCH DL", line):
            collect_latency(line, cell_id, PDCCH_DL, CPU_CUDA_SETUP)
            collect_latency(line, cell_id, PDCCH_DL, CPU_CUDA_RUN)
            collect_latency(line, cell_id, PDCCH_DL, GPU_SETUP)
            collect_latency(line, cell_id, PDCCH_DL, GPU_RUN)
            # collect_latency(line, cell_id, PDCCH_DL, MAX_GPU_TIME)
            collect_latency(line, cell_id, PDCCH_DL, CPU_WAIT_GPU_TIME)

        elif re.search(r"\[PHYDRV\] SFN .* PDCCH UL", line):
            collect_latency(line, cell_id, PDCCH_UL, CPU_CUDA_SETUP)
            collect_latency(line, cell_id, PDCCH_UL, CPU_CUDA_RUN)
            collect_latency(line, cell_id, PDCCH_UL, GPU_SETUP)
            collect_latency(line, cell_id, PDCCH_UL, GPU_RUN)
            # collect_latency(line, cell_id, PDCCH_UL, MAX_GPU_TIME)
            collect_latency(line, cell_id, PDCCH_UL, CPU_WAIT_GPU_TIME)

        elif re.search(r"\[PHYDRV\] SFN .* CSI-RS", line):
            collect_latency(line, cell_id, CSIRS, CPU_CUDA_SETUP)
            collect_latency(line, cell_id, CSIRS, CPU_CUDA_RUN)
            collect_latency(line, cell_id, CSIRS, GPU_SETUP)
            collect_latency(line, cell_id, CSIRS, GPU_RUN)
            # collect_latency(line, cell_id, CSIRS, MAX_GPU_TIME)
            collect_latency(line, cell_id, CSIRS, CPU_WAIT_GPU_TIME)

        elif re.search(r"\[PHYDRV\] SFN .* PUSCH", line):
            collect_latency(line, cell_id, PUSCH, CPU_CUDA_SETUP)
            collect_latency(line, cell_id, PUSCH, CPU_CUDA_RUN)
            collect_latency(line, cell_id, PUSCH, GPU_SETUP_PH1)
            collect_latency(line, cell_id, PUSCH, GPU_SETUP_PH2)
            collect_latency(line, cell_id, PUSCH, GPU_RUN)
            # collect_latency(line, cell_id, PUSCH, MAX_GPU_TIME)
            collect_latency(line, cell_id, PUSCH, CPU_WAIT_GPU_TIME)
            collect_latency(line, cell_id, PUSCH, CALLBACK)

        elif re.search(r"\[PHYDRV\] SFN .* PUCCH", line):
            collect_latency(line, cell_id, PUCCH, CPU_CUDA_SETUP)
            collect_latency(line, cell_id, PUCCH, CPU_CUDA_RUN)
            collect_latency(line, cell_id, PUCCH, GPU_SETUP)
            collect_latency(line, cell_id, PUCCH, GPU_RUN)
            # collect_latency(line, cell_id, PUCCH, MAX_GPU_TIME)
            collect_latency(line, cell_id, PUCCH, CPU_WAIT_GPU_TIME)
            collect_latency(line, cell_id, PUCCH, CALLBACK)

        elif re.search(r"\[PHYDRV\] SFN .* PRACH", line):
            collect_latency(line, cell_id, PRACH, CPU_CUDA_TIME)
            collect_latency(line, cell_id, PRACH, GPU_SETUP)
            collect_latency(line, cell_id, PRACH, GPU_RUN)
            # collect_latency(line, cell_id, PRACH, MAX_GPU_TIME)
            collect_latency(line, cell_id, PRACH, CPU_WAIT_GPU_TIME)
            collect_latency(line, cell_id, PRACH, CALLBACK)

        elif re.search(r"\[PHYDRV\] SFN .* ORDER", line):
            collect_latency(line, cell_id, UL, CPU_RX_U_PLANE)
            collect_latency(line, cell_id, UL, CPU_FREE_U_PLANE)
            collect_latency(line, cell_id, ORDER_KERNEL, CPU_CUDA_TIME)
            collect_latency(line, cell_id, ORDER_KERNEL, GPU_ORDER_IDLE)
            collect_latency(line, cell_id, ORDER_KERNEL, GPU_ORDER_RUN)


# Reduce latencies for a given channel and activity into a number of satistics
def summarize_latencies(arr, percentile):
    mean = np.mean(arr)
    stddev = np.std(arr)
    percentile = np.percentile(arr, percentile)
    return (mean, stddev, percentile)
    

# Print all latencies to console
def print_latencies(percentile):
    for category in LATENCIES:
        print_delimiter(category)

        for op in LATENCIES[category]:
            if not len(LATENCIES[category][op]):
                continue

            print(op)

            if isinstance(LATENCIES[category][op], list):
                arr = np.array(LATENCIES[category][op])
                if arr.size > 0:
                    (mean, stddev, percentile_value) = summarize_latencies(arr, percentile)
                    print("  Avg = {:<6.2f}   Stddev = {:<6.2f}   {:<2.1f}% = {:<6.2f}".format(mean, stddev, percentile, percentile_value))
            else:
                for cell_id in LATENCIES[category][op]:
                    arr = np.array(LATENCIES[category][op][cell_id])
                    if arr.size > 0:
                        (mean, stddev, percentile_value) = summarize_latencies(arr, percentile)
                        print("  Cell {:<3d}:   Avg = {:<6.2f}   Stddev = {:<6.2f}   {:<2.1f}% = {:<6.2f}".format(cell_id, mean, stddev, percentile, percentile_value))


# Plot latencies for a given channel and activity
def plot_latency_distribution(latencies, title, plot, thres = -1):
    arr = np.array(latencies)
    sz = arr.size
    y, x = np.histogram(arr, bins=10000)
    
    if thres >= 0:
        plt.axvline(x=thres, color='black')

    cy = np.cumsum(y) / sz
    plt.plot(x[1:], cy)
    plt.xlabel("Latency [us]")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    filename = title.replace(" ", "_").replace(",", "").replace(":", "") + ".png"
    
    if plot == "save":
        print("Saving figure: " + filename)
        plt.savefig(filename)
        plt.figure()
    else:
        plt.show()


# Plot latency distributions accross all channels, activities (and cells)
def plot_latency_histograms(test_case, plot):
    for category in LATENCIES:
        for op in LATENCIES[category]:
            if not len(LATENCIES[category][op]):
                continue

            if isinstance(LATENCIES[category][op], list):
                arr = np.array(LATENCIES[category][op])
                if arr.size > 0:
                    title = test_case + ": " + category + " " + op
                    plot_latency_distribution(arr, title, plot)
                    
            else:
                for cell_id in LATENCIES[category][op]:
                    arr = np.array(LATENCIES[category][op][cell_id])
                    if arr.size > 0:
                        title = test_case + ": " + category + " " + op + ", Cell " + str(cell_id)
                        plot_latency_distribution(arr, title, plot)



def main(args):
    with open(args.phylog) as f:
        lines = f.readlines()

    cell_ids = find_all_unique_cell_ids(lines)
    prepare_for_collecting_latencies(cell_ids, args.aggregate)
    collect_all_latencies(lines)
    print_latencies(args.percentile)

    if args.plot and args.aggregate:
        plot_latency_histograms(args.test_case, args.plot)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='phy_latencies.py', description="Show latency stats from PHY log", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("phylog", help="input PHY log file", type=str)
    parser.add_argument("-a", "--aggregate", help="show aggregate latencies for all cells", action='store_true')
    parser.add_argument("-p", "--plot", help="plot latency histogram CDFs", type=str, choices=["save", "show"])
    parser.add_argument("--percentile", help="Percentile value to show", type=float, default=99.5)
    parser.add_argument("-t", "--test_case", help="name of the test case in the PHY log", type=str, default='TC')
    args = parser.parse_args()
    main(args)
