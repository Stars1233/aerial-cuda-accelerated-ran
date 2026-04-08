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

#Parses data to an output folder
import os
import time

def main(args):
    time1 = time.time()

    format_found = True
    if(args.format.lower() == "perfmetrics"):
        from aerial_postproc.parsenator_formats import PerfMetricsIO
        io_class = PerfMetricsIO()
    elif(args.format.lower() == "latencysummary"):
        from aerial_postproc.parsenator_formats import LatencySummaryIO
        io_class = LatencySummaryIO(mmimo_enable=args.mmimo_enable)
    else:
        print("ERROR :: Unknown format %s - not processing data"%args.format)
        format_found = False


    if(format_found):
        if(len(args.input_logs) == 3 and os.path.isfile(args.input_logs[0]) and os.path.isfile(args.input_logs[1])  and os.path.isfile(args.input_logs[2])):
            parse_success = io_class.parse(args.input_logs,args.ignore_duration,args.max_duration,num_proc=args.num_proc)
            if parse_success:   
                save_success = io_class.save(args.output_folder,str_output_format=args.bin_format)
                if not save_success:
                    print("Error :: Problem saving data to %s"%args.output_folder)
            else:
                print("Error :: Problem parsing %s"%args.input_logs)
        else:
            print("Error :: Unable to find all three log files (phy/testmac/ru)")
        
    else:
        print("ERROR :: Problem finding phy/testmac/ru in %s for %s processing..."%(args.input_logs,type(io_class).__name__))

    time2 = time.time()

    print("Parsing complete in %.1f sec."%(time2-time1))


if __name__ == "__main__":
    import argparse
    default_output_folder="./"
    default_format="PerfMetrics"
    default_bin_format="feather"
    default_max_duration=999999999.0
    default_ignore_duration=0.0
    default_num_proc=8

    parser = argparse.ArgumentParser(
        description="Analyze PHY/RU log to determine worst case 1% CCDF performance.  Optionally generate output files that CICD can use to manage merge gating"
    )
    parser.add_argument(
        "input_logs", type=str, nargs="+", help="Set of log files needed for processing: {PerfMetrics->phy/test/ru}"
    )
    parser.add_argument(
        "-o", "--output_folder", default=default_output_folder, help="Output folder to place data"
    )
    parser.add_argument(
        "-f", "--format", default=default_format, help="Format to parse (options: {PerfMetrics,LatencySummary})"
    )
    parser.add_argument(
        "-b", "--bin_format", default=default_bin_format, help="Output binary format (options: {feather,parquet,hdf5,json,csv})"
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
        "-e", "--mmimo_enable", action="store_true", help="Modifies timeline requirement to mmimo settings"
    )
    args = parser.parse_args()

    main(args)
