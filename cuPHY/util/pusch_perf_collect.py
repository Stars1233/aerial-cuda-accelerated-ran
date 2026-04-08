#!/usr/bin/env python

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

# Examples of running:
# ../util/pusch_perf_collect.py -d ../../cuPHY_data/perf
# CUDA_VISIBLE_DEVICES=3 ../util/pusch_perf_collect.py -d ../../cuPHY_data/perf -m bler
#
# Comparing to previous runs:
#     First run: collect data in an output file called results0.txt (-o argument)
#         ../util/pusch_perf_collect.py -d ../../cuPHY_data/perf -m all -o results0.txt
#     Subsequent runs: compare new data to previously collected results (-c argument):
#         ../util/pusch_perf_collect.py -d ../../cuPHY_data/perf -m all -c results0.txt

import os
import argparse
import re
import subprocess
import sys

#***********************************************************************
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir",             default=".",                                                           help="Directory containing input files")
parser.add_argument("-e", "--exe",                  default="./examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe", help="Executable program to run")
parser.add_argument("-m", "--mode",                 default="all",                                                         help="perf collection mode ('latency', 'bler', or 'all')")
parser.add_argument("-r", "--num_runs", type=int,   default=100,                                                           help="Number of runs to time (latency or all modes only)")
parser.add_argument("-o", "--out_file",             default="",                                                            help="Output table file name ('all' mode only)")
parser.add_argument("-c", "--compare_file",         default="",                                                            help="Comparison file (from previous run) ('all' mode only)")
parser.add_argument("-v", "--verbose",              action="store_true",                                                   help="Show all program output, instead of just significant lines")
args = parser.parse_args()

#***********************************************************************
# Tuples containing file parameters:
#               F   -   snr MIMO   PRB  Sym QAM
perf_descs = [( 1,  1,  40, 1,  4, 104, 13, 256),
              ( 1,  2,  40, 1,  4, 104, 13, 256),
              ( 1,  3,  40, 1,  4, 104, 13, 256),
              ( 1,  4,  40, 1,  4, 104, 13, 256),
              ( 1,  5,  40, 1,  4, 104, 13, 256),
              ( 1,  6,  40, 1,  4, 104, 13, 256),
              ( 1,  7,  40, 1,  4, 104, 13, 256),
              ( 1,  8,  40, 1,  4, 104, 13, 256),
              ( 1,  9,  40, 1,  4, 104, 13,  64),
              ( 1, 10,  40, 1,  4, 104, 13,  64),
              ( 1, 11,  40, 1,  4, 104, 13,  64),
              ( 1, 12,  40, 1,  4, 104, 13,  64),
              ( 1, 13,  40, 1,  4, 104, 13,  64),
              ( 1, 14,  40, 1,  4, 104, 13,  64),
              ( 1, 15,  40, 1,  4, 104, 13,  64),
              ( 1, 16,  40, 1,  4, 104, 13,  64),
              ( 1, 17,  40, 1,  4, 104, 13,  64),
              ( 1, 18,  40, 1,  4, 104, 13,  16),
              ( 1, 19,  40, 1,  4, 104, 13,  64),
              ( 1, 20,  40, 1,  4, 104, 13,  16),
              ( 1, 21,  40, 1,  4, 104, 13,  16),
              ( 1, 22,  40, 1,  4, 104, 13,  16),
              ( 1, 23,  40, 1,  4, 104, 13,  16),
              (14,  1,  40, 8, 16, 272, 10, 256),
              (14,  2,  40, 8, 16, 272, 10, 256),
              (14,  3,  40, 8, 16, 272, 10, 256),
              (14,  4,  40, 8, 16, 272, 10, 256),
              (14,  5,  40, 8, 16, 272, 10, 256),
              (14,  6,  40, 8, 16, 272, 10, 256),
              (14,  7,  40, 8, 16, 272, 10, 256),
              (14,  8,  40, 8, 16, 272, 10, 256),
              (14,  9,  40, 8, 16, 272, 10,  64),
              (14, 10,  40, 8, 16, 272, 10,  64),
              (14, 11,  40, 8, 16, 272, 10,  64),
              (14, 12,  40, 8, 16, 272, 10,  64),
              (14, 13,  40, 8, 16, 272, 10,  64),
              (14, 14,  40, 8, 16, 272, 10,  64),
              (14, 15,  40, 8, 16, 272, 10,  64),
              (14, 16,  40, 8, 16, 272, 10,  64),
              (14, 17,  40, 8, 16, 272, 10,  64),
              (14, 18,  40, 8, 16, 272, 10,  16),
              (14, 19,  40, 8, 16, 272, 10,  16),
              (14, 20,  40, 8, 16, 272, 10,  16),
              (14, 21,  40, 8, 16, 272, 10,  16),
              (14, 22,  40, 8, 16, 272, 10,  16),
              (14, 23,  40, 8, 16, 272, 10,  16)]

########################################################################
# get_comparison_results()
# Read table results from a previous run
def get_comparison_results(cfile):
    res = {}
    f = open(cfile, 'r')
    while True:
        line = f.readline().strip()
        if not line:
            break
        if line.startswith('#'):
            continue
        fields = line.split()
        fname = fields[0]
        #print(fields)
        res[fname] = {}
        res[fname]['error_CBs'] = int(fields[1])
        res[fname]['latency']   = float(fields[2])
    return res

########################################################################
# gen_cmd_str()
# Generate a command string to run the example program
def gen_cmd_str(args, infile, params):
    return '%s -r %d %s -i %s' % (args.exe, params['num_runs'], params['wait_str'], infile)
    
#re_BLER_str    = r'PuschRx Pipeline\[(\d+)\]: Metric - Block Error Rate\s+: (.+) \(Error CBs (\d+), Total CBs (\d+)\)'
re_BLER_str    = r'Cell # (\d+) : Metric - Block Error Rate\s+: (.+) \(Error CBs (\d+), Total CBs (\d+)\)'
re_latency_str = r'PuschRx Pipeline\[(\d+)\]: Metric - Average execution time: (.+) usec'

params = {'num_runs' : args.num_runs,
          'wait_str' : ''}

if args.mode == 'bler':
    # In BLER-only mode, modify command line arguments to run a little faster...
    params['num_runs'] = 1
    params['wait_str'] = '-w 0'

results = []

for (idx, d) in enumerate(perf_descs):
    fname = 'TV_cuphy_F%02i-US-%02i_snrdb%.2f_MIMO%ix%i_PRB%i_DataSyms%i_qam%i.h5' % d
    fpath = os.path.join(args.data_dir, fname)
    cmd = gen_cmd_str(args, fpath, params)
    print('#-----------------------')
    print('# %d of %d' % (idx + 1, len(perf_descs)))
    print(cmd)
    #-------------------------------------------------------------------
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    res = {}
    while True:
        line = proc.stdout.readline().decode('ascii')
        if line:
            m = re.search(re_BLER_str, line)
            if args.verbose:
                print(line.strip())
            if m:
                if args.mode in ['all', 'bler'] and not args.verbose:
                    print(line.strip())
                #print(m.group(1), m.group(2), m.group(3))
                # Not currently using group(1) = pipeline
                res['BLER']      = float(m.group(2))
                res['error_CBs'] = int(m.group(3))
                res['total_CBs'] = int(m.group(4))
                #print(res)
            else:
                m = re.search(re_latency_str, line)
                if m:
                    if args.mode in ['all', 'latency']:
                        print(line.strip())
                    # Not currently using group(1) = pipeline
                    res['latency']   = float(m.group(2)) 
        else:
            break
    if not res:
        raise RuntimeError("No output results obtained for input file: '%s'" % fpath)
    res['desc'] = d
    res['filename'] = fname
    results.append(res)

if args.mode == 'bler':
    error_list = [r for r in results if r['error_CBs'] > 0]
    if error_list:
        print('ERRORS OCCURRED IN THE FOLLOWING INPUT FILES:')
        for e in error_list:
            print('%s: %d error CBs (%d total CBs)' % (e['filename'], e['error_CBs'], e['total_CBs']))
    else:
        print('NO ERRORS (%d INPUT FILES)' % len(results))
else:
    if args.compare_file:
        max_pct_change = 0
        prev_res = get_comparison_results(args.compare_file)
        print('# FILENAME                                                             ERRCNT LATENCY PREV_LATENCY LATENCY_CHANGE PCT_CHANGE')
        for r in results:
            r['prev_latency'] = prev_res[r['filename']]['latency']
            r['latency_change'] = r['latency'] - r['prev_latency']
            r['latency_change_pct'] = 100.0 * r['latency_change'] / r['prev_latency']
            max_pct_change = max(max_pct_change, abs(r['latency_change_pct']))
        for r in results:
            print('%-70s %6d %6.1f %12.1f %15.1f %+10.1f' % (r['filename'], r['error_CBs'], r['latency'], r['prev_latency'], r['latency_change'], r['latency_change_pct']))
        print('Maximum percent latency change: %.1f %%' % max_pct_change)
    else:
        print('# FILENAME                                                             ERRCNT LATENCY')
        for r in results:
            print('%-70s %6d %6.1f' % (r['filename'], r['error_CBs'], r['latency']))
    if args.out_file:
        print("Writing output to '%s'" % args.out_file)
        f = open(args.out_file, 'w')
        f.write('# FILENAME                                                             ERRCNT LATENCY\n')
        for r in results:
            f.write('%-70s %6d %6.1f\n' % (r['filename'], r['error_CBs'], r['latency']))
        f.close()
        

