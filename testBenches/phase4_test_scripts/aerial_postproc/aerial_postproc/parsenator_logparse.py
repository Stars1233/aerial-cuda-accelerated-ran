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

import os
from aerial_postproc.parsenator import Parsenator
from functools import partial
import pandas as pd
import inspect

from aerial_postproc.logparse import getReceptionWindow, TrafficType, TestCaseType

#Used for compare_logs and cicd_performance_metrics
def perf_metrics_parse(phy_filenames,testmac_filenames,ru_filenames,start_tir,end_tir,slot_list=None,num_proc=8):
    print("Running %s - %s"%(os.path.split(__file__)[1],inspect.stack()[0][3]))

    from aerial_postproc.logparse import get_ref_t0, parse_ti_line, parse_l2_times, parse_gpu_setup_and_run
    from aerial_postproc.logparse import parse_new_compression_message, parse_testmac_times
    from aerial_postproc.logparse import parse_phylog_tick_times, build_df_du_ontime, build_df_ru_ontime

    ### PARSE LOGS
    df_ti_list = []
    df_l2_list = []
    df_gpu_list = []
    df_compression_list = []
    df_testmac_list = []
    for phy_filename,testmac_filename,ru_filename in zip(phy_filenames,testmac_filenames,ru_filenames):

        print("Processing PHY log: %s"%(phy_filename))

        # Determine reference time and parsing bounds
        ref_t0 = get_ref_t0(phy_filename)
        p1 = Parsenator(phy_filename,
                        [parse_ti_line,parse_l2_times,partial(parse_gpu_setup_and_run,condensed_format=True), parse_new_compression_message],
                        [['t0_timestamp','sfn','slot','task','subtask','sequence','start_deadline','end_deadline','duration','cpu'], #Only necessary TI fields
                         ['t0_timestamp','slot','start_deadline','end_deadline','duration'], #Only necessary L2A fields
                         ['t0_timestamp','slot','task','cpu_setup_duration','cpu_run_duration','gpu_setup1_duration','gpu_setup2_duration',
                          'gpu_run_duration','gpu_run_eh_duration','gpu_run_noneh_duration','gpu_run_copy_duration','gpu_run_gap_duration','pdsch_h2d_copy_duration','gpu_total_duration','cell'], #Only necessary GPU fields
                         ['t0_timestamp','slot','task','cell','event','duration'], #Only necessary compression fields
                        ],
                        start_tir,end_tir,num_processes=num_proc,ref_t0=ref_t0)
    
        parsed_result = p1.parse()
        df_ti = parsed_result[0]
        df_l2 = parsed_result[1]
        df_gpu = parsed_result[2]
        df_compression = parsed_result[3]

        if(slot_list):
            # Save on memory
            df_ti = df_ti[df_ti.slot.isin(slot_list)]
            df_gpu = df_gpu[df_gpu.slot.isin(slot_list)]
            df_l2 = df_l2[df_l2.slot.isin(slot_list)]

        df_ti_list.append(df_ti)
        df_l2_list.append(df_l2)
        df_gpu_list.append(df_gpu)
        df_compression_list.append(df_compression)


        print("Processing TESTMAC log: %s"%(testmac_filename))

        # Parse testmac log
        p2 = Parsenator(testmac_filename,
                        [parse_testmac_times],
                        [['t0_timestamp','slot','fapi2_start_timestamp','fapi2_stop_timestamp']
                        ], #Only necessary TI fields???,
                        start_tir,end_tir,num_processes=num_proc,ref_t0=ref_t0)
        parsed_result = p2.parse()
        df_testmac = parsed_result[0]

        df_testmac_list.append(df_testmac)

    df_du_ontime_list = []
    df_ru_ontime_list = []
    for phy_filename, ru_filename in zip(phy_filenames, ru_filenames):
        df_du_ontime_list.append(build_df_du_ontime(phy_filename))
        df_ru_ontime_list.append(build_df_ru_ontime(ru_filename))

    #Read in ticks if available
    df_tick_list = []
    for phy_filename in phy_filenames:
        df_tick = parse_phylog_tick_times(phy_filename, max_lines=10000)
        if(len(df_tick) > 0):
            print("Tick times found, reading in tick times...")
            df_tick = parse_phylog_tick_times(phy_filename, max_duration=end_tir)
            df_tick_list.append(df_tick)
        else:
            print("No tick times found.")
            df_tick_list.append(pd.DataFrame())

    return df_ti_list,df_l2_list,df_gpu_list,df_compression_list,df_testmac_list,df_du_ontime_list,df_ru_ontime_list,df_tick_list


def latency_parsing(phy_filenames,ru_filenames,start_tir,end_tir,slot_list=None,num_proc=8,mmimo_enable=False):
    print("Running %s - %s"%(os.path.split(__file__)[1],inspect.stack()[0][3]))

    from aerial_postproc.logparse import get_ref_t0, parse_du_symbol_timings, parse_ti_line, parse_ru_tx_times, parse_ru_tx_sum_times, parse_ru_symbol_timings

    ### PARSE LOGS
    df_du_rx_timing_list = []
    df_du_rx_srs_timing_list = []
    df_ti_list = []
    df_tx_timing_list = []
    df_tx_timing_sum_list = []
    df_ru_rx_timing_list = []
    for phy_filename,ru_filename in zip(phy_filenames,ru_filenames):

        print("Processing PHY log: %s"%(phy_filename))

        # Determine reference time and parsing bounds
        ref_t0 = get_ref_t0(phy_filename)

        p1 = Parsenator(phy_filename,
                        [partial(parse_du_symbol_timings,mmimo_enable=mmimo_enable,parse_srs=False),
                         partial(parse_du_symbol_timings,mmimo_enable=mmimo_enable,parse_srs=True),
                         parse_ti_line],
                        [['t0_timestamp','tir','cell','sfn','slot','symbol','start_offset','end_offset','window_start_deadline'],
                         ['t0_timestamp','tir','cell','sfn','slot','symbol','start_offset','end_offset','window_start_deadline'],
                         ['t0_timestamp','tir','sfn','slot','task','subtask','end_deadline']],
                        start_tir,end_tir,num_processes=num_proc,ref_t0=ref_t0)
        
        parsed_result1 = p1.parse()
        df_du_rx_timing = parsed_result1[0]
        df_du_rx_srs_timing = parsed_result1[1]
        df_ti = parsed_result1[2]

        if ref_t0 is None and len(df_du_rx_timing)>0:
            ref_t0 = df_du_rx_timing.t0_timestamp[0]
            print("Using DU RX first t0=%i as ref_t0"%ref_t0)

        print("Processing RU log: %s"%(ru_filename))

        p2 = Parsenator(ru_filename,
                        [partial(parse_ru_tx_times,mmimo_enable=mmimo_enable),
                         partial(parse_ru_tx_sum_times,mmimo_enable=mmimo_enable),
                         partial(parse_ru_symbol_timings,mmimo_enable=mmimo_enable)],
                        [['t0_timestamp','tir','sfn','slot','symbol','type','cell','channel','window_start_deadline','tx_deadline','enqueue_deadline'],
                         ['t0_timestamp','tir','sfn','slot','type','cell','channel','window_start_deadline','enqueue_deadline','window_start_sym0_deadline','window_start_sym13_deadline'],
                         ['t0_timestamp','tir','sfn','slot','task','cell','sfn','slot','symbol','start_offset','end_offset','window_start_deadline']],
                        start_tir,end_tir,num_processes=num_proc,ref_t0=ref_t0)

        parsed_result2 = p2.parse()
        df_tx_timing = parsed_result2[0]
        df_tx_timing_sum = parsed_result2[1]
        df_ru_rx_timing = parsed_result2[2]

        if(slot_list):
            # Save on memory
            df_du_rx_timing = df_du_rx_timing[df_du_rx_timing.slot.isin(slot_list)]
            df_du_rx_srs_timing = df_du_rx_srs_timing[df_du_rx_srs_timing.slot.isin(slot_list)]
            df_ti = df_ti[df_ti.slot.isin(slot_list)]
            df_tx_timing = df_tx_timing[df_tx_timing.slot.isin(slot_list)]
            df_tx_timing_sum = df_tx_timing[df_tx_timing.slot.isin(slot_list)]
            df_ru_rx_timing = df_ru_rx_timing[df_ru_rx_timing.slot.isin(slot_list)]

        #Normalize cell in df_du_rx_timing, df_ru_rx_timing, and df_tx_timing
        if(len(df_du_rx_timing) > 0):
            df_du_rx_timing['cell'] = df_du_rx_timing['cell'] - df_du_rx_timing['cell'].min()
        if(len(df_du_rx_srs_timing) > 0):
            df_du_rx_srs_timing['cell'] = df_du_rx_srs_timing['cell'] - df_du_rx_srs_timing['cell'].min()
        if(len(df_ru_rx_timing) > 0):
            df_ru_rx_timing['cell'] = df_ru_rx_timing['cell'] - df_ru_rx_timing['cell'].min()

        df_du_rx_timing_list.append(df_du_rx_timing)
        df_du_rx_srs_timing_list.append(df_du_rx_srs_timing)
        df_ti_list.append(df_ti)
        df_tx_timing_list.append(df_tx_timing)
        df_tx_timing_sum_list.append(df_tx_timing_sum)
        df_ru_rx_timing_list.append(df_ru_rx_timing)

    return df_du_rx_timing_list,df_du_rx_srs_timing_list,df_ti_list,df_tx_timing_list,df_tx_timing_sum_list,df_ru_rx_timing_list


#Old parsers

def cicd_metrics_parsing(phy_filenames,testmac_filenames,ru_filenames,start_tir,end_tir,slot_list=None,num_proc=8):
    print("Running %s - %s"%(os.path.split(__file__)[1],inspect.stack()[0][3]))

    from aerial_postproc.logparse import get_ref_t0, parse_ti_line, parse_l2_times, parse_gpu_setup_and_run, parse_testmac_times

    ### PARSE LOGS
    df_ti_list = []
    df_l2_list = []
    df_gpu_list = []
    df_testmac_list = []
    for phy_filename,testmac_filename,ru_filename in zip(phy_filenames,testmac_filenames,ru_filenames):

        print("Processing PHY log: %s"%(phy_filename))

        # Parse phy log
        ref_t0 = get_ref_t0(phy_filename)
        p1 = Parsenator(phy_filename,
                        [parse_ti_line,parse_l2_times,partial(parse_gpu_setup_and_run,condensed_format=True)],
                        [['t0_timestamp','slot','task','subtask','start_deadline','end_deadline'], #Only necessary TI fields
                         ['t0_timestamp','slot','start_deadline','end_deadline','duration'], #Only necessary L2A fields
                         ['t0_timestamp','slot','task','gpu_run_duration','pdsch_h2d_copy_duration','gpu_total_duration'] #Only Necessary GPU fields
                        ],
                        start_tir,end_tir,num_processes=num_proc,ref_t0=ref_t0)
        parsed_result = p1.parse()
        df_ti = parsed_result[0]
        df_l2 = parsed_result[1]
        df_gpu = parsed_result[2]

        print("Processing TESTMAC log: %s"%(testmac_filename))

        # Parse testmac log
        p2 = Parsenator(testmac_filename,
                        [parse_testmac_times],
                        [['t0_timestamp','slot','fapi2_start_timestamp','fapi2_stop_timestamp']
                        ], #Only necessary TI fields???,
                        start_tir,end_tir,num_processes=num_proc,ref_t0=ref_t0)
        parsed_result = p2.parse()
        df_testmac = parsed_result[0]


        if(slot_list):
            # Save on memory
            df_ti = df_ti[df_ti.slot.isin(slot_list)]
            df_gpu = df_gpu[df_gpu.slot.isin(slot_list)]
            df_l2 = df_l2[df_l2.slot.isin(slot_list)]

        df_ti_list.append(df_ti)
        df_l2_list.append(df_l2)
        df_gpu_list.append(df_gpu)
        df_testmac_list.append(df_testmac)

    return df_ti_list,df_l2_list,df_gpu_list,df_testmac_list
    

def compare_logs_parsing(phy_filenames,start_tir,end_tir,slot_list=None,num_proc=8):
    print("Running %s - %s"%(os.path.split(__file__)[1],inspect.stack()[0][3]))

    from aerial_postproc.logparse import get_ref_t0, parse_ti_line, parse_l2_times, parse_gpu_setup_and_run, parse_new_compression_message, parse_phylog_tick_times

    ### PARSE LOGS
    df_ti_list = []
    df_l2_list = []
    df_gpu_list = []
    df_compression_list = []
    for phy_filename in phy_filenames:

        print("Processing PHY log: %s"%(phy_filename))

        # Determine reference time and parsing bounds
        ref_t0 = get_ref_t0(phy_filename)
        p1 = Parsenator(phy_filename,
                        [parse_ti_line,parse_l2_times,partial(parse_gpu_setup_and_run,condensed_format=True), parse_new_compression_message],
                        [['t0_timestamp','sfn','slot','task','subtask','sequence','start_deadline','end_deadline','duration'], #Only necessary TI fields
                         ['t0_timestamp','slot','start_deadline','end_deadline','duration'], #Only necessary L2A fields
                         ['t0_timestamp','slot','task','cpu_setup_duration','cpu_run_duration','gpu_setup1_duration','gpu_setup2_duration',
                          'gpu_run_duration','gpu_run_eh_duration','gpu_run_noneh_duration','gpu_run_copy_duration','gpu_run_gap_duration','pdsch_h2d_copy_duration','gpu_total_duration'], #Only necessary GPU fields
                         ['t0_timestamp','slot','task','cell','event','duration'], #Only necessary compression fields
                        ],
                        start_tir,end_tir,num_processes=num_proc,ref_t0=ref_t0)
    
        parsed_result = p1.parse()
        df_ti = parsed_result[0]
        df_l2 = parsed_result[1]
        df_gpu = parsed_result[2]
        df_compression = parsed_result[3]

        if(slot_list):
            # Save on memory
            df_ti = df_ti[df_ti.slot.isin(slot_list)]
            df_gpu = df_gpu[df_gpu.slot.isin(slot_list)]
            df_l2 = df_l2[df_l2.slot.isin(slot_list)]

        df_ti_list.append(df_ti)
        df_l2_list.append(df_l2)
        df_gpu_list.append(df_gpu)
        df_compression_list.append(df_compression)

    #Read in ticks if available
    df_tick_list = []
    for phy_filename in phy_filenames:
        df_tick = parse_phylog_tick_times(phy_filename, max_lines=10000)
        if(len(df_tick) > 0):
            print("Tick times found, reading in tick times...")
            df_tick = parse_phylog_tick_times(phy_filename, max_duration=end_tir)
            df_tick_list.append(df_tick)
        else:
            print("No tick times found.")
            df_tick_list.append(pd.DataFrame())

    return df_ti_list,df_l2_list,df_gpu_list,df_compression_list,df_tick_list
