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

#All logic for threshold generation here
def gen_threshold(results_dict,easy_threshold=False):
    MIN_INCREASE_USEC = 0.2
    PERCENT_INCREASE = 1

    thresh_dict = {}
    for key in results_dict.keys():
        thresh_dict[key] = None

    if(easy_threshold):
        key_list = ['rx_start_q99.50','rx_end_q99.50']
    else:
        key_list = ['rx_start_q90.00','rx_end_q90.00',
                    'rx_start_q99.00','rx_end_q99.00',
                    'rx_start_q99.50','rx_end_q99.50',
                    'rx_start_q99.90','rx_end_q99.90',
                    'rx_start_q99.99','rx_end_q99.99']

    for key in key_list:
        thresh_dict[key] = results_dict[key] + max(MIN_INCREASE_USEC,(PERCENT_INCREASE/100.0) * results_dict[key])

    return thresh_dict

#All logic for threshold application here
def execute_threshold(results_dict,thresh_dict):
    assert(results_dict.keys() == thresh_dict.keys())

    success = True
    for key in results_dict.keys():
        if thresh_dict[key] is not None:
            if results_dict[key] >= thresh_dict[key]:
                success = False
                print("Failed threshold for %s (value=%f, threshold=%f)"%(key,results_dict[key],thresh_dict[key]))
            
    return success

#All logic for summary print
def summary_print(results_dict):
    print("")

    print("Configuration: TXQs=%i, Packets Per Burst=%i (%i per que), ETH L1 Frame Size=%i"%(results_dict['num_ques'],
                                                                                             results_dict['packets_per_burst'],
                                                                                             results_dict['packets_per_burst']/results_dict['num_ques'],
                                                                                             results_dict['packet_size']))
    print("Found %i bursts with an interval of %fusec.  Total duration is %fsec"%(results_dict['total_bursts'],
                                                                                  results_dict['interval'],
                                                                                  results_dict['total_bursts']*results_dict['interval']/1e6))
    print("RX Start (min/q90/q99.5/max) = (%f/%f/%f/%f)"%(results_dict['rx_start_min'],
                                                             results_dict['rx_start_q90.00'],
                                                             results_dict['rx_start_q99.50'],
                                                             results_dict['rx_start_max'],))
    print("RX End (min/q90/q99.5/max) = (%f/%f/%f/%f)"%(results_dict['rx_end_min'],
                                                           results_dict['rx_end_q90.00'],
                                                           results_dict['rx_end_q99.50'],
                                                           results_dict['rx_end_max'],))
    print("Burst Throughput: %fGbps"%results_dict['burst_throughput'])
    print("Sustained Throughput: %fGbps"%results_dict['sustained_throughput'])
    print("")

def main(args):

    verification_success = True
    results_dict_list = []
    for input_csv in args.input_csvs:
        #Read in data
        data = pd.read_csv(input_csv,na_values="     None")

        #Apply time filter to start_tir
        data['start_tir'] = data['desired_tir'] + data['rx_start_deadline']*1e-6
        data['end_tir'] = data['desired_tir'] + data['rx_end_deadline']*1e-6
        data = data[(data.start_tir<args.max_duration)&(data.start_tir>=args.ignore_duration)]

        #Verify no gaps in burst_idx (sequence id that just increments)
        start_burst = data['burst_idx'].iloc[0]
        end_burst = data['burst_idx'].iloc[len(data)-1]
        no_gaps1 = (end_burst - start_burst) == len(data)-1
        if(not no_gaps1):
            print("ERROR :: Found gap in burst_idx progression!")
            verification_success = False
        else:
            print("burst_idx progression looks good.")

        #Verify delta in desired tir is consistent
        diffs = data.desired_tir.diff().iloc[1:]
        diffs_diffs = diffs.diff().iloc[1:]
        no_gaps2 = all([abs(aa)<1e-9 for aa in diffs_diffs])
        if(not no_gaps2):
            print("ERROR :: Found inconsistent delta in desired_tir (min/median/max = %e/%e/%e)"%(diffs.min(),diffs.median(),diffs.max()))
            verification_success = False
        else:
            print("desired_tir delta looks good.  (min/median/max = %e/%e/%e)"%(diffs.min(),diffs.median(),diffs.max()))

        #Calculate summary statistics
        results_dict = {
            'total_bursts': len(data),
            'interval': diffs.median()*1e6,
            'num_ques': data.num_ques.min(),
            'packets_per_burst': data.num_packets.min(),
            'packet_size': data.eth_frame_size.min(),
            'rx_start_q90.00': data.rx_start_deadline.quantile(0.9),
            'rx_start_q99.00': data.rx_start_deadline.quantile(0.99),
            'rx_start_q99.50': data.rx_start_deadline.quantile(0.995),
            'rx_start_q99.90': data.rx_start_deadline.quantile(0.999),
            'rx_start_q99.99': data.rx_start_deadline.quantile(0.9999),
            'rx_start_min': data.rx_start_deadline.min(),
            'rx_start_max': data.rx_start_deadline.max(),
            'rx_end_q90.00': data.rx_end_deadline.quantile(0.9),
            'rx_end_q99.00': data.rx_end_deadline.quantile(0.99),
            'rx_end_q99.50': data.rx_end_deadline.quantile(0.995),
            'rx_end_q99.90': data.rx_end_deadline.quantile(0.999),
            'rx_end_q99.99': data.rx_end_deadline.quantile(0.9999),
            'rx_end_min': data.rx_end_deadline.min(),
            'rx_end_max': data.rx_end_deadline.max(),
            'burst_throughput': data.estimated_burst_throughput_gbps.quantile(0.005),
            'sustained_throughput': data.estimated_sustained_throughput_gbps.quantile(0.005)
        }
        results_dict_list.append(results_dict)
        summary_print(results_dict)

    #Take average across all results
    print("Averaging all %i results..."%len(results_dict_list))
    results_dict = {}
    for key in results_dict_list[0]:
        results_dict[key] = np.mean([aa[key] for aa in results_dict_list])

    summary_print(results_dict)

    if(args.stats_csv is not None):
        stats_out = pd.DataFrame(results_dict,index=[0])
        print("Writing stats_csv %s..."%args.stats_csv)
        stats_out.to_csv(args.stats_csv,float_format="%9.9f",index=False,na_rep="     None")

    if(args.output_threshold_csv is not None):
        thresh_dict = gen_threshold(results_dict,easy_threshold=args.easy_threshold)

        df_out = pd.DataFrame(thresh_dict,index=[0])
        print("Writing gen_threshold_csv %s..."%args.output_threshold_csv)
        df_out.to_csv(args.output_threshold_csv,float_format="%9.9f",index=False,na_rep="     None")
    
    if(args.input_threshold_csv is not None):

        if(not verification_success):
            print("ERROR - problem validating input data.  Please check details above.")
            print("Returning failure")
            return 1

        print("Performing thresholding using file %s"%args.input_threshold_csv)
        threshold_data = pd.read_csv(args.input_threshold_csv,na_values="     None")
        threshold_dict = dict(threshold_data.iloc[0])
        success = True

        #Verify we have the same keys
        keys_found1 = all([aa in threshold_dict.keys() for aa in results_dict.keys()])
        keys_found2 = all([aa in results_dict.keys() for aa in threshold_dict.keys()])
        if not keys_found1 or not keys_found2:
            print("Threshold file not compatible...")
            print("result_dict.keys(): %s"%results_dict.keys())
            print("threshold_dict.keys(): %s"%threshold_dict.keys())
            success = False

        if(success):
            for key,value in threshold_dict.items():
                if not np.isnan(value):
                    if(results_dict[key] >= threshold_dict[key]):
                        print("THRESHOLD FAILURE for key %s, value=%f, threshold=%f"%(key,results_dict[key],threshold_dict[key]))
                        success = False
                    else:
                        print("THRESHOLD SUCCESS for key %s, value=%f, threshold=%f"%(key,results_dict[key],threshold_dict[key]))

        if(success):
            print("Returning success")
            return 0
        else:
            print("Returning failure")
            return 1
    else:
        print("No thresholding requested")
        return 1

if __name__ == "__main__":
    default_max_duration=999999999.0
    default_ignore_duration=0.0

    parser = argparse.ArgumentParser(
        description="""
        Script for indicating pass/fail of netperftest results based in input thresholds. 
        Can also be used to generate summary stats and threshold files.
                       
        If more than one input csv specified, all stats/thresholds are based on the average of the statistics of the input files
        (useful for generating thresholds based on several identical runs)
        """
    )
    parser.add_argument(
        "input_csvs", type=str, nargs="+", help="CSV file(s) parsed by rxlogs2csv"
    )
    parser.add_argument(
        "-m", "--max_duration", default=default_max_duration, type=float, help="Maximum run duration to process"
    )
    parser.add_argument(
        "-i", "--ignore_duration", default=default_ignore_duration, type=float, help="Number of seconds at beginning of file to ignore"
    )
    parser.add_argument(
        "-t", "--input_threshold_csv", help="If specified enables thresholding, returns 0 on success, 1 on failure"
    )
    parser.add_argument(
        "-g", "--output_threshold_csv", help="If specified script will generate a threshold file to gen_threshold_csv based on input_csvs"
    )
    parser.add_argument(
        "-e", "--easy_threshold", action="store_true", help="If set threshold file only uses single value protecting 99.5%%"
    )
    parser.add_argument(
        "-s", "--stats_csv", help="If specified script will generate a stats file to stats_csv based on input_csvs"
    )
    parser.add_argument(
        "-o", "--out_filename", type=str, help="Output html file"
    )
    args = parser.parse_args()

    main(args)
