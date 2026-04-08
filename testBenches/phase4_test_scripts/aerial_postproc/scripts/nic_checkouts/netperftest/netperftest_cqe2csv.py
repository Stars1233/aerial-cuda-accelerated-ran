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
import re
import copy

import pandas as pd

#Messages we parse
#
#1.) rte_eth_burst metadata
# Example: 1709679768357545558: tx(p=0, q=0, 192/192 pkts in 2880
parser1 = re.compile(
    r"(\d+): tx\(p=(\d+), q=(\d+), (\d+)\/(\d+) pkts in (\d+)"
)
#
#2.) WAIT metadata
# 0037: WAIT (1709679768358035714, 68, 487276)
parser2 = re.compile(
    r"\s+([a-fA-F0-9]+): WAIT \((\d+), (\d+), (\d+)"
)
#3.) EMPW metadata
# 0038: EMPW (1709679768358035714, 4050, 487276)
parser3 = re.compile(
    r"\s+([a-fA-F0-9]+): EMPW \((\d+), (\d+), (\d+)"
)

def get_cqe_dfs(input_file,only_last_empw):
    global parser1, parser2, parser3

    with open(input_file) as fid:

        #Store all data found for individual components
        rte_list_dict = []
        wait_list_dict = []
        empw_list_dict = []

        #Combined dataset
        summary_list_dict = []

        #Stores the last entry found as we sequential process data
        rte_metdata = None
        wait_metadata = None
        empw_metadata = None

        current_line = "dummystring"
        while(current_line != ""):
            current_line = fid.readline()

            found = parser1.match(current_line)
            if found:
                rte_metadata = {'tx_burst_start_time': int(found[1]),
                                'tx_burst_end_time': int(found[1]) + int(found[6]),
                                'tx_burst_duration': int(found[6]),
                                'vport': int(found[2]),
                                'tx_que': int(found[3]),
                                'num_packets': int(found[4]),
                                'num_packets2': int(found[5])
                               }
                # print("Parsed RTE: %s"%rte_metadata)
                rte_list_dict.append(rte_metadata)

                continue

            found = parser2.match(current_line)
            if found:
                wait_metadata = {'wqe_index': found[1],
                                 'desired_time': int(found[2]),
                                 'completion_time': int(found[3]) + int(found[2]),
                                 'tx_burst_end_time': int(found[2]) - int(found[4])
                                 }
                # print("Parsed WAIT: %s"%wait_metadata)
                wait_list_dict.append(wait_metadata)

                continue

            found = parser3.match(current_line)
            if found:
                empw_metadata = {'wqe_index': found[1],
                                 'desired_time': int(found[2]),
                                 'completion_time': int(found[3]) + int(found[2]),
                                 'tx_burst_end_time': int(found[2]) - int(found[4])
                                 }
                # print("Parsed EMPW: %s"%empw_metadata)
                empw_list_dict.append(empw_metadata)

                #Combine all data into single entry
                if(rte_metadata is not None and wait_metadata is not None):
                    # summary_metadata = copy.copy(empw_metadata)
                    # summary_metadata.rename(columns={'rte_'+key:key for key in empw_metadata})
                    summary_metadata = {'empw_'+key:data for (key,data) in empw_metadata.items()}
                    for key in rte_metadata.keys():
                        summary_metadata['rte_'+key] = rte_metadata[key]
                    for key in wait_metadata.keys():
                        summary_metadata['wait_'+key] = wait_metadata[key]
                    summary_list_dict.append(summary_metadata)

                continue

        #Create dataframes
        df_rte = pd.DataFrame(rte_list_dict)
        df_wait = pd.DataFrame(wait_list_dict)
        df_empw = pd.DataFrame(empw_list_dict)
        df_summary = pd.DataFrame(summary_list_dict)

    if(only_last_empw):
        gb = df_summary.groupby(['empw_desired_time','rte_vport','rte_tx_que'])
        df_summary = df_summary.assign(empw_max_completion_time=gb.empw_completion_time.transform(max))
        df_summary = df_summary[df_summary.empw_completion_time==df_summary.empw_max_completion_time].reset_index()
        df_summary = df_summary.drop(['index','empw_max_completion_time'], axis=1)

    return df_rte,df_wait,df_empw,df_summary

#Note: Old parsing function where rx log information was split into multiple files and
# cqe tracing could merge with RX data
#Used to be used in netperftest_rxlogs2csv.py - stashing it here in case we need this functionality back
def combine_rx_data(rx_logs,cqe_log=None,pkt_len=1518):
    #Read in cqe log if available
    df_cqe = None
    if(cqe_log is not None):
        _,_,_,df_cqe = get_cqe_dfs(cqe_log,only_last_empw=True)

    CQE_ENABLED = df_cqe is not None

    #Read in rx logs
    df_rx_list = []
    for rx_log in rx_logs:
        df_rx_list.append(pd.read_csv(rx_log))

    #Produce timeline plot
    merged_list = []
    ref_time = None
    for df_rx in df_rx_list:

        if(ref_time is None):
            ref_time = df_rx['desired_tx_timestamp'].min()
        else:
            ref_time = min(ref_time,df_rx['desired_tx_timestamp'].min())

        #Merge with cqe data if available
        if(CQE_ENABLED):
            ref_time = min(ref_time,df_cqe['empw_desired_time'].min())
            df_merge = pd.merge(df_cqe.add_suffix("_cqe"),df_rx,left_on=["empw_desired_time_cqe","rte_tx_que_cqe"],right_on=["desired_tx_timestamp","txq"],validate="one_to_one")
            #Run several verifications
            print("Comparing RX/WAIT desired times: %s"%(df_merge.desired_tx_timestamp==df_merge.wait_desired_time_cqe).all())
            print("Comparing RX/EMPW desired times: %s"%(df_merge.desired_tx_timestamp==df_merge.empw_desired_time_cqe).all())
            print("Comparing RTE/WAIT tx burst end times: %s"%(df_merge.rte_tx_burst_end_time_cqe==df_merge.wait_tx_burst_end_time_cqe).all())
            print("Comparing RTE/EMPW tx burst end times: %s"%(df_merge.rte_tx_burst_end_time_cqe==df_merge.empw_tx_burst_end_time_cqe).all())
            print("Comparing RX/RTE txq: %s"%(df_merge.txq==df_merge.rte_tx_que_cqe).all())

        else:
            df_merge = df_rx.copy()
        
        keep_fields = []

        #Add CQE fields if available
        if(CQE_ENABLED):
            df_merge['enqueue_start_deadline'] = (df_merge['rte_tx_burst_start_time_cqe'] - df_merge['desired_tx_timestamp']) / 1e3
            df_merge['enqueue_complete_deadline'] = (df_merge['rte_tx_burst_end_time_cqe'] - df_merge['desired_tx_timestamp']) / 1e3
            df_merge['wait_complete_deadline'] = (df_merge['wait_completion_time_cqe'] - df_merge['desired_tx_timestamp']) / 1e3
            df_merge['empw_complete_deadline'] = (df_merge['empw_completion_time_cqe'] - df_merge['desired_tx_timestamp']) / 1e3
            keep_fields.extend(['enqueue_start_deadline','enqueue_complete_deadline','wait_complete_deadline','empw_complete_deadline'])

        #Add RX fields
        df_merge['tx_que'] = df_merge['txq']
        df_merge['rx_que'] = df_merge['rxq']
        df_merge['burst_idx'] = df_merge['burst_id']
        df_merge['packet_idx'] = df_merge['pkt_within_burst']
        keep_fields.extend(['desired_tx_timestamp','actual_rx_timestamp','tx_que','rx_que','burst_idx','packet_idx'])

        #Filter to only output fields and add to output list
        df_merge = df_merge[keep_fields]
        merged_list.append(df_merge)

    #Combine all data together - dataframe now has information for first/last packet of each tx que for each burst
    df_final = pd.concat(merged_list)

    #Add time fields
    df_final['desired_tir'] = (df_final['desired_tx_timestamp'] - ref_time) / 1e9
    df_final['rx_deadline'] = (df_final['actual_rx_timestamp'] - df_final['desired_tx_timestamp']) / 1e3

    #Reduce information down to burst-level
    gb = df_final.groupby('desired_tx_timestamp')
    keep_keys = []
    if(CQE_ENABLED):
        df_final = df_final.assign(enqueue_start_deadline=gb.enqueue_start_deadline.transform(min))
        df_final = df_final.assign(enqueue_end_deadline=gb.enqueue_complete_deadline.transform(max))
        #Note: Original had these separated out, but now treating entire TX section as one section (tx_start_deadline to tx_end_deadline)
        # df_final = df_final.assign(wait_start_deadline=gb.wait_complete_deadline.transform(min))
        # df_final = df_final.assign(wait_end_deadline=gb.wait_complete_deadline.transform(max))
        # df_final = df_final.assign(empw_start_deadline=gb.empw_complete_deadline.transform(min))
        # df_final = df_final.assign(empw_end_deadline=gb.empw_complete_deadline.transform(max))
        df_final = df_final.assign(tx_start_deadline=gb.wait_complete_deadline.transform(min))
        df_final = df_final.assign(tx_end_deadline=gb.empw_complete_deadline.transform(max))
        keep_keys.extend(['enqueue_start_deadline','enqueue_end_deadline','tx_start_deadline','tx_end_deadline'])

    df_final = df_final.assign(rx_start_deadline=gb.rx_deadline.transform(min))
    df_final = df_final.assign(rx_end_deadline=gb.rx_deadline.transform(max))
    df_final = df_final.assign(min_packet_idx=gb.packet_idx.transform(min))
    df_final = df_final.assign(max_packet_idx=gb.packet_idx.transform(max))
    df_final = df_final.assign(num_ques=gb.packet_idx.transform('count') // 2)#Note: Assume we have first/last packet for each TX que
    df_final = df_final.assign(num_packets=df_final.num_ques*(df_final.max_packet_idx+1))#Note: Assumption here is all ques have same number of packets (and max_packet_idx = num_packets-1)
    keep_keys.extend(['burst_idx','desired_tir','rx_start_deadline','rx_end_deadline','num_ques','num_packets'])
    df_final = df_final[keep_keys]
    df_final = df_final.drop_duplicates()

    #Estimate final throughput
    ETH_L2_FRAME_SIZE = pkt_len + 4 #netperftest definition of pkt_len is payload + all of L2 header - FCS
    ETH_L1_FRAME_SIZE = ETH_L2_FRAME_SIZE + 7 + 1 + 12 #Wire will include preamble (7 bytes), start frame delimiter (1 byte), and inter packet gap (12 bytes)
    df_final = df_final.assign(eth_frame_size=ETH_L1_FRAME_SIZE)
    df_final = df_final.assign(rx_duration=df_final.rx_end_deadline - df_final.rx_start_deadline)

    #Throughput estimate that penalizes first packet latencies
    df_final = df_final.assign(estimated_burst_throughput_gbps=(df_final.num_packets*ETH_L1_FRAME_SIZE*8/1e9) / (df_final.rx_end_deadline*1e-6))

    #Throughput estimate that ignores first packet latencies (useful for Aerial use case as RX window larger than TX period)
    df_final = df_final.assign(estimated_sustained_throughput_gbps=((df_final.num_packets-1)*ETH_L1_FRAME_SIZE*8/1e9) / (df_final.rx_duration*1e-6))

    return df_final

def main(args):
    df_rte,df_wait,df_empw,df_summary = get_cqe_dfs(args.input_file,only_last_empw=args.only_last_empw)
    
    df_summary.to_csv(args.output_file,float_format="%9.3f",index=False,na_rep="     None")
    if(args.rte_file):
        df_rte.to_csv(args.output_file,float_format="%9.3f",index=False,na_rep="     None")
    if(args.wait_file):
        df_wait.to_csv(args.output_file,float_format="%9.3f",index=False,na_rep="     None")
    if(args.empw_file):
        df_empw.to_csv(args.output_file,float_format="%9.3f",index=False,na_rep="     None")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Converts a netperftest cqe txt log to a csv file"
    )
    parser.add_argument(
        "input_file", help="Input txt file containing cqe logging from netperftest"
    )
    parser.add_argument(
        "output_file", help="Output csv file"
    )
    parser.add_argument(
        "-l", "--only_last_empw", action="store_true", help="Output file only contains last EMPW entry"
    )
    parser.add_argument(
        "-r", "--rte_file", help="Write out file with only RTE data"
    )
    parser.add_argument(
        "-w", "--wait_file", help="Write out file with only WAIT data"
    )
    parser.add_argument(
        "-e", "--empw_file", help="Write out file with only EMPW data"
    )
    args = parser.parse_args()

    main(args)
