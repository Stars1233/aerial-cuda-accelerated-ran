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

def read_rx_data(rx_log,pkt_len=1518):

    df_final = pd.read_csv(rx_log)

    #Pair down to only fields we need
    df_final = df_final.rename(columns={'burst_id':'burst_idx'})
    keep_keys = ['desired_tx_timestamp','actual_rx_timestamp','pkt_within_burst','burst_idx']
    df_final = df_final[keep_keys]

    #Add time fields, pair down fields
    ref_time = df_final['desired_tx_timestamp'].min()
    df_final['desired_tir'] = (df_final['desired_tx_timestamp'] - ref_time) / 1e9
    df_final['rx_deadline'] = (df_final['actual_rx_timestamp'] - df_final['desired_tx_timestamp']) / 1e3
    keep_keys = ['desired_tir','rx_deadline','pkt_within_burst','burst_idx']
    df_final = df_final[keep_keys]

    #Reduce information down to burst-level
    gb = df_final.groupby('desired_tir')

    df_final = df_final.assign(rx_start_deadline=gb.rx_deadline.transform("min"))
    df_final = df_final.assign(rx_end_deadline=gb.rx_deadline.transform("max"))
    df_final = df_final.assign(min_packet_idx=gb.pkt_within_burst.transform("min"))
    df_final = df_final.assign(max_packet_idx=gb.pkt_within_burst.transform("max"))
    df_final = df_final.assign(num_ques=gb.pkt_within_burst.transform('count') // 2)#Note: Assume we have first/last packet for each TX que
    df_final = df_final.assign(num_packets=df_final.num_ques*(df_final.max_packet_idx+1))#Note: Assumption here is all ques have same number of packets (and max_packet_idx = num_packets-1)
    keep_keys = ['burst_idx','desired_tir','rx_start_deadline','rx_end_deadline','num_ques','num_packets']
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
    df_final = read_rx_data(args.rx_log,pkt_len=args.pkt_len)
    df_final.to_csv(args.output_csv,float_format="%9.9f",index=False,na_rep="     None")

if __name__ == "__main__":

    default_pkt_len = 1518

    parser = argparse.ArgumentParser(
        description="Converts a netperftest rx log to a summary format useful for analyzing/thresholding"
    )
    parser.add_argument(
        "rx_log", help="List of RX logs (one for each RX que)"
    )
    parser.add_argument(
        "-o", "--output_csv", type=str, default="result.csv", help="Output CSV containing summary data"
    )
    parser.add_argument(
        "-p", "--pkt_len", default=default_pkt_len, help="pkt_len used in netperftest (default=%i)"%default_pkt_len
    )
    args = parser.parse_args()

    main(args)
