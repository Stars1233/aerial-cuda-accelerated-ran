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
import os
import sys
import yaml

def get_prb_size(ud_iq_width, ud_comp_method):
    ud_comp_param_size = (0, 1, 1, 1, 0, 2, 2)
    if (ud_iq_width > 16) or (ud_iq_width == 0):
        raise KeyError("Invalid ud_iq_width value: ", ud_iq_width)
    if ud_comp_method >= len(ud_comp_param_size):
        raise KeyError("Invalid ud_comp_method value: ", ud_comp_method)

    return (ud_iq_width * 2 * 12 // 8) + ud_comp_param_size[ud_comp_method]


def create_fhgen_config_template(args, config):
    prb_size = get_prb_size(args.ud_iq_width, args.ud_comp_meth)

    config = {
        'dpdk_thread' : args.dpdk_thread,
        'dpdk_verbose_logs' : args.dpdk_verbose_logs,
        'accu_tx_sched_res_ns' : args.accu_tx_sched_res_ns,
        'startup_delay_sec' : args.startup_delay_sec,
        'worker_thread_sched_fifo_prio' : args.worker_thread_sched_fifo_prio,
        'send_utc_anchor' : 1,
        'validate_iq_data_buffer_size' : 1,
        'random_seed' : 0,
        'shuffle_cplane_tx' : 0,
        'shuffle_uplane_tx' : 0,
        'max_tx_timestamp_diff_ns' : 0,
        'nics' : [{
            'nic' : args.nic,
            'mtu' : args.mtu,
            'cpu_mbufs' : args.cpu_mbufs,
            'uplane_tx_handles' : args.uplane_tx_handles,
            'txq_count' : args.cells * 2,
            'txq_size' : args.txq_size,
        }],
        'iq_data_buffers' : [{
            'id' : 0,
            'buffer_size' : prb_size * 273 * 14,
            'cuda_device_id' : args.gpu,
        }],
        'cpus' : args.cpus,
    }

    config['cells'] = []
    for cell_id in range(args.cells):
        cell_config = {
            'cell_id': cell_id,
            'src_mac_addr' : '00:00:00:00:00:00',
            'dst_mac_addr': '20:04:9B:9E:27:' + '{:02X}'.format(cell_id),
            'txq_count_uplane': args.txq_count_uplane,
            'ud_iq_width': args.ud_iq_width,
            'ud_comp_meth': args.ud_comp_meth,
            'nic': args.nic,
            'slot_duration_ns': args.slot_duration_ns,
            'slot_count': len(args.tdd),
            'tcp_adv_dl_ns': args.tcp_adv_dl_ns,
            't1a_max_cp_ul_ns': args.t1a_max_cp_ul_ns,
            'tx_time_advance_ns': args.tx_time_advance_ns,
            'window_end_ns': args.window_end_ns,
        }
        config['cells'].append(cell_config)

    config['flows'] = []
    slot_count = len(args.tdd)
    for slot_id in range(slot_count):
        for symbol_id in range(14):
            for cell_id in range(args.cells):
                for flow_id in range(args.flows):
                    if args.tdd[slot_id] == 'U':
                        flow_config = {
                            'eAxC': (flow_id * 14 * slot_count) + (slot_id * 14) + symbol_id,
                            'vlan' : 0,
                            'pcp': args.pcp,
                            'cell_id': cell_id,
                            'cplane_tx' : [{
                                'slot_id': slot_id,
                                'symbol_id': symbol_id,
                                'section_count' : args.section_count,
                                'data_direction': 'UL',
                            }],
                        }
                    else:
                        flow_config = {
                            'eAxC': (flow_id * 14 * slot_count) + (slot_id * 14) + symbol_id,
                            'vlan' : 0,
                            'pcp': args.pcp,
                            'cell_id': cell_id,
                            'cplane_tx' : [{
                                'slot_id': slot_id,
                                'symbol_id': symbol_id,
                                'section_count' : args.section_count,
                                'data_direction': 'DL',
                            }],
                            'uplane_tx' : [{
                                'slot_id': slot_id,
                                'symbol_id': symbol_id,
                                'start_prb': 0,
                                'num_prb' : args.prbs,
                                'iq_data_buffer': 0,
                            }]
                        }

                    config['flows'].append(flow_config)


    with open(args.output, 'w') as f:
        data = yaml.dump(config, f, sort_keys=False, default_flow_style=False, explicit_start=True, explicit_end=True)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='create_fhgen_config_template.py', description="Create FH generator config template for 5G test cases")

    # Required params
    parser.add_argument("--output", help="Output FH generator config filepath", type=str, required=True)
    parser.add_argument("--nic", help="NIC port name", type=str, default="0000:b5:00.1", required=True)
    parser.add_argument("--cells", help="Cell count", type=int, required=True)
    parser.add_argument("--tdd", help="TDD pattern", type=str, default="DDDSUUDDDD", required=True)
    parser.add_argument("--flows", help="Number of eCPRI flows", type=int, required=True)
    parser.add_argument("--cpus", action="extend", nargs="+", type=int, required=True)

    # Global FH generator and DPDK params
    parser.add_argument("--dpdk_thread", help="CPU core to use for DPDK main lcore", type=int, default=0)
    parser.add_argument("--dpdk_verbose_logs", help="Enable max log level in DPDK", type=int, default=0)
    parser.add_argument("--accu_tx_sched_res_ns", help="Accurate TX cheduling clock resolution", type=int, default=500)
    parser.add_argument("--startup_delay_sec", help="Start generating traffic after the specified delay from launch", type=int, default=2)
    parser.add_argument("--worker_thread_sched_fifo_prio", help="SCHED_FIFO priority of each traffic generator thread", type=int, default=95)
    parser.add_argument("--pcp", help="VLAN Priority Code Point (PCP)", type=int, default=0)

    # Ethdev params
    parser.add_argument("--mtu", help="MTU size", type=int, default=1514)
    parser.add_argument("--cpu_mbufs", help="CPU mbuf mempool size", type=int, default=65536)
    parser.add_argument("--uplane_tx_handles", help="U-plane TX handles count", type=int, default=16)
    parser.add_argument("--txq_size", help="TXQ size", type=int, default=8192)

    # IQ data buffer params
    parser.add_argument("--gpu", help="Use GPU memory for IQ data buffers", type=int, default=0)

    # Cell params
    parser.add_argument("--txq_count_uplane", help="Number TXQs for U-plane messages", type=int, default=1)
    parser.add_argument("--ud_iq_width", help="User data compression IQ sample size", type=int, default=16)
    parser.add_argument("--ud_comp_meth", help="User data compression method", type=int, default=0)
    parser.add_argument("--slot_duration_ns", help="Slot duration in nanoseconds", type=int, default=500000)
    parser.add_argument("--tcp_adv_dl_ns", help="Time offset between C-plane and corresponding U-plane DL", type=int, default=125000)
    parser.add_argument("--t1a_max_cp_ul_ns", help="Time offset between C-plane and corresponding U-plane UL", type=int, default=336000)
    parser.add_argument("--tx_time_advance_ns", help="How much time before each C-plane TX time the packets for are enqueued onto the NIC", type=int, default=500000)
    parser.add_argument("--window_end_ns", help="(T1a_max_up - T1a_max_up) or (T1a_max_cp_ul - T1a_min_cp_ul)", type=int, default=51000)

    parser.add_argument("--prbs", help="PRB count for each flow in each symbol", type=int, default=273)
    parser.add_argument("--section_count", help="Section count for each C-plane message", type=int, default=1)

    config = {}
    create_fhgen_config_template(parser.parse_args(), config)
