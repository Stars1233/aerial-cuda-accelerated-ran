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
from enum import Enum
import os
import sys
import yaml


class FlowType(Enum):
    cplane = 2
    uplane = 0

class Direction(Enum):
    uplink = 0
    downlink = 1

dir_map = {
    'UL' : Direction.uplink,
    'DL' : Direction.downlink,
}

class Flow:
    @staticmethod
    def get_packets_from_prbs(mtu, ud_iq_width, ud_comp_meth, prbs):
        UD_COMP_PARAM_SIZE = [0, 1, 1, 1, 0, 2, 2]
        RE_PER_PRB = 12
        ORAN_UPLANE_HDR_SIZE = 34

        prb_size = ((ud_iq_width * 2 * RE_PER_PRB) / 8) + UD_COMP_PARAM_SIZE[ud_comp_meth]
        prbs_per_packet = (mtu - ORAN_UPLANE_HDR_SIZE) // prb_size
        return int((prbs + prbs_per_packet -1) / prbs_per_packet)

    def __init__(self, cell, flow_type, direction, ecpri_id, vlan_id, slot_id, symbol_id, prbs = 0):
        self.cell = cell
        self.ecpri_id = str(ecpri_id)
        self.vlan_id = str(vlan_id)
        self.name = str(flow_type).split('.')[1] + "_" + self.ecpri_id + "_cell_" + str(self.cell.cell_id)+ "_slot_" + str(slot_id)+ "_symbol_" + str(symbol_id)
        self.epcri = "0x00020000/0x00FF0000" if flow_type == FlowType.cplane else "0x00000000/0x00FF0000"
        self.window_offset = (self.cell.slot_duration * slot_id) + (self.cell.symbol_duration * symbol_id)
    
        if flow_type == FlowType.cplane:
            if direction == Direction.downlink:
                self.window_offset -= self.cell.tcp_adv_dl
            else:
                self.window_offset -= self.cell.t1a_max_cp_ul_ns
        
        if flow_type == FlowType.cplane:
            self.min_allowed_packets = 1
        else:
            self.min_allowed_packets = Flow.get_packets_from_prbs(self.cell.nic.mtu, self.cell.ud_iq_width, self.cell.ud_comp_meth, prbs)

        self.max_allowed_packets = self.min_allowed_packets

    def update_allowed_pkts(self, prbs):
        self.min_allowed_packets += Flow.get_packets_from_prbs(self.cell.nic.mtu, self.cell.ud_iq_width, self.cell.ud_comp_meth, prbs)
        self.max_allowed_packets = self.min_allowed_packets


    def __str__(self):
        out = "\nflow=" + self.name
        out += ",ecpri_id=" + self.ecpri_id
        out += ",ecpri=" + self.epcri
        out += ",vlan=" + self.vlan_id
        out += ",window_offset=" + str(self.window_offset)
        out += ",min_allowed_packets=" + str(self.min_allowed_packets)
        out += ",max_allowed_packets=" + str(self.max_allowed_packets)
        return out


class Cell:
    def __init__(self, nic, cell_id, dst_mac_addr, slot_duration, slot_count, window_end, tcp_adv_dl, t1a_max_cp_ul_ns, ud_iq_width, ud_comp_meth):
        self.nic = nic
        self.cell_id = cell_id
        self.dst_mac_addr = dst_mac_addr
        self.window_interval = slot_count * slot_duration
        self.window_end = window_end
        self.slot_duration = slot_duration
        self.tcp_adv_dl = tcp_adv_dl
        self.t1a_max_cp_ul_ns = t1a_max_cp_ul_ns
        self.symbol_duration = slot_duration // 14
        self.ud_iq_width = ud_iq_width
        self.ud_comp_meth = ud_comp_meth
        self.flows = []

    def __str__(self):
        out = "\n\n; Cell " + str(self.cell_id)
        out += "\ndmac=" + str(self.dst_mac_addr)
        out += "\nwindow_interval=" + str(self.window_interval)
        out += "\nwindow_end=" + str(self.window_end)

        for flow in self.flows:
            out += str(flow)

        return out

    def add_flow(self, flow):
        self.flows.append(flow)


class Nic:
    def __init__(self, mtu):
        self.mtu = mtu
        self.cells = []

    def __str__(self):
        out = ""
        for cell in self.cells:
            out += str(cell)

        return out

    def add_cell(self, cell):
        self.cells.append(cell)


class Config:
    def __init__(self, fhgen_config_filename, max_jitter_start, max_jitter_end):
        self.fhgen_config_filename = os.path.basename(fhgen_config_filename)
        self.max_jitter_start = max_jitter_start
        self.max_jitter_end = max_jitter_end
        self.window_start = 0
        self.nics = []

    def __str__(self):
        out = "; TimeCheck config for " + str(self.fhgen_config_filename)
        out += "\nmax_jitter_start=" + str(self.max_jitter_start)
        out += "\nmax_jitter_end=" + str(self.max_jitter_end)
        out += "\nwindow_start=" + str(self.window_start)

        for nic in self.nics:
            out += str(nic)

        return out

    def add_nic(self, nic):
        self.nics.append(nic)


def parse_nic_cfg(nic_cfg, config, nic_map):
    nic = Nic(nic_cfg['mtu'])
    config.add_nic(nic)
    nic_map[nic_cfg['nic']] = nic


def parse_cell_cfg(cell_cfg, cell_map, nic_map):
    nic_name = cell_cfg['nic']
    nic = nic_map[nic_name]
    cell_id = cell_cfg['cell_id']
    dst_mac_addr = cell_cfg['dst_mac_addr']
    slot_duration = cell_cfg['slot_duration_ns']
    slot_count = cell_cfg['slot_count']
    window_end = cell_cfg['window_end_ns']
    tcp_adv_dl = cell_cfg['tcp_adv_dl_ns']
    t1a_max_cp_ul_ns = cell_cfg['t1a_max_cp_ul_ns']
    ud_iq_width = cell_cfg['ud_iq_width']
    ud_comp_meth =  cell_cfg['ud_comp_meth']

    cell = Cell(nic, cell_id, dst_mac_addr, slot_duration, slot_count, window_end, tcp_adv_dl, t1a_max_cp_ul_ns, ud_iq_width, ud_comp_meth)
    nic.add_cell(cell)
    cell_map[cell_id] = cell

flow_dict = {}

def parse_flow_cfg(flow_cfg, cell_map):
    cell_id = flow_cfg['cell_id']
    cell = cell_map[cell_id]
    ecpri_id = flow_cfg['eAxC']
    vlan_id = flow_cfg['vlan']

    if 'cplane_tx' in flow_cfg:
        for cplane_tx in flow_cfg['cplane_tx']:
            slot_id = cplane_tx['slot_id']
            symbol_id = cplane_tx['symbol_id']
            direction = dir_map[cplane_tx['data_direction']]
            flow = Flow(cell, FlowType.cplane, direction, ecpri_id, vlan_id, slot_id, symbol_id)
            cell.add_flow(flow)

    if 'uplane_tx' in flow_cfg:
        for uplane_tx in flow_cfg['uplane_tx']:
            slot_id = uplane_tx['slot_id']
            symbol_id = uplane_tx['symbol_id']
            prbs = uplane_tx['num_prb']
            flow_key = f"uplane_tx,{cell_id},{ecpri_id},{vlan_id},{slot_id},{symbol_id}"
            if flow_key in flow_dict:
                flow_dict[flow_key].update_allowed_pkts(prbs)
            else:
                flow = Flow(cell, FlowType.uplane, Direction.downlink, ecpri_id, vlan_id, slot_id, symbol_id, prbs)
                cell.add_flow(flow)
                flow_dict[flow_key] = flow


def main(args):
    fh_gen_config = None

    with open(args.fhgen) as f:
        fh_gen_config = yaml.load(f, Loader=yaml.FullLoader)
        config = Config(args.fhgen, args.max_jitter_start, args.max_jitter_end)

        nic_map = {}
        for nic_cfg in fh_gen_config['nics']:
            parse_nic_cfg(nic_cfg, config, nic_map)

        cell_map = {}
        for cell_cfg in fh_gen_config['cells']:
            parse_cell_cfg(cell_cfg, cell_map, nic_map)
    
        for flow_cfg in fh_gen_config['flows']:
            parse_flow_cfg(flow_cfg, cell_map)


    with open(args.timecheck, 'w') as f:
        f.write(str(config))

    if args.verbose:
        print(config)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='fhgen_to_timecheck.py', description="Convert FH traffic generator config to TimeCheck config")
    parser.add_argument("fhgen", help="input FH generator config file", type=str)
    parser.add_argument("timecheck", help="output TimeCheck config file", type=str)
    parser.add_argument("--max_jitter_start", help="max offset from the window start to the first packet (ns)", type=int, default=0)
    parser.add_argument("--max_jitter_end", help="max offset from the last packet to the window end (ns)", type=int, default=0)
    parser.add_argument("-v", "--verbose", help="print TimeCheck config to stdout", action='store_true')
    main(parser.parse_args())
