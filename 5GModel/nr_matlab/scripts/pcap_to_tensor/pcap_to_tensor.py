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
from collections import defaultdict
import datetime
import os
import pyshark
import subprocess
import yaml

PRBS_PER_SYMBOL = 273
SYMBOLS_PER_SLOT = 14
NEW_SFN_THRES = 1.0

DATA_DIRECTIONS = {
    "UL" : 0,
    "DL" : 1
}

DATA_DIRECTIONS_REV = {
    0 : "UL",
    1 : "DL"
}

# Tree (recursive dictionary) to group and store all IQ samples
def tree(): return defaultdict(tree)


def parse_yaml(filepath):
    with open(filepath, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def parse_fapi(filepath, filterstring):
    return pyshark.FileCapture(filepath, include_raw=False, use_json=True, display_filter=filterstring,custom_parameters=[ '-X','lua_script:scffapi_nv_update.lua'])

def extract_from_fapi(fapifile, filterstring):
       print("Reading NVIPC PCAP")
       # Example filterstring, to be improved:
       # filterstring='sfn_slot.SFN == 326 && sfn_slot.Slot == 4 && ((FAPI.message_type_id == 0x0081 && PDU.PDUType == 2) || (FAPI.message_type_id == 0x0087)) && frame.number < 180000'
       try:
           fapi = parse_fapi(fapifile, '')
           fapi.load_packets()
           filter_from_fapi=''
           expected_outputs=0
           for packet in fapi:
               # There are some nasty json layering that happens between lua->python struct passing.
               #packet.frame_info.time_epoch.pretty_print()
               #print('msg_type_id: '+packet['nvidia_fapi'].text[0].message_type_id)
               #packet['nvidia_fapi'].text[1].text.text.pretty_print()
               startTime=float(packet.frame_info.time_epoch)
               endTime=float(packet.frame_info.time_epoch)+.003 #3ms should be enough from start end in the capturing server
               fapiSfn=int(packet['nvidia_fapi'].text[1].SFN)
               fapiSlot=int(packet['nvidia_fapi'].text[1].Slot)
               oranFrame=fapiSfn%256
               oranSubframe=int(fapiSlot/2) # TBD for mu=1
               oranSlot=fapiSlot%2
               tmpFilter='(frame.time_epoch >= '   +str(startTime)+ \
                      ' && frame.time_epoch <= '   +str(endTime)+ \
                      ' && oran_fh_cus.frameId == '+str(oranFrame)+ \
                      ' && oran_fh_cus.subframe_id == '+str(oranSubframe)+ \
                      ' && oran_fh_cus.slotId ==   '+str(oranSlot)+')'
               print("Extracted filter from packet: ",tmpFilter)
               if filter_from_fapi != '':
                  filter_from_fapi += ' || '
               filter_from_fapi += tmpFilter
               expected_outputs+=1
               prefix= config['output_directory']+'/'+config['output_file_prefix']
               timestamp_short = str(startTime).split('.')[0][6:] #Use the last 4 digits of linux time for identifying the packet
               filename =  "{}fapi_{}_{}.{}_oran_{}.{}.{}.yaml".format(prefix, timestamp_short, fapiSfn, fapiSlot, oranFrame, oranSubframe, oranSlot)
               print("Writing NVIPC info to: ",filename)

               write_yaml(packet['nvidia_fapi'].text[1].text,filename)

       except Exception as e:
               print(e)

       finally:
           fapi.close()
           print("Overriding filter with filter based on nvipc, expect",expected_outputs,"output files")
           return filter_from_fapi

def parse_pcap(filepath, config,filterstring):
    if config['raw_mode']:
        return pyshark.FileCapture(filepath, include_raw=True, use_json=True, display_filter=filterstring)
    else:
        return pyshark.FileCapture(filepath)


def validate_field(reference, actual):
    if isinstance(reference, int):
        return (actual == reference) or (reference == -1)
    elif isinstance(reference, str):
        return actual.lower() == str(reference).lower()
    else:
        return False


def filter_eth(config, eth):
    (src_mac_addr, dst_mac_addr, etype) = (str(eth.src), str(eth.dst), int(eth.type, 16))
    src_mac_addr_ok = validate_field(config['src_mac_addr'], src_mac_addr)
    dst_mac_addr_ok = validate_field(config['dst_mac_addr'], dst_mac_addr)
    etype_ok = etype == 0x8100
    return src_mac_addr_ok and dst_mac_addr_ok and etype_ok


def filter_vlan(config, vlan):
    (vlan_id, etype) = (int(vlan.id), int(vlan.etype, 16))
    vlan_id_ok = validate_field(config['vlan_id'], vlan_id)
    etype_ok = etype == 0xaefe
    return vlan_id_ok and etype_ok


def filter_ecpri(config, ecpri):
    ecpri_msg_type = int(ecpri.type, 16)
    return ecpri_msg_type == 0


def filter_oran_fh_cus(config, oran_fh_cus):
    ru_port_id_ok = validate_field(config['ru_port_id'], int(oran_fh_cus.ru_port_id))
    direction_ok = validate_field(config['direction'], int(oran_fh_cus.data_direction))
    frame_ok = validate_field(config['frame_id'], int(oran_fh_cus.frameid))
    subframe_ok = validate_field(config['subframe_id'], int(oran_fh_cus.subframe_id))
    slot_ok = validate_field(config['slot_id'], int(oran_fh_cus.slotid))

    return ru_port_id_ok and direction_ok and frame_ok and subframe_ok and slot_ok

def parse_oran_fh_cus(config, oran_fh_cus):
    ru_port_id = int(oran_fh_cus.ru_port_id)
    direction = int(oran_fh_cus.data_direction)
    frame = int(oran_fh_cus.frameid)
    subframe = int(oran_fh_cus.subframe_id)
    slot = int(oran_fh_cus.slotid)
    symbol = int(oran_fh_cus.startsymbolid)
    start_prb = int(oran_fh_cus.startprbu)
    num_prb = int(oran_fh_cus.numprbu)

    return (ru_port_id, direction, frame, subframe, slot, symbol, start_prb, num_prb)


def get_prb_size(ud_iq_width, ud_comp_method):
    ud_comp_param_size = (0, 1, 1, 1, 0, 2, 2)

    if (ud_iq_width > 16) or (ud_iq_width == 0):
        raise KeyError("Invalid ud_iq_width value: ", ud_iq_width)
    if ud_comp_method >= len(ud_comp_param_size):
        raise KeyError("Invalid ud_comp_method value: ", ud_comp_method)

    return (ud_iq_width * 2 * 12 // 8) + ud_comp_param_size[ud_comp_method]


def validate_ud_buffer(ud_buffer, num_prbs, ud_iq_width, ud_comp_method):
    ud_buffer_len = len(ud_buffer)
    prb_size = get_prb_size(ud_iq_width, ud_comp_method)

    if (num_prbs * prb_size) != ud_buffer_len:
        raise AttributeError("Invalid user data IQ buffer! {} vs {} (actual vs estimated)".format(ud_buffer_len, num_prbs * prb_size))


def filter_packets(pcap, config):
    ud_buffers = tree()
    ud_buffer_offset = 12 if config['ud_comp_hdr_present'] == 0 else 14

    for packet in pcap:
        
        if filter_eth(config, packet.eth) and filter_vlan(config, packet.vlan) and filter_ecpri(config, packet.ecpri) and filter_oran_fh_cus(config, packet.oran_fh_cus):

            (ru_port_id, direction, frame, subframe, slot, symbol, start_prb, num_prbs) = parse_oran_fh_cus(config, packet.oran_fh_cus)
   
            if (num_prbs == 0):
                num_prbs = PRBS_PER_SYMBOL

            ud_buffer = str(packet.ecpri.payload).split(':')[ud_buffer_offset:]
            validate_ud_buffer(ud_buffer, num_prbs, config['ud_iq_width'], config['ud_comp_method'])
            sniff_ts = float(packet.sniff_timestamp)
            slot_timestamps = ud_buffers[direction][frame][subframe][slot].keys()

            # NOTE: Logic to distinguish repeating SFN numbers
            if len(slot_timestamps) == 0:
                ud_buffers[direction][frame][subframe][slot][sniff_ts][symbol][ru_port_id][(start_prb, num_prbs)] = ud_buffer
            else:
                last_sniff_ts = max(slot_timestamps)
                if sniff_ts > last_sniff_ts + NEW_SFN_THRES:
                    ud_buffers[direction][frame][subframe][slot][sniff_ts][symbol][ru_port_id][(start_prb, num_prbs)] = ud_buffer
                else:
                    ud_buffers[direction][frame][subframe][slot][last_sniff_ts][symbol][ru_port_id][(start_prb, num_prbs)] = ud_buffer

    return ud_buffers


def extract_value(data, byte_offset, bitmask, shift_value):
    value = data[byte_offset]
    value = (value & bitmask) >> shift_value
    return value


def filter_ecpri_raw(config, ecpri):
    ecpri_msg_type = int(ecpri.header_tree.type, 16)
    return ecpri_msg_type == 0


def filter_oran_fh_cus_raw(config, ecpri_payload):
    ru_port_id_ok = validate_field(config['ru_port_id'], extract_value(ecpri_payload, 1, 0x0f, 0))
    direction_ok = validate_field(config['direction'], extract_value(ecpri_payload, 4, 0x80, 7))
    frame_ok = validate_field(config['frame_id'], extract_value(ecpri_payload, 5, 0xff, 0))
    subframe_ok = validate_field(config['subframe_id'], extract_value(ecpri_payload, 6, 0xf0, 4))
    slot_ok = validate_field(config['slot_id'], (extract_value(ecpri_payload, 6, 0x0f, 0) << 2) | extract_value(ecpri_payload, 7, 0xc0, 6))
    
    return ru_port_id_ok and direction_ok and frame_ok and subframe_ok and slot_ok


def parse_oran_fh_cus_raw(config, ecpri_payload):
    ru_port_id = extract_value(ecpri_payload, 1, 0x0f, 0)
    direction = extract_value(ecpri_payload, 4, 0x80, 7)
    frame = extract_value(ecpri_payload, 5, 0xff, 0)
    subframe = extract_value(ecpri_payload, 6, 0xf0, 4)
    slot = (extract_value(ecpri_payload, 6, 0x0f, 0) << 2) | extract_value(ecpri_payload, 7, 0xc0, 6)
    symbol = extract_value(ecpri_payload, 7, 0x3f, 0)
    start_prb = (extract_value(ecpri_payload, 9, 0x03, 0) << 8) | extract_value(ecpri_payload, 10, 0xff, 0)
    num_prb = extract_value(ecpri_payload, 11, 0xff, 0)

    return (ru_port_id, direction, frame, subframe, slot, symbol, start_prb, num_prb)


def filter_packets_raw(pcap, config):
    ud_buffers = tree()
    ud_buffer_offset = 12 if config['ud_comp_hdr_present'] == 0 else 14

    for packet in pcap:

        raw_frame = bytes.fromhex(packet.frame_raw.value)
        ecpri_payload = raw_frame[22:]
        
        if filter_eth(config, packet.eth) and filter_vlan(config, packet.vlan) and filter_ecpri_raw(config, packet.ecpri) and filter_oran_fh_cus_raw(config, ecpri_payload):

            (ru_port_id, direction, frame, subframe, slot, symbol, start_prb, num_prbs) = parse_oran_fh_cus_raw(config, ecpri_payload)
   
            if (num_prbs == 0):
                num_prbs = PRBS_PER_SYMBOL

            ud_buffer = [format(x, '02x') for x in ecpri_payload[ud_buffer_offset:]]
            validate_ud_buffer(ud_buffer, num_prbs, config['ud_iq_width'], config['ud_comp_method'])
            sniff_ts = float(packet.sniff_timestamp)
            slot_timestamps = ud_buffers[direction][frame][subframe][slot].keys()

            # NOTE: Logic to distinguish repeating SFN numbers
            if len(slot_timestamps) == 0:
                ud_buffers[direction][frame][subframe][slot][sniff_ts][symbol][ru_port_id][(start_prb, num_prbs)] = ud_buffer
            else:
                last_sniff_ts = max(slot_timestamps)
                if sniff_ts > last_sniff_ts + NEW_SFN_THRES:
                    ud_buffers[direction][frame][subframe][slot][sniff_ts][symbol][ru_port_id][(start_prb, num_prbs)] = ud_buffer
                else:
                    ud_buffers[direction][frame][subframe][slot][last_sniff_ts][symbol][ru_port_id][(start_prb, num_prbs)] = ud_buffer

    return ud_buffers


def check_all_symbols_present(packets):
    for direction in packets:
        for frame in packets[direction]:
            for subframe in packets[direction][frame]:
                for slot in packets[direction][frame][subframe]:
                    for timestamp in packets[direction][frame][subframe][slot]:

                        symbols = packets[direction][frame][subframe][slot][timestamp]
                        if len(symbols) != SYMBOLS_PER_SLOT:
                            raise AttributeError("Missing {} symbols for instance of direction={} FrameId={} SubframeId={} Slot={} at {}".format(DATA_DIRECTIONS_REV[direction], frame, subframe, slot, timestamp))

def add_missing_symbols(prbs, eAxC_list, prb_size):
    ZERO_BYTE = ['00']
    START_PRB = 0

    for eAxC in eAxC_list:
        prbs[eAxC][(START_PRB, PRBS_PER_SYMBOL)] = ZERO_BYTE * PRBS_PER_SYMBOL * prb_size

def check_all_eaxc_present(packets):
    eAxC_list = []
    for direction in packets:
        for frame in packets[direction]:
            for subframe in packets[direction][frame]:
                for slot in packets[direction][frame][subframe]:
                    for timestamp in packets[direction][frame][subframe][slot]:
                        for symbol in packets[direction][frame][subframe][slot][timestamp]:
                            if eAxC_list == []:
                                eAxC_list = packets[direction][frame][subframe][slot][timestamp][symbol].keys()
                            else:
                                if eAxC_list == packets[direction][frame][subframe][slot][timestamp][symbol].keys():
                                    continue
                                else:
                                    print(eAxC_list, packets[direction][frame][subframe][slot][timestamp][symbol].keys())
                                    raise AttributeError("Missing {} eAxC for instance of FrameId={} SubframeId={} Slot={} at {}".format(DATA_DIRECTIONS_REV[direction], frame, subframe, slot, timestamp))
    return eAxC_list

def zero_fill_missing_symbols(packets):
    for direction in packets:
        for frame in packets[direction]:
            for subframe in packets[direction][frame]:
                for slot in packets[direction][frame][subframe]:
                    for timestamp in packets[direction][frame][subframe][slot]:
                        #get the eAxC list of the first non-empty symbol
                        eAxC_list = packets[direction][frame][subframe][slot][timestamp][0].keys()
                        for symbol in range(SYMBOLS_PER_SLOT):
                            if symbol not in packets[direction][frame][subframe][slot][timestamp]:
                                add_missing_symbols(packets[direction][frame][subframe][slot][timestamp][symbol], eAxC_list, prb_size)

def add_missing_prbs(prbs, prb_size):
    ZERO_BYTE = ['00']

    sorted_prbs = sorted(prbs, key=lambda tup: tup[0])
    i = 0
    start_prb = 0

    while (start_prb < PRBS_PER_SYMBOL):

        if i >= len(sorted_prbs):
            missing_prb_count = PRBS_PER_SYMBOL - start_prb
            prbs[(start_prb, missing_prb_count)] = ZERO_BYTE * missing_prb_count * prb_size
            start_prb = PRBS_PER_SYMBOL

        elif start_prb == sorted_prbs[i][0]:
            start_prb = sorted_prbs[i][0] + sorted_prbs[i][1]
            i += 1

        elif start_prb < sorted_prbs[i][0]:
            missing_prb_count = sorted_prbs[i][0] - start_prb
            prbs[(start_prb, missing_prb_count)] = ZERO_BYTE * missing_prb_count * prb_size
            start_prb = sorted_prbs[i][0] + sorted_prbs[i][1]
            i += 1


def zero_fill_missing_prbs(packets, prb_size):
    for direction in packets:
        for frame in packets[direction]:
            for subframe in packets[direction][frame]:
                for slot in packets[direction][frame][subframe]:
                    for timestamp in packets[direction][frame][subframe][slot]:
                        for symbol in packets[direction][frame][subframe][slot][timestamp]:
                            for eAxC in packets[direction][frame][subframe][slot][timestamp][symbol]:
                                add_missing_prbs(packets[direction][frame][subframe][slot][timestamp][symbol][eAxC], prb_size)


def dump_tensor(tensor, prefix, direction, frame, subframe, slot, prb_size, timestamp):
    timestamp_short = str(timestamp).split('.')[1][:4]
    filename =  "{}{}_frame_{}_subframe_{}_slot_{}_ts_{}.txt".format(prefix, DATA_DIRECTIONS_REV[direction], frame, subframe, slot, timestamp_short)
    print(f"Generating: {filename}")

    with open(filename, "w") as f:
        for symbol in sorted(tensor):
            for eAxC in sorted(tensor[symbol]):
                # NOTE: Sorting eAxC's!
                for prb in sorted(tensor[symbol][eAxC], key=lambda tup: tup[0]):
                    superrow = tensor[symbol][eAxC][prb]
                    L = int(len(superrow) / prb_size)
                    for k in range(L):
                        row = superrow[k*prb_size:(k+1)*prb_size]
                        f.write(' '.join(row))
                        f.write('\n')


def dump_tensors(packets, prb_size, prefix, dirpath):

    print(f"Output directory: {dirpath}")
    path_and_prefix = dirpath + "/" + prefix
    generated_files = 0

    for direction in packets:
        for frame in packets[direction]:
            for subframe in packets[direction][frame]:
                for slot in packets[direction][frame][subframe]:
                    for timestamp in packets[direction][frame][subframe][slot]:
                        dump_tensor(packets[direction][frame][subframe][slot][timestamp], path_and_prefix, direction, frame, subframe, slot, prb_size, timestamp)
                        generated_files += 1
    print("Generated {} files".format(generated_files))


def debug_print(packets):
    for direction in packets:
        for frame in packets[direction]:
            for subframe in packets[direction][frame]:
                for slot in packets[direction][frame][subframe]:
                    for timestamp in packets[direction][frame][subframe][slot]:
                        for symbol in packets[direction][frame][subframe][slot][timestamp]:
                            for eAxC in packets[direction][frame][subframe][slot][timestamp][symbol]:
                                for prb in packets[direction][frame][subframe][slot][timestamp][symbol][eAxC]:
                                    print(direction, frame, subframe, slot, timestamp, symbol, eAxC, prb, packets[direction][frame][subframe][slot][timestamp][symbol][eAxC][prb])

def write_yaml(obj, filename):
    f=open(filename,'w')
    for field_line in obj._get_all_field_lines():
        if ':' in field_line:
            field_name, field_line = field_line.split(':', 1)
            f.write(field_name + ':')
        f.write(field_line)
    f.close()

if __name__ == "__main__":
    default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    parser = argparse.ArgumentParser(prog='pcap_to_tensor', description="Extract O-RAN FH user data buffer(s) from PCAP", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pcap", help="Input PCAP file", type=str)
    parser.add_argument("-f", "--fapi",   help="Input nvipc fapi file", type=str)
    parser.add_argument("-c", "--config", help="Config file", type=str, default=default_config_path)
    args = parser.parse_args()

    config = parse_yaml(args.config)

    filters = [
        ['src_mac_addr', 'eth.src'],
        ['dst_mac_addr', 'eth.dst'],
        ['vlan_id',      'vlan.id'],
        ['direction',    'oran_fh_cus.data_direction'],
        ['ru_port_id',   'oran_fh_cus.ru_port_id'],
        ['frame_id',     'oran_fh_cus.frameId'],
        ['subframe_id',  'oran_fh_cus.subframe_id'],
        ['slot_id',      'oran_fh_cus.slotId']]

    filterstring = ''
    if config['use_display_filter']:
       for filterName,filterString in filters:
          if config[filterName] != -1:
              if filterstring != '':
                  filterstring += ' and '
              filterstring += filterString + '==' +str(config[filterName])
       print('Using display_filter: '+filterstring+'\n')
    else:
       print(f'Not using display_filter\n')


    if args.fapi:
        filterstring=extract_from_fapi(args.fapi,filterstring)

    try:
        pcap = parse_pcap(args.pcap, config, filterstring)

        if config['raw_mode']:
            packets = filter_packets_raw(pcap, config)
        else:
            packets = filter_packets(pcap, config)

        if config['check_all_symbols_present']:
            check_all_symbols_present(packets)

        prb_size = get_prb_size(config['ud_iq_width'], config['ud_comp_method'])
        if config['zero_fill_missing_prbs']:
            zero_fill_missing_prbs(packets, prb_size)

        if 'zero_fill_missing_symbols' in config and config['zero_fill_missing_symbols']:
            zero_fill_missing_symbols(packets)    

        #debug_print(packets)
        dump_tensors(packets, prb_size, config['output_file_prefix'], config['output_directory'])

    except Exception as e:
            print(e)
    
    finally:
        pcap.close()
