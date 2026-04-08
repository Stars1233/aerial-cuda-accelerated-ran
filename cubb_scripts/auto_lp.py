#!//usr/bin/env python3

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
import glob
import h5py as h5
import json
import numpy as np
import os
import re
import sys
import yaml
from yaml.loader import SafeLoader
import copy

RADIO_FRAME = 10

class TV:
    mappings = {
        1  : "PBCH",
        2  : "PDCCH_DL",
        3  : "PDSCH",
        4  : "CSI_RS",
        6  : "PRACH",
        7  : "PUCCH",
        8  : "PUSCH",
        9  : "SRS",
        10 : "BFW_DL",
        11 : "BFW_UL",
        12 : "PDCCH_UL"
    }
    # def __init__(self, tv_h5, tv_name, slot, types):
    #     self.tv_h5 = tv_h5
    #     self.tv_name = tv_name
    #     self.slot = slot
    #     self.types = types

    def __init__(self, bf, precoding):
        self.bf = bf
        self.precoding = precoding

    def __str__(self):
        return str(vars(self))

    def parse_tv_string(self, tv_name):
        lpc = os.path.basename(tv_name)
        out = [x for x in re.split('TVnr_|_gNB_FAPI_|.h5', lpc) if x]
        slot = int(re.sub('[^0-9]', '', out[1]))
        # print(f'slot = {slot}')
        f = h5.File(tv_name, 'r')

        nPdu = np.array(f['nPdu'][0])
        chList = [ ]
        csirs_row = []
        num_layer_list = defaultdict(list)
        num_prg_list = defaultdict(list)
        num_layer_pdu_idx = {}
        self.mu = f['Cell_Config']['mu'][0]
        for i in range(nPdu[0]):
            dset = 'PDU' + str(i+1)
            if dset not in list(f.keys()):
                continue
            chtype = f[dset]['type'][0]
            channel = TV.mappings[chtype]
            #Get num layers for PUSCH and PDSCH
            if chtype == 3 or chtype == 8:
                num_layer_list[channel].append(f[dset]["nrOfLayers"][0])
            #Get num Rxantena for UL channels 
            self.num_rx_ant = f['Cell_Config']['numRxAnt'][0]
            self.num_tx_ant = f['Cell_Config']['numTxAnt'][0]
            if chtype == 4:
                csirs_row.append( f[dset]['Row'][0])
            if chtype == 2 and f[dset]["dciUL"][0] == 1:
                chtype = 12
            if chtype == 10 and f[dset]["bfwUL"][0] == 1:
                chtype = 11
            chtypeStr = TV.mappings[(chtype)]
            if f[dset].attrs.__contains__('numPRGs') :
                num_prg_list[chtypeStr].append(f[dset]['numPRGs'][0].item())
            else:
                num_prg_list[chtypeStr].append(1)            
            if chtypeStr not in chList:
                chList.append(chtypeStr)
        self.types = []
        self.tv_h5 = lpc
        self. tv_name = out[0]
        self.slot = slot
        self.types = chList
        self.num_layer_list = num_layer_list
        self.csirs_row = csirs_row
        self.num_prg_list = num_prg_list

        # print('Channel List = '.join(self.types))


"""
def parse_tv_string(tv_name):
    if tv_name == None or type(tv_name) != str:
        pass
    if  len(tv_name) == 0:
        pass
    lpc = os.path.basename(tv_name)
    out = [x for x in re.split('TVnr_|_gNB_FAPI_|.h5', lpc) if x]
    slot = int(re.sub('[^0-9]', '', out[1]))
    print(f'slot = {slot}')
    f = h5.File(tv_name, 'r')
    nPdu = np.array(f['nPdu'][0])
    chList = [ ]
    for i in range(nPdu[0]):
        dset = 'PDU' + str(i+1)
        chtype = f[dset]['type'][0]
        print(chtype)
        if chtype == 2 and f[dset]["dciUL"][0] == 1:
            chtype = 12
        if chtype == 10 and f[dset]["bfwUL"][0] == 1:
            chtype = 11
        chtypeStr = TV.mappings[(chtype)]
        if chtypeStr not in chList:
            chList.append(chtypeStr)
        return  TV(lpc, out[0], slot=slot, types = chList)
"""

class LP:
    beam_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    csirs_nPorts = [1, 1, 2, 4, 4, 8, 8, 8, 12, 12, 16, 16, 24, 24, 24, 32, 32, 32]
    start_index = 0

    def __init__(self, tpl, outdir, numcells, all_slots) -> None:
        self.outdir = outdir
        self.numcells = numcells
        self.all_slots = all_slots
        with open(tpl, 'r') as stream:
            self.yf_tpl = yaml.safe_load(stream)
        self.template_name, self.template_ext = os.path.splitext(os.path.basename(tpl))
        self.count = 0
        
        
    def __str__(self):
        return json.dumps(self.yf)


    def __update_single_slot_yaml__(self , slot, beam_data):
        self.yf['SCHED'][slot]['config'] = []
        no_beam_channels=['BFW_DL', 'BFW_UL']
        for cell_index in range(self.numcells):
            self.yf['SCHED'][slot]['config'].append({'cell_index': float(cell_index), 'channels': [] })
            self.yf['SCHED'][slot]['config'][cell_index]['channels'].append(tv.tv_h5)

            # for ch in tv.types:
            #     if ch not in pdu_idx:
            #         pdu_idx[ch] = 0
            #     if ch in no_beam_channels:
            #         num_beams = 0
            #     else:
            #         num_beams = beam_data[ch][pdu_idx[ch]]
            #     if self.start_index + num_beams  > len(self.beam_ids):
            #         beam_ids = (self.beam_ids[self.start_index : ])
            #         for i in range(self.start_index + num_beams - len(self.beam_ids)):
            #             beam_ids.append(self.beam_ids[i])
            #         self.start_index = self.start_index + num_beams - len(self.beam_ids)
            #     else:
            #         beam_ids = (self.beam_ids[self.start_index:self.start_index + num_beams])
            #         self.start_index = self.start_index + num_beams
            #     pdu_idx[ch] += 1

                # if tv.bf:
                #     self.yf['SCHED'][slot]['config'][cell_index]['channels'].append({'type' : ch, 'tv' : tv.tv_name+ '_' + ch, 'beam_ids' : beam_ids })
                # else:
                #     self.yf['SCHED'][slot]['config'][cell_index]['channels'].append({'type' : ch, 'tv' : tv.tv_name+ '_' + ch})

    def write(self, tv):
        self.yf = copy.deepcopy(self.yf_tpl)
        self.yf['Cell_Configs'] = []
        for cell_index in range(self.numcells):
            self.yf['Cell_Configs'].append(tv.tv_h5)
        beam_data = defaultdict(list)
        pdu_idx={}
        slots_per_frame = RADIO_FRAME * (1 << tv.mu)
        if tv.slot >= slots_per_frame:
            num_frames = tv.slot//slots_per_frame + 1
            total_slots = num_frames * slots_per_frame
            tpl_slot_len  = len(self.yf['SCHED'])
            for s in range (tpl_slot_len, total_slots):
                self.yf['SCHED'].append({'slot':s, 'config':{}})
        # for ch in tv.types:
        #     if ch not in self.yf['TV']:
        #         self.yf['TV'][ch] = []
        #         pdu_idx[ch] = 0
        #     self.yf['TV'][ch].append({'name' : tv.tv_name+ '_' + ch, 'path': tv.tv_h5}) 
        #     if "PDSCH" == ch:  
        #         beam_data[ch].append( tv.num_tx_ant * tv.num_prg_list[ch][pdu_idx[ch]] if tv.precoding else tv.num_layer_list[ch][pdu_idx[ch]] * tv.num_prg_list[ch][pdu_idx[ch]] )
        #     elif "PDCCH_DL" == ch or "PDCCH_UL" == ch or "PBCH" == ch:
        #         beam_data[ch].append( tv.num_tx_ant * tv.num_prg_list[ch][pdu_idx[ch]])
        #     elif "PUSCH" == ch or "PUCCH" == ch or "SRS" == ch or "PRACH" == ch:
        #         beam_data[ch].append(tv.num_rx_ant * tv.num_prg_list[ch][pdu_idx[ch]])
        #     elif "CSI_RS" == ch:
        #         beam_data[ch].append(self.csirs_nPorts[tv.csirs_row[pdu_idx[ch]] - 1] * tv.num_prg_list[ch][pdu_idx[ch]])
        #     pdu_idx[ch] += 1
        if not self.all_slots:
            self.__update_single_slot_yaml__(tv.slot, beam_data)
        else:
            numslots = len(self.yf['SCHED'])
            for slot in range(numslots):
                self.__update_single_slot_yaml__(slot, beam_data)

        filename = os.path.join(self.outdir, f'{self.template_name}_{tv.tv_name}{self.template_ext}')
        with open(filename, 'w+') as stream:
            yaml.dump(self.yf, stream, explicit_start=True, explicit_end=True, default_flow_style= None, width=float("inf"), sort_keys=False, default_style='')
        
        self.count += 1

        # with open('launch_pattern_nrSim_1901.yaml', 'r') as stream2:
        #     yf = yaml.safe_load(stream2)
        #     print(yf)
        self.yf['Cell_Configs'] = None
        self.yf['TV'] = None
        self.yf['SCHED'][tv.slot]['config'] = None

if __name__ == "__main__":
    LAUNCH_PATTERN_TEMPLATE = 'launch_pattern_nrSim.yaml'

    script_dir = os.path.realpath(os.path.dirname(__file__))
    cuBB_SDK=os.environ.get("cuBB_SDK", os.path.normpath(os.path.join(script_dir, '..')))
    CUBB_HOME=os.environ.get("CUBB_HOME", cuBB_SDK)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help='Input directory containing Test Vector files')
    parser.add_argument("-o", "--output_dir", default=None, help='Output Directory for Launch Patterns')
    parser.add_argument("-t", '--template_file', default=os.path.join(script_dir, LAUNCH_PATTERN_TEMPLATE), help='Launch Pattern Template')
    parser.add_argument("-c", '--cells', type=int, default=1, help='Number of cells')
    parser.add_argument("-a", '--all_slots', action='store_true', help='Send TV in all slots')
    parser.add_argument("-b", '--bf_enabled', type=int, help='Beam forming enable')
    parser.add_argument("--test_case", help='Specific test case to process (e.g., "1234" for TVnr_1234_gNB_FAPI_s*.h5)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f'Path = {args.input_dir} is not a valid directory')
        sys.exit(-1)

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir

    if not os.path.isdir(output_dir):
        print(f'Path = {output_dir} is not a valid directory')
        sys.exit(-1)

    with open(os.path.join(output_dir, 'out.txt'), "w") as log_file:
        count = 0
        lp = LP(args.template_file, output_dir, args.cells, args.all_slots)
        
        # Define the file pattern based on test_case argument
        if args.test_case:
            file_pattern = f'TVnr_{args.test_case}_gNB_FAPI_s*.h5'
        else:
            file_pattern = '*_gNB_FAPI_*.h5'

        # Search for matching files
        for h5file in glob.glob(os.path.join(args.input_dir, file_pattern)):
            if re.search('PUSCH_HARQ|F01|F08|F13|F14|CP', h5file):
                print('Skipping TV ' + os.path.basename(h5file))
                continue

            precoding = re.search('1010|2031|3248|3249|3250|3251|3252|3253|3254|3258|3259|3260', h5file) is not None

            try:
                tv = TV(args.bf_enabled, precoding)
                tv.parse_tv_string(h5file)
                lp.write(tv)
                count += 1
                log_file.write(f'{tv.tv_name}\n')

            except KeyboardInterrupt:
                raise
            except Exception as x:
                print(f'Exception raised for TV {h5file}:', x)
                continue

            if os.isatty(1):
                print(f'count={count}', end='\r')

        if args.test_case and count == 0:
            print(f'No matching test vectors found for test case {args.test_case}')
            sys.exit(1)

    print(f'Total TVs parsed {count} {lp.count}')

