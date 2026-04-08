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

import numpy as np
import h5py
import os
import re
import pandas as pd
import argparse
import glob
import xlsxwriter

PDCCH_UL_TYPE_INDEX = 99
PRACH_IND_FIELDS = ["PreamblePwr", "TimingAdvance", "TimingAdvanceNano", "preambleIndex"]
UCI_PUCCH_F01_IND_FIELDS = ["HarqValue", "HarqValueFapi1002"]
UCI_PUCCH_F2_IND_FIELDS = ["CsiPart1Payload", "HarqPayload", "SrPayload"]
UCI_PUCCH_F3_IND_FIELDS = ["CsiPart1Payload", "CsiPart2Payload", "HarqPayload", "SrPayload"]
UCI_PUSCH_IND_FIELDS = ["CsiPart1Payload", "CsiPart2Payload", "HarqPayload"]
CRC_IND_FIELD = ["CbCrcStatus"]

def get_keys_from_value(dic, val):
    return [k for k, v in dic.items() if v == val]
    
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
        10 : "BFW",
        PDCCH_UL_TYPE_INDEX : "PDCCH_UL"
    }

    ch_pars_df = {}
    tc_pdu_list_df = None
    count = 0

    def __init__(self):
        for ch in self.mappings.values():
            self.ch_pars_df[ch] = None

        # self.tc_pdu_list_df = pd.DataFrame(self.ch_pars_df)
        self.tc_pdu_list_df = pd.DataFrame({
            'TC': [None],
            'PBCH': [None], 
            'PDCCH_DL': [None],
            'PDSCH': [None],
            'CSI_RS': [None],
            'PRACH': [None],
            'PUCCH': [None],
            'PUSCH': [None],
            'SRS': [None],
            'PDCCH_UL': [None],
            })

    def _create_pdu_ind_mapping(self, h5_obj, num_pdu):
        """Create a mapping table between PDU and IND indices in the given H5 file"""
        pdu_ind_map = [None] * num_pdu
        ind_index = 1
        while 1:
            ind_index_str = "IND" + str(ind_index)
            try:
                pdu_index = h5_obj[ind_index_str]["idxPdu"][0]
                if pdu_ind_map[pdu_index-1] == None:
                    pdu_ind_map[pdu_index-1] = [ind_index]
                else:
                    pdu_ind_map[pdu_index-1].append(ind_index)
                ind_index += 1
            except:
                break
        return pdu_ind_map

    def _parse_channels(self, h5_obj, tc_num, slot):
        pdu_count = {}
        num_pdu = h5_obj['nPdu'][0][0]
        pdu_ind_map = self._create_pdu_ind_mapping(h5_obj, num_pdu)
        for pdu_index in range(1,num_pdu+1):
            pdu_index_str = 'PDU' + str(pdu_index)
            if pdu_index_str in h5_obj:
                ch_type = h5_obj[pdu_index_str]['type'][0]
                if ch_type == 2 and h5_obj[pdu_index_str]['dciUL'][0] == 1:  # PDCCH_UL
                    ch_type = PDCCH_UL_TYPE_INDEX  
                ch_type_str = self.mappings[(ch_type)]

                idx_df = pd.DataFrame({'TC': [tc_num], 'Slot': [slot], 'PDU #': [pdu_index]})
                pdu_df = pd.DataFrame(np.array(list(h5_obj[pdu_index_str][0]), dtype=object)).T
                pdu_df.columns = h5_obj[pdu_index_str].dtype.names

                # DCI
                if ch_type_str == 'PDCCH_DL' or ch_type_str == 'PDCCH_UL':
                    num_dci = pdu_df.at[pdu_df.index[0], 'numDlDci']
                    for dci_pdu_index in range(1, num_dci+1):
                        dci_num_str = pdu_index_str + '_DCI' + str(dci_pdu_index)
                        tmp_dci_df = pd.DataFrame(np.array(list(h5_obj[dci_num_str][0]), dtype=object)).T
                        tmp_dci_df.columns = h5_obj[dci_num_str].dtype.names
                        if num_dci >= 2:
                            if dci_pdu_index == 1:
                                dci_df = tmp_dci_df.copy()
                                for col in dci_df.columns:
                                    dci_df.at[dci_df.index[0], col] = []
                            for col in dci_df.columns:
                                dci_df.at[dci_df.index[0], col].append(tmp_dci_df.at[tmp_dci_df.index[0], col])
                        else:
                            dci_df = tmp_dci_df.copy()

                    # Add prefix 'DCI_' to the column names
                    for col in dci_df.columns:
                        dci_df.rename(columns={col:'DCI_'+col}, inplace=True)

                    pdu_df = pd.concat([pdu_df, dci_df], axis=1)

                # IND
                if ch_type_str == 'PRACH' or ch_type_str == 'PUCCH' or ch_type_str == 'PUSCH':
                    for ind_index in pdu_ind_map[pdu_index-1]:
                        ind_num_str = 'IND' + str(ind_index)
                        ind_df = pd.DataFrame(np.array(list(h5_obj[ind_num_str][0]), dtype=object)).T
                        ind_df.columns = h5_obj[ind_num_str].dtype.names

                        # Determine the type of indication message and prepare to read additional fields according to the indication message
                        if ind_df.at[ind_df.index[0], "type"] == 15:  # UCI.ind for PUSCH
                            prefix = "UCI_PUSCH_IND."
                            additional_fields = UCI_PUSCH_IND_FIELDS
                            additional_fields_prefix = "UCI_PUSCH_IND_"
                        elif ind_df.at[ind_df.index[0], "type"] == 16:  # RACH.ind
                            prefix = "RACH_IND."
                            additional_fields = PRACH_IND_FIELDS
                            additional_fields_prefix = "RACH_IND_"
                        elif ind_df.at[ind_df.index[0], "type"] == 17:  # UCI.ind for PUCCH
                            prefix = "UCI_PUCCH_IND."
                            if ind_df.at[ind_df.index[0], "PucchFormat"] == 0 or ind_df.at[ind_df.index[0], "PucchFormat"] == 1:
                                additional_fields = UCI_PUCCH_F01_IND_FIELDS
                            elif ind_df.at[ind_df.index[0], "PucchFormat"] == 2:
                                additional_fields = UCI_PUCCH_F2_IND_FIELDS
                            elif ind_df.at[ind_df.index[0], "PucchFormat"] == 3:
                                additional_fields = UCI_PUCCH_F3_IND_FIELDS
                            additional_fields_prefix = "UCI_PUCCH_IND_"
                        elif ind_df.at[ind_df.index[0], "type"] == 18:  # CRC.ind
                            prefix = "CRC_IND."
                            additional_fields = CRC_IND_FIELD
                            additional_fields_prefix = "CRC_IND_"
                        else:
                            print("Non defined indication type: ", ind_df.at[ind_df.index[0], "type"])
                            exit(-1)

                        # Clarify the type of indication message by adding indication prefix 'XXX_IND.' to the column names
                        for col in ind_df.columns:
                            ind_df.rename(columns={col:prefix+col}, inplace=True)

                        pdu_df = pd.concat([pdu_df, ind_df], axis=1)

                        # Read additional fields according to the indication massage
                        for field in additional_fields:
                            add_ind_str = 'IND' + str(ind_index) + '_' + field
                            if add_ind_str in h5_obj and h5_obj[add_ind_str].shape[0] != 0 and h5_obj[add_ind_str].shape[1] != 0:  # Check if the field exists and the number of row or column is not empty
                                if h5_obj[add_ind_str].shape[1] == 1:  # if single column
                                    add_ind_df = pd.DataFrame(data=[list(h5_obj[add_ind_str][0])], columns=[additional_fields_prefix+field])
                                else:  # If multiple columns, it makes contents a single list
                                    add_ind_df = pd.DataFrame(data=[[list(h5_obj[add_ind_str][0])]], columns=[additional_fields_prefix+field])
                                pdu_df = pd.concat([pdu_df, add_ind_df], axis=1)

                self.ch_pars_df[ch_type_str] = pd.concat([
                    self.ch_pars_df[ch_type_str], 
                    pd.concat([idx_df, pdu_df], axis=1)])

                # Count each PDU in the current TV                
                if ch_type_str in pdu_count.keys():
                    pdu_count[ch_type_str][0] += 1
                else:
                    pdu_count[ch_type_str] = [1]
                    pdu_count['TC'] = tc_num

        # Concatinate the list of the counted number of PDUs
        self.tc_pdu_list_df = pd.concat([self.tc_pdu_list_df, pd.DataFrame.from_dict(pdu_count)])


    def sort_and_reindex(self):
        for ch in self.mappings.values():
            if self.ch_pars_df[ch] is not None:
                self.ch_pars_df[ch].sort_values('PDU #', inplace=True)
                self.ch_pars_df[ch].sort_values('TC', inplace=True)
                self.ch_pars_df[ch] = self.ch_pars_df[ch].reset_index(drop=True)

        self.tc_pdu_list_df.sort_values('TC', inplace=True)
        self.tc_pdu_list_df = self.tc_pdu_list_df.reset_index(drop=True)
        

    def parse_tv(self, tv_path):
        if tv_path == None or type(tv_path) != str:
            pass
        if  len(tv_path) == 0:
            pass

        tv_name = os.path.basename(tv_path)
        tc_and_slot = [x for x in re.split('TVnr_|_gNB_FAPI_|.h5', tv_name) if x]
        tc_num = tc_and_slot[0]
        slot = int(re.sub('[^0-9]', '', tc_and_slot[1]))

        h5_obj = h5py.File(tv_path, 'r')
        self._parse_channels(h5_obj, tc_num, slot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help='Input Directory of Test Vectors')

    args = parser.parse_args()

    if args.input_dir:
        path = args.input_dir.rstrip('/')
    else:
        path = '/opt/nvidia/cuBB/testVectors/'

    if os.path.isdir(path) == False:
        print ('Path = {path} is not a valid directory')
    else: 
        tv = TV()
        count = 0
        file_list = glob.glob(path + '/*_gNB_FAPI_*.h5') #'/*_gNB_FAPI_*.h5'
        for h5_path in file_list:
            count += 1
            print('[{}/{}] {}'.format(count, len(file_list), h5_path))
            # try:
            tv.parse_tv(h5_path)
            # except:
            #     e = sys.exc_info()[0]
            #     print('Exception raised ' + str(e) + ' for TV ' + os.path.basename(h5_path))
            #     continue
        print(tv.tc_pdu_list_df)
        print()
        tv.sort_and_reindex()

    # Export into Excel
    output_name = 'TV_param_list.xlsx'
    xlWriter = pd.ExcelWriter(output_name, engine='xlsxwriter')
    for ch in tv.mappings.values():
        if tv.ch_pars_df[ch] is not None:
            tv.ch_pars_df[ch].to_excel(xlWriter, sheet_name=ch)
            worksheet = xlWriter.sheets[ch]
            worksheet.autofilter("A1:{}1".format(xlsxwriter.utility.xl_col_to_name(len(tv.ch_pars_df[ch].columns))))  # Enable autofilter
            worksheet.freeze_panes(1,2)  # Freeze panes at B3
            worksheet.set_column(0, len(tv.ch_pars_df[ch].columns), 5)  # Change column width
            workbook = xlWriter.book
            # Rewrite column indices with the custom format
            header_format = workbook.add_format({"rotation": 90, "align": "center", "valign": "top"})  # Change the cell format of the header row
            column_names = ["No."]
            column_names.extend(tv.ch_pars_df[ch].columns.values.tolist())
            for i in range(len(column_names)):
                worksheet.write(0, i, column_names[i], header_format)  

    tv.tc_pdu_list_df.to_excel(xlWriter, sheet_name='TC List')
    worksheet = xlWriter.sheets['TC List']
    worksheet.autofilter("A1:{}1".format(xlsxwriter.utility.xl_col_to_name(len(tv.tc_pdu_list_df.columns))))  # Enable autofilter
    worksheet.freeze_panes(1,2)  # Freeze panes at B3
    worksheet.set_column(0, len(tv.tc_pdu_list_df.columns), 13)  # Change column width
    xlWriter.close()

