#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import glob
import h5py as h5
import json
import math
import numpy as np
import os
import pdb
import re
import sys
import yaml
import traceback

yaml.Dumper.ignore_aliases = lambda *args : True

class TestVector:
    mappings = {
        1  : "PBCH",
        2  : "PDCCH_DL",
        3  : "PDSCH",
        4  : "CSI_RS",
        6  : "PRACH",
        7  : "PUCCH",
        8  : "PUSCH",
        9  : "SRS",
        10 : "BFW"
    }

    channel_eAxC_key_mapping = {
        1 : "eAxC_id_ssb_pbch",
        2 : "eAxC_id_pdcch",
        3 : "eAxC_id_pdsch",
        4 : "eAxC_id_csirs",
        6 : "eAxC_id_prach",
        7 : "eAxC_id_pucch",
        8 : "eAxC_id_pusch",
        9 : "eAxC_id_srs",
        10: "eAxC_id_pdcch"
    }

    channel_tv_key_mapping = {
        1 : "tv_pbch",
        2 : "tv_pdcch_dl",
       #3 : "tv_pdsch", // Hold 
        6 : "tv_prach",
       #8 : "tv_pusch", // Hold ; breaks 7103
        9 : "tv_srs",
       10: "tv_pdcch_ul"
        }

    UPLINK_CHANNEL_SET = set([6, 7, 8, 9])
    # SRS_FLOWLIST =[0x7F40, 0x7F41, 0x7F42, 0x7F43, 0x7F44, 0x7F45, 0x7F46, 0x7F47, 0x7F48, 0x7F49, 0x7F4A, 0x7F4B, 0x7F4C, 0x7F4D, 0x7F4E, 0x7F4F]
    # FIXME: Trying to keep mutually exclusive SRS_FLOWLISTand FLOWLIST and PRACH_FLOWLIST
    # SRS_FLOWLIST = [9,3,5,6,8,0,1,2,8,0,1,2,9,3,5,6]
    FLOWLIST = [ 0x7F00, 0x7F01, 0x7F02, 0x7F03, 0x7F04, 0x7F05, 0x7F06, 0x7F07, 0x7F08, 0x7F09, 0x7F0A, 0x7F0B, 0x7F0C, 0x7F0D, 0x7F0E, 0x7F0F,
                0x7F10, 0x7F11, 0x7F12, 0x7F13, 0x7F14, 0x7F15, 0x7F16, 0x7F17, 0x7F18, 0x7F19, 0x7F1A, 0x7F1B, 0x7F1C, 0x7F1D, 0x7F1E, 0x7F1F]

    PRACH_FLOWLIST =[0x7F20, 0x7F21, 0x7F22, 0x7F23, 0x7F24, 0x7F25, 0x7F26, 0x7F27, 0x7F28, 0x7F29, 0x7F2A, 0x7F2B, 0x7F2C, 0x7F2D, 0x7F2E, 0x7F2F,
                   0x7F30, 0x7F31, 0x7F32, 0x7F33, 0x7F34, 0x7F35, 0x7F36, 0x7F37, 0x7F38, 0x7F39, 0x7F3A, 0x7F3B, 0x7F3C, 0x7F3D, 0x7F3E, 0x7F3F]

    SRS_FLOWLIST =[0x7F40, 0x7F41, 0x7F42, 0x7F43, 0x7F44, 0x7F45, 0x7F46, 0x7F47, 0x7F48, 0x7F49, 0x7F4A, 0x7F4B, 0x7F4C, 0x7F4D, 0x7F4E, 0x7F4F,
                   0x7F50, 0x7F51, 0x7F52, 0x7F53, 0x7F54, 0x7F55, 0x7F56, 0x7F57, 0x7F58, 0x7F59, 0x7F5A, 0x7F5B, 0x7F5C, 0x7F5D, 0x7F5E, 0x7F5F,
                   0x7F60, 0x7F61, 0x7F62, 0x7F63, 0x7F64, 0x7F65, 0x7F66, 0x7F67, 0x7F68, 0x7F69, 0x7F6A, 0x7F6B, 0x7F6C, 0x7F6D, 0x7F6E, 0x7F6F,
                   0x7F70, 0x7F71, 0x7F72, 0x7F73, 0x7F74, 0x7F75, 0x7F76, 0x7F77, 0x7F78, 0x7F79, 0x7F7A, 0x7F7B, 0x7F7C, 0x7F7D, 0x7F7E, 0x7F7F]
    OUTDIR = ''
    DEFAULT_COMPRESSION_BITS = 14
    COMPRESSION_FIXED = 0
    COMPRESSION_BFP = 1
    COMPRESSION_MOD = 4
    DEFAULT_COMPRESSION_METHOD = COMPRESSION_BFP
    DEFAULT_FAPI_ERROR_HANDLE_MASK = 1 << 9
    JUMBO_FRAME = 8192
    SLOT_PERIOD = 20


    def __init__(self):
        self.special_l2adapter_config_file = None
        self.l2adapter_attributes = None
        self.enable_codebook_BF = None
        self.negTV_enable = None
        self.enable_dynamic_BF = None
        self.enable_static_dynamic_beamforming = None
        self.fh_msg_mode = 0 # fhMsgMode: 0 disabled, non-zero enables FH modulation compression mode(s)
        pass

    def get_prach_prb_stride(self,parsed_testvector):


        prach_config = parsed_testvector['Prach_Config']
        preamble_seq_length = prach_config['prachSequenceLength'][0] 
        
        delta_f = 15000 * (2**self.mu)
        if preamble_seq_length == 1: #short preamble

            L_RA = 139

            u = prach_config['prachSubCSpacing'][0] 
            delta_f_RA = 15000 * (2**u)  


        elif preamble_seq_length == 0: #long preamble

            L_RA = 839 
            prachConfigIndex= prach_config['prachConfigIndex'][0]

            if prachConfigIndex < 40: # Preamble format 0,1,2
                delta_f_RA = 1250

            elif prachConfigIndex >=40 and prachConfigIndex <= 66: # Preamble format 3
                    delta_f_RA = 5000

        
        nRA_RB = math.ceil( (L_RA/12.0) * float(delta_f_RA)/delta_f)
        
        #print('L_RA ',L_RA,'delta_f_RA ',delta_f_RA,'delta_f ',delta_f,'nRA_RB ' ,nRA_RB)
        return nRA_RB

    def check_pusch_cases(self, sched_slot=None):

        tc_name = self.tv_name.split('_')[-1]
        tv_name_int = None
        if tc_name.isnumeric():
            tv_name_int = int(self.tv_name.split('_')[-1])
        attr_dict = {}
        attr_dict['l2adapter_attributes'] = True
        self.l2adapter_attributes ['staticPuschSlotNum'] = -1
        if self.algoOptions['enableRssiMeas'] == 1:
            attr_dict['pusch_rssi'] = 1
        if self.algoOptions['enableSinrMeas'] == 1:
            attr_dict['pusch_sinr'] = int(self.algoOptions['pusch_sinr_selector'])
        if self.algoOptions['enableIrc'] == 1:
            attr_dict['pusch_enable_irc'] = 1
        if self.algoOptions['enableCfoCorrection'] == 1:
            attr_dict['pusch_cfo'] = 1
        if self.algoOptions['enableWeightedAverageCfo'] == 1:
            attr_dict['pusch_weighted_average_cfo'] = 1
        if self.algoOptions['enableToEstimation'] == 1:
            attr_dict['pusch_to'] = 1
        if self.algoOptions['TdiMode'] == 1:
            attr_dict['pusch_tdi'] = 1
        if self.algoOptions['enableDftSOfdm'] == 1:
            attr_dict['pusch_dftsofdm'] = 1
        if (tv_name_int and tv_name_int >= 7321 and tv_name_int <= 7323):
            attr_dict['l2adapter_attributes'] = True
            self.l2adapter_attributes ['lbrm'] = 1
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        if sched_slot and ((self.slot - sched_slot) % TestVector.SLOT_PERIOD) != 0:
            attr_dict['l2adapter_attributes'] = True
            self.l2adapter_attributes ['staticPuschSlotNum'] = self.slot
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        if 'staticPuschSlotNum' in self.algoOptions:
            self.l2adapter_attributes ['staticPuschSlotNum'] = self.algoOptions['staticPuschSlotNum'].item()
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        attr_dict['pusch_ldpc_flags'] = int(self.algoOptions['LDPC_flags'])

        if self.enable_dynamic_BF == 1:
            attr_dict['enable_srs'] = 1
            attr_dict['mMIMO_enable'] = 1
            attr_dict['mps_sm_srs'] = 32

        if self.enable_static_dynamic_beamforming == 1:
            attr_dict['enable_srs'] = 1
            attr_dict['mMIMO_enable'] = 1
            attr_dict['mps_sm_srs'] = 32

        if 'ChEst_alg_selector' in self.algoOptions:
            attr_dict['pusch_select_chestalgo'] = int(self.algoOptions['ChEst_alg_selector'])
            
        if 'enablePerPrgChEst' in self.algoOptions:
            attr_dict['pusch_enable_perprgchest'] = int(self.algoOptions['enablePerPrgChEst'])
            
        if 'bfwPowerNormAlg_selector' in self.algoOptions:
            attr_dict['bfw_power_normalization_alg_selector'] = int(self.algoOptions['bfwPowerNormAlg_selector'])
        if 'bfw_beta_prescaler' in self.algoOptions:
            attr_dict['bfw_beta_prescaler'] = int(self.algoOptions['bfw_beta_prescaler'])
            
        if self.numRxPort >= 8:
            attr_dict['ul_order_timeout_gpu_ns'] = 4000000
            
        if self.compression_bits == 16:
            attr_dict['ul_order_timeout_gpu_ns'] = 4000000

        if int(self.algoOptions['enableIrc']) == 1:
            attr_dict['pusch_select_eqcoeffalgo'] = 2
            if int(self.algoOptions['enable_nCov_shrinkage']) == 1:
                if int(self.algoOptions['nCov_shrinkage_method']) == 0:
                    attr_dict['pusch_select_eqcoeffalgo'] = 3
                elif int(self.algoOptions['nCov_shrinkage_method']) == 1:
                    attr_dict['pusch_select_eqcoeffalgo'] = 4
        elif int(self.algoOptions['enableIrc']) == 0: 
            attr_dict['pusch_select_eqcoeffalgo'] = 1


        return attr_dict
        print (tv.name, self.algoOptions)

    def check_pdsch_cases(self, sched_slot=None):
        attr_dict = {}

        if self.algoOptions['enablePrcdBf'] == 1:
            attr_dict['l2adapter_attributes'] = True 
            self.l2adapter_attributes['enable_precoding'] = 1
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            attr_dict['l2adapter_attributes'] = True
            self.l2adapter_attributes['enable_precoding'] = 0
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"

        if sched_slot is not None and ((self.slot - sched_slot) % TestVector.SLOT_PERIOD) != 0:
            attr_dict['l2adapter_attributes'] = True 
            self.l2adapter_attributes ['staticPdschSlotNum'] = self.slot
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            attr_dict['l2adapter_attributes'] = True
            self.l2adapter_attributes ['staticPdschSlotNum'] = -1

        if self.enable_dynamic_BF == 1:
            attr_dict['enable_srs'] = 1
            attr_dict['mMIMO_enable'] = 1
            attr_dict['mps_sm_srs'] = 32

        if self.enable_static_dynamic_beamforming == 1:
            attr_dict['enable_srs'] = 1
            attr_dict['mMIMO_enable'] = 1
            attr_dict['mps_sm_srs'] = 32

        return attr_dict

    def check_pbch_cases(self, sched_slot=None):

        tc_name = self.tv_name.split('_')[-1]
        tv_name_int = None
        if tc_name.isnumeric():
            tv_name_int = int(self.tv_name.split('_')[-1])
        attr_dict = {}
        attr_dict['l2adapter_attributes'] = True 
        self.l2adapter_attributes.update({'staticSsbSFN': self.SFN, 'staticSsbPcid': -1})
        if tv_name_int and (tv_name_int == 228 or tv_name_int == 229):
            self.l2adapter_attributes['staticSsbSlotNum'] = 10
        else:
            self.l2adapter_attributes['staticSsbSlotNum'] = -1

        if self.algoOptions['enablePrcdBf'] == 1:  # FIXME This has to be populated by 5GModel; it isn't right now
            self.l2adapter_attributes['enable_precoding'] = 1
        else:
            self.l2adapter_attributes['enable_precoding'] = 0
        
        # LP based pattern
        if sched_slot is not None:
            if ((self.slot - sched_slot) % TestVector.SLOT_PERIOD) != 0:
                self.l2adapter_attributes['staticSsbSlotNum'] = -1
                self.l2adapter_attributes['staticSsbPcid'] = -1
        # TV based pattern
        else:
            self.l2adapter_attributes['staticSsbPcid'] = self.PCID


        self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"

        return attr_dict

    def check_pdcch_cases(self, sched_slot=None):
        attr_dict = {}
        attr_dict['l2adapter_attributes'] = True

        #FIXME if we always generate a special file we can just do:
        self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"

        if self.algoOptions['enablePrcdBf'] == 1:  # FIXME This has to be populated by 5GModel; it isn't right now
            self.l2adapter_attributes['enable_precoding'] = 1
        else:
            self.l2adapter_attributes['enable_precoding'] = 0

        if sched_slot and ((self.slot - sched_slot) % TestVector.SLOT_PERIOD) != 0:
            self.l2adapter_attributes ['staticPdcchSlotNum'] = self.slot
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            self.l2adapter_attributes['staticPdcchSlotNum'] = -1

        return attr_dict

    def check_csirs_cases(self, sched_slot=None):
        attr_dict = {}
        attr_dict['l2adapter_attributes'] = True

        #FIXME if we always generate a special file
        self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        if self.algoOptions['enablePrcdBf'] == 1:  # FIXME This has to be populated by 5GModel; it isn't right now
            self.l2adapter_attributes['enable_precoding'] = 1
        else:
            self.l2adapter_attributes['enable_precoding'] = 0
        if sched_slot and ((self.slot - sched_slot) % TestVector.SLOT_PERIOD) != 0:
            self.l2adapter_attributes ['staticCsiRsSlotNum'] = self.slot
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            self.l2adapter_attributes['staticCsiRsSlotNum'] = -1
        return attr_dict

    def check_srs_cases(self, sched_slot=None):
        attr_dict = {}
        attr_dict['l2adapter_attributes'] = True

        self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        if self.algoOptions['enablePrcdBf'] == 1:  # FIXME This has to be populated by 5GModel; it isn't right now
            self.l2adapter_attributes['enable_precoding'] = 1
        else:
            self.l2adapter_attributes['enable_precoding'] = 0
        if sched_slot and ((self.slot - sched_slot) % TestVector.SLOT_PERIOD) != 0:
            self.l2adapter_attributes ['staticSrsSlotNum'] = self.slot
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            self.l2adapter_attributes['staticSrsSlotNum'] = -1

        attr_dict['enable_srs'] = 1
        
        if self.enable_dynamic_BF == 1:
            attr_dict['mMIMO_enable'] = 1
            attr_dict['mps_sm_srs'] = 32

        if self.enable_static_dynamic_beamforming == 1:
            attr_dict['mMIMO_enable'] = 1
            attr_dict['mps_sm_srs'] = 32

        if self.algoOptions['srs_chEst_toL2_normalization_algo_selector'] == 1:
            attr_dict['srs_chest_tol2_normalization_algo_type'] = 1
            attr_dict['srs_chest_tol2_constant_scaler'] = float(0)
        else:
            attr_dict['srs_chest_tol2_normalization_algo_type'] = 0
            attr_dict['srs_chest_tol2_constant_scaler'] = float(self.algoOptions['srs_chEst_toL2_constant_scaler'])

        return attr_dict

    def check_bfw_cases(self, sched_slot=None):
        attr_dict = {}
        attr_dict['l2adapter_attributes'] = True

        self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        if self.algoOptions['enablePrcdBf'] == 1:  # FIXME This has to be populated by 5GModel; it isn't right now
            self.l2adapter_attributes['enable_precoding'] = 1
        else:
            self.l2adapter_attributes['enable_precoding'] = 0
        if sched_slot and ((self.slot - sched_slot) % TestVector.SLOT_PERIOD) != 0:
            self.l2adapter_attributes ['staticSrsSlotNum'] = self.slot
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            self.l2adapter_attributes['staticSrsSlotNum'] = -1

        if self.enable_dynamic_BF == 1:
            attr_dict['enable_srs'] = 1
            attr_dict['mMIMO_enable'] = 1
            attr_dict['mps_sm_srs'] = 32

        if self.enable_static_dynamic_beamforming == 1:
            attr_dict['enable_srs'] = 1
            attr_dict['mMIMO_enable'] = 1
            attr_dict['mps_sm_srs'] = 32

        return attr_dict

    def check_pucch_cases(self, sched_slot=None):
        attr_dict = {}
        attr_dict['l2adapter_attributes'] = True

        self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        if sched_slot and ((self.slot - sched_slot) % TestVector.SLOT_PERIOD) != 0:
            self.l2adapter_attributes ['staticPucchSlotNum'] = self.slot
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            self.l2adapter_attributes['staticPucchSlotNum'] = -1
        return attr_dict

    def get_flow_list(self, chtype = None, numTxPort= None, numRxPort = None, numRxAntSrs = None):
        chFlowList = []
        if chtype in TestVector.UPLINK_CHANNEL_SET:     # Uplink Channel
            if chtype == 6: #PRACH
                chFlowList = [x for x in TestVector.PRACH_FLOWLIST[0:numRxPort]]
            elif chtype == 9:
                chFlowList = [x for x in TestVector.SRS_FLOWLIST[0:numRxAntSrs]]
            else:
                chFlowList = [ x for x in TestVector.FLOWLIST[0:numRxPort]]
        else:                                           # Downlink channel
            chFlowList = [ x for x in TestVector.FLOWLIST[0:numTxPort]]
        return chFlowList

    def parse_tv_string(self, tv_name, sched_slot=None):
        
        if tv_name == None or type(tv_name) != str:
            pass
        
        if  len(tv_name) == 0:
            pass
        lpc = os.path.basename(tv_name)
        out = [x for x in re.split('TVnr_|_gNB_FAPI_|.h5', lpc) if x]
        slot = int(re.sub('[^0-9]', '', out[1]))
        parsed_testvector = h5.File(tv_name, 'r')

        #Cell_Config
        numTxPort = parsed_testvector['Cell_Config']['numTxPort'][0]
        numRxPort = parsed_testvector['Cell_Config']['numRxPort'][0]
        self.numRxPort = numRxPort
        numRxAntSrs = parsed_testvector['Cell_Config']['numRxAntSrs'][0]
        self.mu = parsed_testvector['Cell_Config']['mu'][0]
        optionVals = list(parsed_testvector['Alg_Config'][0])
        optionKeys = list(parsed_testvector['Alg_Config'].dtype.names)
        self.algoOptions = dict(zip(optionKeys, optionVals))
        cellConfigKeys = parsed_testvector['Cell_Config'].dtype.names
        if 'enable_codebook_BF' in cellConfigKeys:
            self.enable_codebook_BF = bool(parsed_testvector['Cell_Config']['enable_codebook_BF'][0])

        self.compression_bits = TestVector.DEFAULT_COMPRESSION_BITS
        self.dl_comp_meth = TestVector.DEFAULT_COMPRESSION_METHOD
        self.ul_comp_meth = TestVector.DEFAULT_COMPRESSION_METHOD
        if 'BFPforCuphy' in list(parsed_testvector.keys()):
            self.compression_bits = parsed_testvector['BFPforCuphy'][0][0]
        if 'FixedPointforCuphy' in list(parsed_testvector.keys()):
            if(0 != int(parsed_testvector['FixedPointforCuphy'][0][0])):
                self.compression_bits = parsed_testvector['FixedPointforCuphy'][0][0]
                self.dl_comp_meth = TestVector.COMPRESSION_FIXED
                self.ul_comp_meth = TestVector.COMPRESSION_FIXED
        if 'fhMsgMode' in list(parsed_testvector.keys()):
            self.fh_msg_mode = int(parsed_testvector['fhMsgMode'][0][0])
            if self.fh_msg_mode != 0:
                self.dl_comp_meth = TestVector.COMPRESSION_MOD

        if 'ul_gain_calibration' in cellConfigKeys:
            self.ul_gain_calibration = float(parsed_testvector['Cell_Config']['ul_gain_calibration'][0])
            # print(cellConfigKeys, self.ul_gain_calibration, tv_name)


        if 'max_amp_ul' in cellConfigKeys:
            self.max_amp_ul = int(parsed_testvector['Cell_Config']['max_amp_ul'][0])
        
        if 'negTV_enable' in cellConfigKeys:
            self.negTV_enable = int(parsed_testvector['Cell_Config']['negTV_enable'][0])

        if 'enable_dynamic_BF' in cellConfigKeys:
            self.enable_dynamic_BF = int(parsed_testvector['Cell_Config']['enable_dynamic_BF'][0])

        if 'enable_static_dynamic_beamforming' in cellConfigKeys:
            self.enable_static_dynamic_beamforming = int(parsed_testvector['Cell_Config']['enable_static_dynamic_beamforming'][0])

        self.pusch_ldpc_max_num_itr_algo_type = int(self.algoOptions['LDPC_DMI_method'])

        # PDUX
        nPdu = np.array(parsed_testvector['nPdu'][0])
        chList = [ ]
        updateChanneleAxC = [ ]
        updateTVChannel = [ ]
        keyCache = set()

        for chtype in TestVector.mappings:
            chFlowList =  self.get_flow_list(chtype, numTxPort, numRxPort, numRxAntSrs)
            eAxC_key = TestVector.channel_eAxC_key_mapping[chtype]
            if eAxC_key not in keyCache:
                updateChanneleAxC.append( (eAxC_key, chFlowList))
                keyCache.add(eAxC_key)

        for i in range(nPdu[0]):
            dset = 'PDU' + str(i+1)
            chtype = parsed_testvector[dset]['type'][0]
            if chtype == 2 and parsed_testvector[dset]["dciUL"][0] == 1:
                chtype = 10
            # get channel name
            chtypeStr = TestVector.mappings[(chtype)]
            if chtypeStr not in chList:
                chList.append(chtypeStr)
            # determine eAxC IDs
            # if chtype in TestVector.UPLINK_CHANNEL_SET:     # Uplink Channel
            #     if chtype == 6: #PRACH
            #         chFlowList = [x for x in TestVector.PRACH_FLOWLIST[0:numRxPort]]
            #     else:
            #         chFlowList = [ x for x in TestVector.FLOWLIST[0:numRxPort]]
            # else:                                           # Downlink channel
            #     chFlowList = [ x for x in TestVector.FLOWLIST[0:numTxPort]]
            chFlowList = self.get_flow_list(chtype, numTxPort, numRxPort, numRxAntSrs)

            if chtype not in TestVector.UPLINK_CHANNEL_SET:
                if chtype == 1:
                    self.SFN = parsed_testvector['SFN'][0][0].item()
                    self.PCID = parsed_testvector['Cell_Config']['phyCellId'][0].item()
            eAxC_key = TestVector.channel_eAxC_key_mapping[chtype]
            if eAxC_key not in keyCache:
                updateChanneleAxC.append( (eAxC_key, chFlowList))
                keyCache.add(eAxC_key)
            tv_key = None
            if chtype in TestVector.channel_tv_key_mapping:
                tv_key = TestVector.channel_tv_key_mapping[chtype]
                tv_val = f"TVnr_{out[0]}_gNB_FAPI_{out[1]}.h5"

        #self.pusch_prb_stride = int(parsed_testvector['Cell_Config']['ulGridSize'][0])
        #self.prach_prb_stride = self.get_prach_prb_stride(parsed_testvector)
        self.tv_h5 = lpc
        self.tv_name = out[0]
        self.slot = slot
        self.types = chList
        self.eAxCList = updateChanneleAxC
        self.tvChannelList =  updateTVChannel
        self.l2adapter_attributes = {}
        self.nPdu = int(nPdu[0].item())

        self.pusch_special_attributes = self.check_pusch_cases(sched_slot) if 'PUSCH' in self.types else {}
        self.pdsch_special_attributes = self.check_pdsch_cases(sched_slot) if 'PDSCH' in self.types else {}
        self.pbch_special_attributes = self.check_pbch_cases(sched_slot) if 'PBCH' in self.types else {}
        self.pdcch_special_attributes = self.check_pdcch_cases(sched_slot) if 'PDCCH_DL' in self.types else {}
        self.csirs_special_attributes = self.check_csirs_cases(sched_slot) if 'CSI_RS' in self.types else {}
        self.srs_special_attributes = self.check_srs_cases(sched_slot) if 'SRS' in self.types else {}
        self.pucch_special_attributes = self.check_pucch_cases(sched_slot) if 'PUCCH' in self.types else {}
        self.bfw_special_attributes = self.check_bfw_cases(sched_slot) if 'BFW' in self.types else {}

        if self.enable_codebook_BF:
            self.l2adapter_attributes.update({'enable_beam_forming': int(self.enable_codebook_BF)})
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            self.l2adapter_attributes.update({'enable_beam_forming': 0})
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        
        if self.negTV_enable==1: # 1: invalid PUSCH test case
            self.l2adapter_attributes.update({'fapi_config_check_mask': TestVector.DEFAULT_FAPI_ERROR_HANDLE_MASK})
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"
        else:
            self.l2adapter_attributes.update({'fapi_config_check_mask': 0})
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{self.mu}_{self.tv_name.zfill(4)}.yaml"            

        #print(self.eAxCList)

class LaunchPattern:
    def __init__(self):
        self.celltvs = None
        self.name = None
        self.yaml_file = None
        self.cells = None
        self.l2adapter_attributes = None
        self.special_l2adapter_config_file = None
        self.mus = None
        self.has_fh_msg_mode = False

    def parse_launch_pattern_file(self, lpname):
        if lpname == None or  type(lpname) != str:
            pass
        if len(lpname) == 0:
            pass
        
        with open(lpname, 'r') as template_content:
            self.yaml_file = yaml.safe_load(template_content)
        tmpdirname = os.path.dirname(lpname)
        dirname = tmpdirname.rstrip("/multi-cell")
        self.name = get_lp_name(lpname)
        self.name_str = str(self.name).zfill(4)
        self.l2adapter_attributes = { }
        num_cells = len(self.yaml_file['Cell_Configs'])
        celltvs = self.yaml_file['SCHED']
        tvlist = []
        self.cells = []
        for i in range(len(self.yaml_file['Cell_Configs'])):
            #self.cells.append({"eAxCChList": {}, "pusch_prb_stride" : 0, "prach_prb_stride" : 0, "pdsch_special_attributes": {}, "pusch_special_attributes": {}, "pbch_special_attributes": {}, "pdcch_special_attributes": {}, "csirs_special_attributes": {}, "srs_special_attributes": {}, "pucch_special_attributes" :{}, "l2adapter_attributes": {}})
            self.cells.append({"eAxCChList": {}, "pdsch_special_attributes": {}, "pusch_special_attributes": {}, "pbch_special_attributes": {}, "pdcch_special_attributes": {}, "csirs_special_attributes": {}, "srs_special_attributes": {}, "pucch_special_attributes" :{}, "l2adapter_attributes": {}, "bfw_special_attributes": {}})
        self.mus = []
        special_l2a_file = False
        # Initialize comp_meth from Cell_Configs so SCHED order doesn't override it
        for config_index in range(len(self.yaml_file['Cell_Configs'])):
            cfg_tv_rel = self.yaml_file['Cell_Configs'][config_index]
            cfg_tv_abs = dirname + f"/{cfg_tv_rel}"
            tv_cfg = TestVector()
            if(num_cells > 1):
                tv_cfg.multicell_test = 1
            tv_cfg.parse_tv_string(cfg_tv_abs, None)
            if tv_cfg.fh_msg_mode != 0:
                self.has_fh_msg_mode = True
            self.cells[config_index]['dl_comp_meth'] = int(tv_cfg.dl_comp_meth)
        for slotindex in range(len(celltvs)):
            cell_configs = celltvs[slotindex]['config']
            if cell_configs:
                for config_index in range(len(cell_configs)):
                    tvnames = cell_configs[config_index]['channels']
                    if len(tvnames) == 0:
                        continue
                    for tvindex in range(len(tvnames)):
                        tvlist.append(tvnames[tvindex])
                        abstvname = dirname + f"/{tvnames[tvindex]}"
                        tv = TestVector()
                        if(num_cells > 1):
                            tv.multicell_test = 1
                        tv.parse_tv_string(abstvname, slotindex)
                        if tv.fh_msg_mode != 0:
                            self.has_fh_msg_mode = True
                        self.mus.append(int(tv.mu))
                        for key,eaxc_channel_val in tv.eAxCList:
                            if key not in self.cells[config_index]["eAxCChList"]:
                                self.cells[config_index]["eAxCChList"][key] = eaxc_channel_val
                        #if self.cells[config_index]["pusch_prb_stride"] == 0 and tv.pusch_prb_stride != None:
                        #   self.cells[config_index]["pusch_prb_stride"] = tv.pusch_prb_stride
                        #if self.cells[config_index]["prach_prb_stride"] == 0 and tv.prach_prb_stride != None:
                        #   self.cells[config_index]["prach_prb_stride"] = tv.prach_prb_stride 
                        if tv.pdsch_special_attributes:
                            self.cells[config_index]["pdsch_special_attributes"].update(tv.pdsch_special_attributes)
                        if tv.pusch_special_attributes:
                            #Save the current state of self.cells[config_index]["pusch_special_attributes"]['pusch_subSlotProcEn']
                            #before updating pusch_special_attributes . If it was set to 1. Restore it after updating 
                            # pusch_special_attributes
                            self.cells[config_index]["pusch_special_attributes"].update(tv.pusch_special_attributes)
                        if tv.pbch_special_attributes:
                            self.cells[config_index]["pbch_special_attributes"].update(tv.pbch_special_attributes)
                        if tv.pdcch_special_attributes:
                            self.cells[config_index]["pdcch_special_attributes"].update(tv.pdcch_special_attributes)
                        if tv.csirs_special_attributes:
                            self.cells[config_index]["csirs_special_attributes"].update(tv.csirs_special_attributes)
                        if tv.srs_special_attributes:
                            self.cells[config_index]["srs_special_attributes"].update(tv.srs_special_attributes)
                        if tv.pucch_special_attributes:
                            self.cells[config_index]["pucch_special_attributes"].update(tv.pucch_special_attributes)
                        if tv.l2adapter_attributes:
                            self.l2adapter_attributes.update(tv.l2adapter_attributes)
                        if tv.bfw_special_attributes:
                            self.cells[config_index]["bfw_special_attributes"].update(tv.bfw_special_attributes)

                        self.cells[config_index]["compression_bits"] = int(tv.compression_bits)
                        self.cells[config_index]['ul_comp_meth'] = int(tv.ul_comp_meth)
                        self.cells[config_index]['ul_gain_calibration'] = tv.ul_gain_calibration
                        self.cells[config_index]['max_amp_ul'] = tv.max_amp_ul
                        self.cells[config_index]['enable_dynamic_BF'] = tv.enable_dynamic_BF
                        self.cells[config_index]['enable_static_dynamic_beamforming'] = tv.enable_static_dynamic_beamforming
                        self.cells[config_index]['pusch_ldpc_max_num_itr_algo_type'] = tv.pusch_ldpc_max_num_itr_algo_type
                    
                        # if tv.special_l2adapter_config_file:
                        #     special_l2a_file = True
                            
        if len(self.mus) > 0:
            self.special_l2adapter_config_file = f"l2_adapter_config_nrSim_SCF_mu_{max(self.mus)}_{str(self.name).zfill(4)}.yaml"
            # print(self.special_l2adapter_config_file)

        
class CuphyConfiguration:


    def __init__(self, template_file, outdir):
        
        self.count = 0
        self.template_file = template_file
        self.template_name, self.template_ext = os.path.splitext(os.path.basename(template_file))
        self.outdir = outdir
        self.lpcount = 0

        
    def get_fresh_template(self):
        with open(self.template_file, 'r') as template_content:
            self.yaml_file = yaml.safe_load(template_content)
        
    def __str__(self):
        return json.dumps(self.yaml_file)

    def write(self, tv, modify_all_cells=False):
        
        self.get_fresh_template()

        modify_cell_count = 1
        if modify_all_cells==True:
            modify_cell_count = len(self.yaml_file['cuphydriver_config']['cells'])

        for cell_idx in range(modify_cell_count):
            
            #eAxC update
            for key,flowlist in tv.eAxCList:
                self.yaml_file['cuphydriver_config']['cells'][cell_idx][key]= flowlist
                
            #tv_channel update
            for key,tv_channel_val in tv.tvChannelList:
                self.yaml_file['cuphydriver_config']['cells'][cell_idx][key] = tv_channel_val

            #stride(s)
            #self.yaml_file['cuphydriver_config']['cells'][cell_idx]['pusch_prb_stride'] = tv.pusch_prb_stride
            #self.yaml_file['cuphydriver_config']['cells'][cell_idx]['prach_prb_stride'] = tv.prach_prb_stride

            # print(tv.tv_name, tv.compression_bits)
            # self.yaml_file['cuphydriver_config']['cells'][cell_idx]['compression_bits'] = tv.compression_bits
            # self.yaml_file['cuphydriver_config']['cells'][cell_idx]['decompression_bits'] = tv.compression_bits

            # pusch attributes : pusch_tdi, pusch_cfo
            if tv.pusch_special_attributes:
                for key,attr_value in tv.pusch_special_attributes.items():
                    if key != 'pusch_ldpc_flags':
                        self.yaml_file['cuphydriver_config'][key] = attr_value
                    else:
                        self.yaml_file['cuphydriver_config']['cells'][cell_idx][key] = attr_value

            if tv.pdsch_special_attributes:
                for key,attr_value in tv.pdsch_special_attributes.items():
                    self.yaml_file['cuphydriver_config'][key] = attr_value
            
            if tv.srs_special_attributes:
                for key,attr_value in tv.srs_special_attributes.items():
                    self.yaml_file['cuphydriver_config'][key] = attr_value
            
            if tv.bfw_special_attributes:
                for key,attr_value in tv.bfw_special_attributes.items():
                    self.yaml_file['cuphydriver_config'][key] = attr_value
                self.yaml_file['cuphydriver_config']['nics'][0]['mtu'] = TestVector.JUMBO_FRAME

            if tv.nPdu > 35: # Large numbers of PDUs in a slot can cause FH to generate frames larger than 1500 bytes which will be dropped with lower NIC MTU
                self.yaml_file['cuphydriver_config']['nics'][0]['mtu'] = TestVector.JUMBO_FRAME

            if tv.ul_gain_calibration:
                self.yaml_file['cuphydriver_config']['cells'][cell_idx]['ul_gain_calibration'] = tv.ul_gain_calibration
            if tv.max_amp_ul:
                self.yaml_file['cuphydriver_config']['cells'][cell_idx]['max_amp_ul'] = tv.max_amp_ul

        #mu
        if tv.special_l2adapter_config_file:
            # Specific l2 adapter config file generation is required
            self.yaml_file['l2adapter_filename'] = tv.special_l2adapter_config_file
        else:
            self.yaml_file['l2adapter_filename'] = f"l2_adapter_config_nrSim_SCF_mu_{tv.mu}.yaml"

            
        filename = os.path.join(self.outdir, f'{self.template_name}_{tv.tv_name}{self.template_ext}')
        self.filename = filename
        with open(filename, 'w+') as stream:
            yaml.dump(self.yaml_file, stream, explicit_start=True, explicit_end=True, default_flow_style=None, width=float("inf"), sort_keys=False ,default_style='')
        
        self.count += 1

    def write_lp_config(self, lp):
        self.get_fresh_template()

        # If any TV in the launch pattern enables fhMsgMode, ensure mMIMO_enable is set in output config.
        if getattr(lp, "has_fh_msg_mode", False):
            self.yaml_file['cuphydriver_config']['mMIMO_enable'] = 1

        if (type(lp.name) is int and lp.name >= 90024 and lp.name <= 90026):
            self.yaml_file['cuphydriver_config']['pusch_aggr_per_ctx'] = 15
            self.yaml_file['cuphydriver_config']['prach_aggr_per_ctx'] = 8
            self.yaml_file['cuphydriver_config']['pucch_aggr_per_ctx'] = 8
            self.yaml_file['cuphydriver_config']['ul_input_buffer_per_cell'] = 21

        if (type(lp.name) is int and lp.name >= 90061 and lp.name <= 90063):
            self.yaml_file['cuphydriver_config']['pusch_aggr_per_ctx'] = 9
            self.yaml_file['cuphydriver_config']['prach_aggr_per_ctx'] = 4
            self.yaml_file['cuphydriver_config']['ul_input_buffer_per_cell'] = 15

        if type(lp.name) is int and lp.name in [90027, 90028, 90029, 90064]:
            self.yaml_file['cuphydriver_config']['pusch_aggr_per_ctx'] = 9
            self.yaml_file['cuphydriver_config']['prach_aggr_per_ctx'] = 6
            self.yaml_file['cuphydriver_config']['pucch_aggr_per_ctx'] = 6
            self.yaml_file['cuphydriver_config']['ul_input_buffer_per_cell'] = 15

        if type(lp.name) is int and lp.name >= 90065 and lp.name <= 90067:
            self.yaml_file['cuphydriver_config']['ul_srs_aggr3_task_launch_offset_ns'] = 4000000
            self.yaml_file['cuphydriver_config']['pusch_aggr_per_ctx'] = 9
            self.yaml_file['cuphydriver_config']['prach_aggr_per_ctx'] = 6
            self.yaml_file['cuphydriver_config']['pucch_aggr_per_ctx'] = 6
            self.yaml_file['cuphydriver_config']['srs_aggr_per_ctx'] = 6
            self.yaml_file['cuphydriver_config']['ul_input_buffer_per_cell'] = 15
            self.yaml_file['cuphydriver_config']['ul_input_buffer_per_cell_srs'] = 8

        if type(lp.name) is int and (lp.name in [90159, 90160]):
            self.yaml_file["cuphydriver_config"]["split_ul_cuda_streams"] = 0
            self.yaml_file["cuphydriver_config"]["max_harq_tx_count_bundled"] = 10
            self.yaml_file["cuphydriver_config"]["max_harq_tx_count_non_bundled"] = 4

        if (type(lp.name) is int and lp.name == 90626 or lp.name == 90622): # temporarily for Rel-25-2
            self.yaml_file['cuphydriver_config']['pusch_aggr_per_ctx'] = 4
            self.yaml_file['cuphydriver_config']['srs_aggr_per_ctx'] = 5

        if (type(lp.name) is int and lp.name >= 90700 and lp.name <= 90705):
            self.yaml_file['cuphydriver_config']['prach_aggr_per_ctx'] = 6
            self.yaml_file['cuphydriver_config']['ul_input_buffer_per_cell'] = 24
            self.yaml_file['cuphydriver_config']['pucch_aggr_per_ctx'] = 6
            self.yaml_file['cuphydriver_config']['srs_aggr_per_ctx'] = 6
            self.yaml_file['cuphydriver_config']['pusch_aggr_per_ctx'] = 6

        if (type(lp.name) is int and lp.name > 90605):
            self.yaml_file['cuphydriver_config']['fix_beta_dl'] = 0
            self.yaml_file['cuphydriver_config']['mps_sm_srs'] = 32
            self.yaml_file['cuphydriver_config']['mps_sm_ul_order'] = 10
            self.yaml_file['cuphydriver_config']['dlc_bfw_enable_divide_per_cell'] = 1
            self.yaml_file['cuphydriver_config']['ulc_alloc_cplane_bfw_txq'] = 1
            self.yaml_file['cuphydriver_config']['use_batched_memcpy'] = 1
            self.yaml_file['cuphydriver_config']['ul_srs_aggr3_task_launch_offset_ns'] = 500000
            self.yaml_file['cuphydriver_config']['pusch_aggr_per_ctx'] = 6

        if type(lp.name) is int and (lp.name in [90200, 90201]):
            self.yaml_file['cuphydriver_config']['fix_beta_dl'] = 1

        if len(lp.cells) > 0:
            self.yaml_file['cuphydriver_config']['cell_group_num'] = len(lp.cells)
        for cell_index in range(len(lp.cells)):

            self.yaml_file['cuphydriver_config']['cells'][cell_index].update(lp.cells[cell_index]['eAxCChList'])
            #self.yaml_file['cuphydriver_config']['cells'][cell_index]['pusch_prb_stride'] = lp.cells[cell_index]['pusch_prb_stride']
            #self.yaml_file['cuphydriver_config']['cells'][cell_index]['prach_prb_stride'] = lp.cells[cell_index]['prach_prb_stride']

            self.yaml_file['cuphydriver_config']['cells'][cell_index]['dl_iq_data_fmt']['comp_meth'] = lp.cells[cell_index]['dl_comp_meth']
            self.yaml_file['cuphydriver_config']['cells'][cell_index]['ul_iq_data_fmt']['comp_meth'] = lp.cells[cell_index]['ul_comp_meth']
            self.yaml_file['cuphydriver_config']['cells'][cell_index]['dl_iq_data_fmt']['bit_width'] = lp.cells[cell_index]['compression_bits']
            self.yaml_file['cuphydriver_config']['cells'][cell_index]['ul_iq_data_fmt']['bit_width'] = lp.cells[cell_index]['compression_bits']
            self.yaml_file['cuphydriver_config']['cells'][cell_index]['ul_gain_calibration'] = lp.cells[cell_index]['ul_gain_calibration']
            self.yaml_file['cuphydriver_config']['cells'][cell_index]['max_amp_ul'] = lp.cells[cell_index]['max_amp_ul']
            self.yaml_file['cuphydriver_config']['cells'][cell_index]['pusch_ldpc_max_num_itr_algo_type'] = lp.cells[cell_index]['pusch_ldpc_max_num_itr_algo_type']

            if len(lp.cells[cell_index]['pusch_special_attributes']) > 0:
                self.yaml_file['cuphydriver_config'].update(lp.cells[cell_index]['pusch_special_attributes'])

            if len(lp.cells[cell_index]['pdsch_special_attributes']) > 0:
                self.yaml_file['cuphydriver_config'].update(lp.cells[cell_index]['pdsch_special_attributes'])

            if len(lp.cells[cell_index]['srs_special_attributes']) > 0:
                self.yaml_file['cuphydriver_config'].update(lp.cells[cell_index]['srs_special_attributes'])
                if  lp.cells[cell_index]['enable_dynamic_BF'] == 1:
                    self.yaml_file['cuphydriver_config']['nics'][0]['mtu'] = TestVector.JUMBO_FRAME
                if  lp.cells[cell_index]['enable_static_dynamic_beamforming'] == 1:
                    self.yaml_file['cuphydriver_config']['nics'][0]['mtu'] = TestVector.JUMBO_FRAME

            if len(lp.cells[cell_index]['bfw_special_attributes']) > 0:
                self.yaml_file['cuphydriver_config'].update(lp.cells[cell_index]['bfw_special_attributes'])
                self.yaml_file['cuphydriver_config']['nics'][0]['mtu'] = TestVector.JUMBO_FRAME

            if (type(lp.name) is int and lp.name > 90605):
                self.yaml_file['cuphydriver_config']['cells'][cell_index]['T1a_max_cp_ul_ns'] = 535000
                self.yaml_file['cuphydriver_config']['cells'][cell_index]['Ta4_min_ns_srs'] = 676300
                self.yaml_file['cuphydriver_config']['cells'][cell_index]['Ta4_max_ns_srs'] = 1800300
                self.yaml_file['cuphydriver_config']['cells'][cell_index]['Tcp_adv_dl_ns'] = 324000

            if (type(lp.name) is int and lp.name == 90302): # Single Section TV TestCase
                self.yaml_file['cuphydriver_config']['nics'][0]['mtu'] = TestVector.JUMBO_FRAME 
                self.yaml_file['cuphydriver_config']['cells'][cell_index]['ru_type'] = 1 # O-RU type 1
                self.yaml_file['cuphydriver_config']['cells'][cell_index]['dl_iq_data_fmt']['bit_width'] = 9
                self.yaml_file['cuphydriver_config']['cells'][cell_index]['ul_iq_data_fmt']['bit_width'] = 9


            #Note - currently only nrSIM uses auto_controllerConfig.py
            # 
            #The below was an attempt to solidify defaults for perf here.  Leaving this code in case we revive
            # this path, but punting for now out of fear we will break nrSim.  I expect we will come back to this.
            #
            # ##UL/DL TIMING CONFIGURATION
            # # Please note this couples with auto_RuEmulatorConfig.py
            # #Note: In cuphycontroller yaml this config is set on a per cell basis.  As of 250130 we do not
            # # test this functionality, as RU does not have independent control as function of cell.  Nevertheless
            # # this is how our cuphycontroller yaml is current structured.

            # #DLU timing offset
            # self.yaml_file['cuphydriver_config']['cells'][cell_idx]["T1a_max_up_ns"] = 345000

            # #ULU transmit offset - the actual start time for ULU traffic
            # # Note: we rely on default value in corresponding platform cuphycontroller yamls
            # #       Perhaps in the future this will change.
            # # self.yaml_file['cuphydriver_config']['cells'][cell_idx]["ul_u_plane_tx_offset_ns"] = 280000

            # #ULU/SRS reception windows
            # self.yaml_file['cuphydriver_config']['cells'][cell_idx]["Ta4_min_ns"] = 50000
            # self.yaml_file['cuphydriver_config']['cells'][cell_idx]["Ta4_max_ns"] = 331000
            # self.yaml_file['cuphydriver_config']['cells'][cell_idx]["Ta4_min_ns_srs"] = 621000
            # self.yaml_file['cuphydriver_config']['cells'][cell_idx]["Ta4_max_ns_srs"] = 1831000

            # if int(self.yaml_file['cuphydriver_config']['mMIMO_enable']) == 1:
            #     #ULC timing offset
            #     self.yaml_file['cuphydriver_config']['cells'][cell_idx]["T1a_max_cp_ul_ns"] = 535000

            #     #DLC timing offset
            #     #TODO - DLC should be sent at T0-669us.  Need to patch here to make this so.
            #     #This will currently send DLC at T0-470us = T0-345us-125us
            #     self.yaml_file['cuphydriver_config']['cells'][cell_idx]["Tcp_adv_dl_ns"] = 125000

            # else:
            #     #ULC timing offset
            #     self.yaml_file['cuphydriver_config']['cells'][cell_idx]["T1a_max_cp_ul_ns"] = 336000

            #     #DLC timing offset
            #     #Note - DLC is sent at T0-470us = T0-345us-125us
            #     self.yaml_file['cuphydriver_config']['cells'][cell_idx]["Tcp_adv_dl_ns"] = 125000


            

            #mu
        if lp.special_l2adapter_config_file:
            # Specific l2 adapter config file generation is required
            self.yaml_file['l2adapter_filename'] = lp.special_l2adapter_config_file

        filename = os.path.join(self.outdir, f'{self.template_name}_{lp.name_str}{self.template_ext}')
        self.filename = filename
        with open(filename, 'w+') as stream:
            yaml.dump(self.yaml_file, stream, explicit_start=True, explicit_end=True, default_flow_style=None, width=float("inf"), sort_keys=False ,default_style='')
        
        self.lpcount += 1

class l2AdapterConfig:

    def __init__(self, template_file, outdir, l2_outfile=None):
        
        self.count = 0
        self.template_name, self.template_ext = os.path.splitext(os.path.basename(template_file))
        self.outdir = outdir
        self.file = l2_outfile

        with open(template_file, 'r') as template_content:
            self.yaml_file = yaml.safe_load(template_content)

    def get_cfg_filename(self, mu_val, tv_name=None):
        if self.file is not None:
            return os.path.join(self.outdir, self.file)
        if tv_name:
            return os.path.join(self.outdir, f'{self.template_name}_mu_{mu_val}_{tv_name.zfill(4)}{self.template_ext}')
        else:
            return os.path.join(self.outdir, f'{self.template_name}_mu_{mu_val}{self.template_ext}')
    
    def write(self, mu_val):
        file = self.get_cfg_filename(mu_val)
        
        self.yaml_file['mu_highest'] = mu_val
            
        with open(file, 'w+') as stream:
            yaml.dump(self.yaml_file, stream, explicit_start=True, explicit_end=True, default_flow_style=None, width=float("inf"), sort_keys=False, default_style='')

    def write_special(self, mu_val, tv_name, additional_attributes):

        file = self.get_cfg_filename(mu_val, tv_name)
        self.filename = file

        self.yaml_file['mu_highest'] = mu_val

        for key,attr_value in additional_attributes.items():
            self.yaml_file[key] = attr_value

        with open(file, 'w+') as stream:
            yaml.dump(self.yaml_file, stream, explicit_start=True, explicit_end=True, default_flow_style=None, width=float("inf"), sort_keys=False, default_style='')
            
        

def get_lp_name(file=None):
    f = os.path.basename(file)
    f = os.path.splitext(f)[0]
    components = f.split('_')
    if components[-1].isnumeric():
        if components[-2] in ['ULMIX', 'DLMIX']:
            return components[-2] + '_' + components[-1]
        else:
            return int(components[-1])
    else:
        return components[-2] + '_' + components[-1]

def generate_controller_config_files(input_dir, output_dir, template_file, all_cells=False, test_case=None):
    launch_pattern = None
    tv_specific_l2adapter_files = {}
    lp_specific_l2adapter_files = {}
    cuphyConfig = CuphyConfiguration(template_file, output_dir)
    cuphyConfig.get_fresh_template()
    l2_template_file = os.path.join(os.path.dirname(template_file), cuphyConfig.yaml_file['l2adapter_filename'])

    with open(os.path.join(output_dir, 'out.txt'), "w") as log_file:

        # Launch pattern
        lpatterns = glob.glob(os.path.join(input_dir, 'multi-cell/launch_pattern_nrSim*.yaml'))
        print(f"Using {l2_template_file} as template for l2 adapter config")
        for lp in lpatterns:
            if test_case is None or test_case in lp:
                print("Trying launch pattern: "+lp)
                try:
                    launch_pattern = LaunchPattern()
                    launch_pattern.parse_launch_pattern_file(lp)
                    cuphyConfig.write_lp_config(launch_pattern)
                    print(f'Wrote {cuphyConfig.filename}')
                    log_file.write(f'{lp}\n')
                    if launch_pattern.special_l2adapter_config_file:
                        lp_specific_l2adapter_files[launch_pattern.special_l2adapter_config_file] = (max(launch_pattern.mus), str(launch_pattern.name).zfill(4), launch_pattern.l2adapter_attributes)
                except KeyboardInterrupt:
                    raise
                except Exception as x:
                    print(f'Exception raised for LP {lp}:', x)
                    traceback.print_exc()
                    continue


                # l2_adapter
                l2AdapterCfg = l2AdapterConfig(l2_template_file, output_dir, cuphyConfig.yaml_file['l2adapter_filename'])

                # # # write TV specific l2 adapter files
                # for l2adapter_filename,attributes_tuple in tv_specific_l2adapter_files.items():
                #     mu_val, tv_name, attributes_dict = attributes_tuple
                #     try:
                #         l2AdapterCfg.write_special(mu_val, tv_name, attributes_dict)
                #     except KeyboardInterrupt:
                #         raise
                #     except Exception as x:
                #         print(f'Exception raised for TV {testvector}:', x)
                #         continue

                # # write Launch pattern specific l2 adapter files
                for l2adapter_filename,attributes_tuple in lp_specific_l2adapter_files.items():
                    mu_val, lp_name, attributes_dict = attributes_tuple
                    if test_case is None or test_case in lp_name:
                        try:
                            l2AdapterCfg.write_special(mu_val, lp_name, attributes_dict)
                            print(f'Wrote {l2AdapterCfg.filename}')
                        except KeyboardInterrupt:
                            raise
                        except Exception as x:
                            print(f'Exception raised for LP {lp_name}:', x)
                            continue

                lp_specific_l2adapter_files.clear()
                # # l2 adapter files for mu range 0,5
                # for mu_val in range(0,6):
                #     try:
                #         l2AdapterCfg.write(mu_val)

                #     except KeyboardInterrupt:
                #         raise
                #     except Exception as x:
                #         print(f'Exception raised for TV {testvector}"', x)
                #         continue


    print(f'Total Launch patterns parsed {cuphyConfig.lpcount}')
    return launch_pattern

if __name__ == "__main__":

    script_dir = os.path.realpath(os.path.dirname(__file__))
    cuBB_SDK=os.environ.get("cuBB_SDK", os.path.normpath(os.path.join(script_dir, '../..')))
    CUBB_HOME=os.environ.get("CUBB_HOME", cuBB_SDK)

    #CONFIG_FILE_TEMPLATE = os.path.join(cuBB_SDK, 'cuPHY-CP/cuphycontroller/config/cuphycontroller_F08.yaml')
    CONFIG_FILE_TEMPLATE = os.path.join(cuBB_SDK, 'cuPHY-CP/cuphycontroller/config/cuphycontroller_nrSim_SCF.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help='Input Directory of Test Vectors')
    parser.add_argument("-o", "--output_dir", default=None, help='Output Directory for Configurations')
    parser.add_argument("-t", '--template_file', default=CONFIG_FILE_TEMPLATE , help='Config File Template')
    parser.add_argument("-a", '--all_cells', action='store_true', help='If used, all the cells configuration will be modified otherwise only the first cell configuration in modified')
    parser.add_argument("-c", '--test_case', help='Specify an nrSim testcase to generate config files for, otherwise generate for all launch_pattern_xyx.yaml files')
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir 

    launch_pattern = generate_controller_config_files(args.input_dir, output_dir, args.template_file, args.all_cells, args.test_case)
