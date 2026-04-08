#!/usr/bin/python3

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

import glob
import argparse
import os
import re
import yaml
import h5py as h5

# This HexInt and representer code is so that we can dump hex values w/o
# formatting them as strings - i.e. name: 0xf instead of name: '0xf'
class HexInt(int): pass
def representer(dumper, data):
    return yaml.ScalarNode('tag:yaml.org,2002:int', hex(data))
yaml.add_representer(HexInt, representer)

def create_string(list_of_strings):
  """Creates a string by appending all elements of a list of strings."""
  string = ""
  for item in list_of_strings:
    string += item
  return string

def generate_testmac_config_file(output_file_path, template_file, test_case, platform, copy_template, test_slots,input_dir):
    with open(template_file, 'r') as template_content:
        test_mac_yaml_file = yaml.safe_load(template_content)
    
        if platform == 'R750':
            test_mac_yaml_file['recv_thread_config']['cpu_affinity'] = 33
            test_mac_yaml_file['sched_thread_config']['cpu_affinity'] = 27
        elif platform == 'devkit':
            test_mac_yaml_file['recv_thread_config']['cpu_affinity'] = 21
            test_mac_yaml_file['sched_thread_config']['cpu_affinity'] = 16
        elif platform == 'bf3-arm':
            test_mac_yaml_file['low_priority_core'] = 1
            test_mac_yaml_file['recv_thread_config']['cpu_affinity'] = 13
            test_mac_yaml_file['sched_thread_config']['cpu_affinity'] = 12
        elif platform == 'CG1':
            test_mac_yaml_file['recv_thread_config']['cpu_affinity'] = 21
            test_mac_yaml_file['sched_thread_config']['cpu_affinity'] = 16

        # Set test_slots for all tests - test-mac to stop running after X slots
        test_mac_yaml_file['test_slots'] = test_slots

        # GT-6121 - Enable this for all tests
        test_mac_yaml_file['data']['rsrpMeasurement'] = 1
        test_mac_yaml_file['data']['pnMeasurement'] = 1
        test_mac_yaml_file['data']['pf_234_interference'] = 1
        test_mac_yaml_file['data']['prach_interference'] = 1
        num_cells = 1
        gNB_FAPI_testvectors = []
        if test_case.isdigit():  # NRSIM tests are all integers
            test_mac_yaml_file['validate_log_opt'] = 2
            test_mac_yaml_file['validate_enable'] = 2
            test_mac_yaml_file["indicationPerSlot"]["uciIndPerSlot"] = 2
            if test_case == "90023":
                test_mac_yaml_file['enable_dummy_tti'] = 1
            if test_case == "90624":
                test_mac_yaml_file["indicationPerSlot"]["uciIndPerSlot"] = 0
            if test_case == "90629": # temporarily for Rel-25-2
                test_mac_yaml_file["indicationPerSlot"]["uciIndPerSlot"] = 0
            if test_case in ("90628", "90629", "90636"):
                test_mac_yaml_file['builder_thread_enable'] = 1
                test_mac_yaml_file['schedule_total_time'] = 450000
                test_mac_yaml_file['fapi_delay_bit_mask'] = HexInt(255)
            if test_case[:2] == '90':
                yaml_file = None
                lpname = "launch_pattern_nrSim_" + test_case +".yaml"
                lp_arr = []
                lp_arr.append(input_dir)
                if os.path.exists(os.path.join(input_dir, 'multi-cell')):
                    lp_arr.append('/multi-cell')
                lp_dir =  create_string(lp_arr)
                lp = os.path.join(lp_dir, lpname)
                with open(lp, 'r') as template_content:
                    yaml_file = yaml.safe_load(template_content)
                num_cells = len(yaml_file["Cell_Configs"])
                celltvs = yaml_file['SCHED']
                for slotindex in range(len(celltvs)):
                    cell_configs = celltvs[slotindex]['config']
                    for config_index in range(len(cell_configs)):
                        tvnames = cell_configs[config_index]['channels']
                        if len(tvnames) == 0:
                            continue
                        for tvindex in range(len(tvnames)):
                            gNB_FAPI_testvectors.extend(glob.glob( os.path.join(input_dir, tvnames[tvindex])))
                if test_case in ["90159", "90160"]:
                    test_mac_yaml_file['data']['pusch_aggr_factor'] = 2
                else:
                    test_mac_yaml_file['data']['pusch_aggr_factor'] = 1 
            else:
                tv_name = "TVnr_" + test_case + "_gNB_FAPI_*.h5"
                gNB_FAPI_testvectors.extend(glob.glob(os.path.join(input_dir, tv_name)))
                
        elif test_case.startswith('F08'):
            # E.g. F08_F_10C_41_BFP9_CCDF_STT450000 -> schedule_total_time set to 450000
            # E.g. F08_F_10C_41_BFP9_CCDF_STT       -> use the list of values
            if "STT" in test_case:
                test_mac_yaml_file['builder_thread_enable'] = 1
                test_mac_yaml_file['fapi_delay_bit_mask'] = HexInt(255)
                # 35 for R750 - 22 for devkit
                if platform == 'R750':
                    test_mac_yaml_file['builder_thread_config']['cpu_affinity'] = 35
                elif platform == 'devkit':
                    test_mac_yaml_file['builder_thread_config']['cpu_affinity'] = 22
                elif platform == 'bf3-arm':
                    test_mac_yaml_file['builder_thread_config']['cpu_affinity'] = 14
                elif platform == 'CG1':
                    test_mac_yaml_file['builder_thread_config']['cpu_affinity'] = 20
                    
                m = re.search('STT(?P<stt>[0-9]*)', test_case)
                g = m.groupdict()
                if g['stt'] == "":
                    test_mac_yaml_file['schedule_total_time'] = [225000, 225000, 450000, 450000, 450000, 450000, 450000, 225000, 225000, 225000]
                else:
                    test_mac_yaml_file['schedule_total_time'] = int(g['stt'])

            if "restart" in test_case:
                test_mac_yaml_file['cell_run_slots'] = 10000
                test_mac_yaml_file['cell_stop_slots'] = 10000
            yaml_file = None
            test_substrings = test_case.split("_")

            if "SWDISABLEEH" in test_case:
                test_mac_yaml_file["indicationPerSlot"]["uciIndPerSlot"] = 0
            elif "EH" in test_case:
                test_mac_yaml_file["indicationPerSlot"]["uciIndPerSlot"] = 2

            lpname = "launch_pattern_" + test_substrings[0] + "_" + test_substrings[2] + "_" + test_substrings[3] + ".yaml"
            lp_arr = []
            lp_arr.append(input_dir)
            if os.path.exists(os.path.join(input_dir, 'multi-cell')):
                lp_arr.append('/multi-cell')
            lp_dir =  create_string(lp_arr)
            lp = os.path.join(lp_dir, lpname)

            with open(lp, 'r') as template_content:
                yaml_file = yaml.safe_load(template_content)
            num_cells = len(yaml_file["Cell_Configs"])
            tvnames = yaml_file["Cell_Configs"]
            for tvindex in range(len(tvnames)):
                gNB_FAPI_testvectors.extend(glob.glob( os.path.join(input_dir, tvnames[tvindex])))

        for testvector in gNB_FAPI_testvectors:
                parsed_testvector = h5.File(testvector, 'r')

                if 'enable_dynamic_BF' in parsed_testvector['Cell_Config'].dtype.fields:
                    if parsed_testvector['Cell_Config']['enable_dynamic_BF'][0] == 1 :
                        test_mac_yaml_file["indicationPerSlot"]["srsIndPerSlot"] = 2 
                else:
                    print('No field named enable_dynamic_BF in Cell_Config')

                if 'enable_static_dynamic_beamforming' in parsed_testvector['Cell_Config'].dtype.fields:
                    if parsed_testvector['Cell_Config']['enable_static_dynamic_beamforming'][0] == 1 :
                        test_mac_yaml_file["indicationPerSlot"]["srsIndPerSlot"] = 2
                else:
                    print('No field named enable_static_dynamic_beamforming in Cell_Config')

        if output_file_path is None:
            template_dir = os.path.realpath(os.path.dirname(template_file))
            print(template_dir)
            output_filename = f'test_mac_config_{test_case}.yaml'
            output_file_path = os.path.join(template_dir, output_filename)

        with open(output_file_path, 'w+') as stream:
            yaml.dump(
                test_mac_yaml_file, 
                stream,
                explicit_start=True,
                explicit_end=True,
                default_flow_style=False,
                sort_keys=False,
                default_style='',
                width=1000,
            )

    if copy_template:
        with open(template_file, 'r') as template_content:
            original = yaml.safe_load(template_content)
            output_file_path_template_copy = f'{output_file_path}_template'
            with open(output_file_path_template_copy, 'w+') as stream:
                yaml.dump(
                    original,
                    stream,
                    explicit_start=True,
                    explicit_end=True,
                    default_flow_style=False,
                    sort_keys=False,
                    default_style='',
                    width=1000,
                )


def print_descriptions():
    print('''********* TEST CASE NAMING CONVENTION DESCRIPTIONS **********
This output will describe the different test_case names and formats that will be supported.

NRSIM tests:
    These are all integer values describing the specific NRSIM test vector (TV) to be tested.
    0123, 0124 <- zero padded for values between 1 and 999 is supported
    123, 124   <- non-zero padded is also supported
    1001, 1002, 1003, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000
    90000

F08 tests (examples):
    F08_F_1C_41_BFP9
    F08_F_1C_44_BFP9
    F08_F_7C_44_BFP9
    F08_F_7C_44_BFP14
    F08_F_10C_41_BFP14
    F08_F_10C_41_BFP9_CCDF
    F08_F_10C_41_BFP9_CCDF_STT
    F08_F_10C_41_BFP9_CCDF_STT450000
    F08_F_10C_41_BFP9_CCDF_STT450000_restart
    F08_F_10C_41_BFP9_CCDF_STT450000_restart_EH
        These tests all start with "F08".
        The next value is either one of [0,A,B,C,D,E,F] describing which channels to use. Currently not used in this script.
        NC describes the number of cells to use, where N is the number of cells.
        The next value describes the pattern to use.
        BFP9 vs BFP14 describes whether to use compression at 9 or 14 bits.
        CCDF describes what level of logging is required.
        STT describes what the setting for scheduled_total_time will be.
        restart is a type of test that causes cells to turn on and off - or restart - automatically.
        EH to enable early HARQ feature.
        Note that the order of these flags is not important. It must start with F08, but BFP, CCDF, STT, etc can be in any order.

F13 tests:
    F13
''')

if __name__ == '__main__':
    script_dir = os.path.realpath(os.path.dirname(__file__))
    cuBB_SDK=os.environ.get('cuBB_SDK', os.path.normpath(os.path.join(script_dir, '../..')))
    CUBB_HOME=os.environ.get('CUBB_HOME', cuBB_SDK)

    TEST_MAC_CONFIG_FILE_TEMPLATE = os.path.join(cuBB_SDK, 'cuPHY-CP/testMAC/testMAC/test_mac_config.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file_path', default=None , help='Output filename and path for the new test mac configurations. If not set, uses directory of template file and test_mac_config_{test_case}.yaml')
    parser.add_argument('-t', '--test_mac_template_file', default=TEST_MAC_CONFIG_FILE_TEMPLATE, help='test mac Config File Template')
    parser.add_argument('-c', '--test_case', help='Test case being run - e.g. 1001, F08_*, F13_*, etc.')
    parser.add_argument('-p', '--platform', help='Platform where testmac is running - R750, devkit, bf3-arm')
    parser.add_argument('-d', '--description', default=False, action='store_true', help='Print useful description of the different test case naming conventions that are supported.')
    parser.add_argument('-x', '--copy_template', default=False, action='store_true', help='Copies template in same format as output file for easier diffs. Same name as -o but with _template appended.')
    parser.add_argument('-s', '--test_slots', default=0, type=int, help='Number of slots to run the test. Set to 0 to run indefinitely (default).')
    parser.add_argument("-i", "--input_dir", default="../../testVectors/", help='Input Directory of Test Vectors')
    args = parser.parse_args()

    if args.description:
        print_descriptions()
    else:
        generate_testmac_config_file(args.output_file_path, args.test_mac_template_file, args.test_case, args.platform, args.copy_template, args.test_slots, args.input_dir)
