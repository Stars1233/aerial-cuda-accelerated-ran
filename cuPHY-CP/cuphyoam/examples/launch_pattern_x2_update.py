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
import glob
import json
import numpy as np
import os
import re
import sys
import yaml

yaml.Dumper.ignore_aliases = lambda *args : True

def launch_pattern_x2_update_all(input_dir, output_dir):
    count = 0
    # launch pattern files
    launch_patterns = glob.glob(os.path.join(input_dir, 'launch_pattern_*.yaml'))
    for pattern in launch_patterns:
        try:
            with open(pattern, 'r') as pattern_content:
                pattern_file = yaml.safe_load(pattern_content)

                numslots = len(pattern_file['SCHED'])
                for slot in range(numslots):
                    numcells = len(pattern_file['SCHED'][slot]['config'])
                    for cell_index in range(numcells):
                        pattern_file['SCHED'][slot]['config'].append({'cell_index': float(cell_index + numcells), 'channels': pattern_file['SCHED'] [slot]['config'][cell_index]['channels'], 'type': pattern_file['SCHED'][slot]['config'][cell_index]['type']})
                for cell_index in range(numcells):
                    pattern_file['Cell_Configs'].append(
                        pattern_file['Cell_Configs'][cell_index])

                filename = os.path.basename(pattern).replace('.yaml','_X2.yaml')
                with open(os.path.join(output_dir, filename), 'w+') as stream:
                    yaml.dump(pattern_file, stream, explicit_start=True, explicit_end=True, default_flow_style= None, sort_keys= False ,default_style='')
                count += 1

        except KeyboardInterrupt:
            raise
        except:
            print('Exception raised for pattern_file')
            continue

    print('Total RuEmulator config files generated ' + str(count))

def launch_pattern_x2_update(launch_pattern_file, output_dir):
    try:
        with open(launch_pattern_file, 'r') as pattern_content:
            pattern_file = yaml.safe_load(pattern_content)

            numslots = len(pattern_file['SCHED'])
            for slot in range(numslots):
                numcells = len(pattern_file['SCHED'][slot]['config'])
                for cell_index in range(numcells):
                    pattern_file['SCHED'][slot]['config'].append({'cell_index': float(cell_index + numcells), 'channels': pattern_file['SCHED'] [slot]['config'][cell_index]['channels'], 'type': pattern_file['SCHED'][slot]['config'][cell_index]['type']})
            for cell_index in range(numcells):
                pattern_file['Cell_Configs'].append(
                    pattern_file['Cell_Configs'][cell_index])

            filename = os.path.basename(launch_pattern_file).replace('.yaml','_X2.yaml')
            with open(os.path.join(output_dir, filename), 'w+') as stream:
                yaml.dump(pattern_file, stream, explicit_start=True, explicit_end=True, default_flow_style= None, sort_keys= False ,default_style='')

    except KeyboardInterrupt:
        raise
    except:
        print(f'Exception raised for pattern_file: {launch_pattern_file}')


if __name__ == "__main__":

    script_dir = os.path.realpath(os.path.dirname(__file__))
    cuBB_SDK=os.environ.get("cuBB_SDK", os.path.normpath(os.path.join(script_dir, '../..')))
    CUBB_HOME=os.environ.get("CUBB_HOME", cuBB_SDK)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", '--launch_pattern_file', default=None , help='Origin launch pattern file', required=True)
    parser.add_argument("-o", '--output_dir', default=None , help='Output Directory for the new X2 launch pattern files', required=True)
    #parser.add_argument("-i", "--input_dir", help='Input Directory of launch pattern files')

    args = parser.parse_args()
    # process template file argument

    #if not os.path.isdir(args.input_dir):
    #    exit(f"InvalidPathError: The path {args.input_dir} does not exist \nExiting...")
    if not os.path.isdir(args.output_dir):
        exit(f"InvalidPathError: The path {args.output_dir} does not exist \nExiting...")

    launch_pattern_x2_update(args.launch_pattern_file, args.output_dir)
