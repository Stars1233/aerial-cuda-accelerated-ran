#!/usr/bin/env python3

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

import os
import argparse
from auto_controllerConfig import generate_controller_config_files
from auto_RuEmulatorConfig import generate_ru_config_files
from auto_TestMacConfig import generate_testmac_config_file


####
# Generate configs assuming that you would run the following:
# auto_controllerConfig.py -i $cuBB_SDK/testVectors/ -t $cuBB_SDK/cuPHY-CP/cuphycontroller/config/cuphycontroller_nrSim_SCF_CG1.yaml -o $cuBB_SDK/cuPHY-CP/cuphycontroller/config
# auto_RuEmulatorConfig.py -i $cuBB_SDK/cuPHY-CP/cuphycontroller/config -t $cuBB_SDK/cuPHY-CP/ru-emulator/config/config.yaml -o $cuBB_SDK/cuPHY-CP/ru-emulator/config
# auto_TestMacConfig.py -t $cuBB_SDK/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml -c <testcase no.> -p CG1 -o $cuBB_SDK/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml -x -s 60000 -i $cuBB_SDK/testVectors

####

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--test_case', help='Test case being run - e.g. 1001, F08_*, F13_*, etc.')
    parser.add_argument('-b', '--base_dir', help='Assume that $cuBB_SDK is read-only, and configs are somewhere else')
    parser.add_argument('-t', '--template_file', help='Template file to be used for cuphycontroller. Default is cuphycontroller_nrSim_SCF_CG1.yaml')
    parser.add_argument('-p', '--platform', help='Defaults to CG1, but could be something else')
    args = parser.parse_args()

    script_dir = os.path.realpath(os.path.dirname(__file__))
    cuBB_SDK=os.environ.get("cuBB_SDK", os.path.normpath(os.path.join(script_dir, '../..')))

    cuphyDir=cuBB_SDK+'/cuPHY-CP/cuphycontroller/config/'
    ruTemplate=cuBB_SDK+'/cuPHY-CP/ru-emulator/config/config.yaml'
    testmacTemplate=cuBB_SDK+'/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml'

    if args.base_dir is None:
        base_dir=cuBB_SDK
    else:
        base_dir=args.base_dir

    testVectorDir=base_dir+'/testVectors/'
    cuphyOutputDir=base_dir+'/cuPHY-CP/cuphycontroller/config'
    ruOutputDir=base_dir+'/cuPHY-CP/ru-emulator/config'
    testmacOutput=base_dir+'/cuPHY-CP/testMAC/testMAC/test_mac_config.yaml'

    if args.template_file is None:
        cuphyTemplate=cuphyDir+'cuphycontroller_nrSim_SCF_CG1.yaml'
    else:
        cuphyTemplate=cuphyDir+args.template_file

    if args.platform is None:
        testPlatform='CG1'
    else:
        testPlatform=args.platform


    print("Generating cuphycontroller configs")
    launchPattern = generate_controller_config_files(testVectorDir, cuphyOutputDir, cuphyTemplate, False, args.test_case)
    if launchPattern is None:
        print("No launch patterns generated. Please check that there are testvectors for test case: " + args.test_case)
        exit(1)
    print("Generating ru emulator configs")
    generate_ru_config_files(cuphyOutputDir, ruOutputDir, ruTemplate, args.test_case)
    if args.test_case is None:
        print("Generating testMac configs for nrSim testcase: " + launchPattern.name_str + " based on last generated cuphycontroller yaml.")
        print("   If this is not expected, specify a case using the -c option")
    else:
        print("Generating testMac configs for nrSim testcase: " + launchPattern.name_str);

    generate_testmac_config_file(testmacOutput, testmacTemplate, launchPattern.name_str, testPlatform, True, 60000, testVectorDir)
