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
import glob
import os 
import shutil

CUBB_SDK_ENV_VAR = "cuBB_SDK"

ARTIFACTS_LIST = [
    "build/cuPHY-CP/aerial-fh-driver/app/fh_generator/fh_generator",
    "build/cuPHY-CP/aerial-fh-driver/libaerial-fh.so",
    "build/cuPHY-CP/aerial-fh-driver/test/ut",
    "build/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf",
    "build/cuPHY-CP/cuphydriver/src/libcuphydriver.so",
    "build/cuPHY-CP/cuphyoam",
    "build/cuPHY-CP/cuphyoam/libcuphyoamlib.so",
    "build/cuPHY-CP/gt_common_libs/aerial_metrics/src/libaerial_metrics.so",
    "build/cuPHY-CP/gt_common_libs/nvIPC/libnvipc.so",
    "build/cuPHY-CP/gt_common_libs/nvlog/libnvlog.so",
    "build/cuPHY-CP/gt_common_libs/nvlog/nvlog_collect",
    "build/cuPHY-CP/ru-emulator/ru_emulator/ru_emulator",
    "build/cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/l2_adapter_cuphycontroller_scf",
    "build/cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/tick_unit_test",
    "build/cuPHY-CP/testMAC/testMAC/test_mac",
    "build/cuPHY/src/cuphy/libcuphy.so",
    "cuPHY-CP/aerial-fh-driver/app/fh_generator/config/simple",
    "cuPHY-CP/aerial-fh-driver/scripts/phy_latencies/phy_latencies.py",
    "cuPHY-CP/aerial-fh-driver/scripts/rx_pcap_capture/rx_pcap_capture.py",
    "cuPHY-CP/cuphycontroller/config",
    "cuPHY-CP/gt_common_libs/nvlog/config",
    "cuPHY-CP/ru-emulator/config",
    "cuPHY-CP/ru-emulator/scripts",
    "cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/cuphycontroller_run_phy_local_test.yaml",
    "cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/cuphycontroller.yaml",
    "cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/tick_poll_mode.yaml",
    "cuPHY-CP/scfl2adapter/scf_app/cuphycontroller/tick_sleep_mode.yaml",
    "cuPHY-CP/testMAC/scripts",
    "cuPHY-CP/testMAC/testMAC/test_mac_config.yaml",
    "cuPHY/util/cuBB_system_checks/cuBB_system_checks.py",
]

F08_TVS = [
    'testVectors/demo_msg2_4_ant_pdcch_dl_gNB_FAPI_s7.h5',
    'testVectors/demo_msg4_4_ant_pdcch_ul_gNB_FAPI_s7.h5',
    'testVectors/multi-cell/launch_pattern_F08_1C.yaml',
    'testVectors/multi-cell/launch_pattern_F08_2C.yaml',
    'testVectors/multi-cell/launch_pattern_F08_3C.yaml',
    'testVectors/multi-cell/launch_pattern_F08_4C.yaml',
    'testVectors/multi-cell/testMac_config_params_F08.h5',
    'testVectors/testMac_DL_ctrl-TC2002_pdcch.h5',
    'testVectors/testMac_DL_ctrl-TC2006_pdcch.h5',
    'testVectors/TV_cuphy_V14-DS-08_slot0_MIMO2x16_PRB82_DataSyms10_qam64.h5',
    'testVectors/TVnr_1902_gNB_FAPI_s0.h5',
    'testVectors/TVnr_1902_SSB_gNB_CUPHY_s0p0.h5',
    'testVectors/TVnr_5901_PRACH_gNB_CUPHY_s1p0.h5',
    'testVectors/TVnr_CP_F08_DS_01.1_126_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F08_DS_01.1_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F08_DS_01.1_PDSCH_gNB_CUPHY_s5p0.h5',
    'testVectors/TVnr_CP_F08_DS_01.2_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F08_US_01.1_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F08_US_01.1_PUSCH_gNB_CUPHY_s5p0.h5',
    'testVectors/WFreq.h5',
]

F13_TVS = [
    'testVectors/TVnr_CP_F14_US_33_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F14_US_01.yaml',
    'testVectors/TVnr_CP_F14_DS_01.yaml',
    'testVectors/TVnr_CP_F14_US_34_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F14_US_33.yaml',
    'testVectors/TV_cuphy_F14-DC-02_PRB273.h5',
    'testVectors/TV_cuphy_F14-RA-02.h5',
    'testVectors/TV_fapi_F14-UC-02_PRB273.h5',
    'testVectors/TVnr_CP_F14_US_33_PUSCH_gNB_CUPHY_s5p1.h5',
    'testVectors/TV_cuphy_F14-RA-01.h5',
    'testVectors/TV_fapi_F14-US-31_snrdb40.00_MIMO2x16_PRB40_DataSyms12_qam64.h5',
    'testVectors/TV_fapi_F14-DS-31_slot0_MIMO4x16_PRB40_DataSyms10_qam64.h5',
    'testVectors/TVnr_CP_F14_US_01_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F14_DS_34_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F14_DS_35.yaml',
    'testVectors/TVnr_CP_F14_US_35_PUSCH_gNB_CUPHY_s5p0.h5',
    'testVectors/TVnr_CP_F14_DS_35_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F14_DS_33_PDSCH_gNB_CUPHY_s5p3.h5',
    'testVectors/TVnr_CP_F14_DS_01_gNB_FAPI_s5.h5',
    'testVectors/TVnr_CP_F14_US_35_gNB_FAPI_s5.h5',
    'testVectors/TV_cuphy_F14-UC-02_PRB273.h5',
    'testVectors/TVnr_CP_F14_US_34.yaml',
    'testVectors/TVnr_CP_F14_US_01_PUSCH_gNB_CUPHY_s5p1.h5',
    'testVectors/TVnr_CP_F14_US_35.yaml',
    'testVectors/TVnr_CP_F14_DS_34_PDSCH_gNB_CUPHY_s5p1.h5',
    'testVectors/TVnr_CP_F14_DS_35_PDSCH_gNB_CUPHY_s5p1.h5',
    'testVectors/TV_cuphy_F14-UC-01_PRB273.h5',
    'testVectors/TVnr_CP_F14_US_34_PUSCH_gNB_CUPHY_s5p1.h5',
    'testVectors/TV_fapi_F14-UC-01_PRB273.h5',
    'testVectors/TVnr_CP_F14_DS_33.yaml',
    'testVectors/TV_fapi_F14-RA-01_PRB273.h5',
    'testVectors/TV_fapi_F14-DC-01_PRB273.h5',
    'testVectors/multi-cell/testMac_config_params_F14.h5',
    'testVectors/TV_fapi_F14-RA-02_PRB273.h5',
    'testVectors/TVnr_CP_F14_DS_34.yaml',
    'testVectors/TVnr_CP_F14_DS_33_gNB_FAPI_s5.h5',
    'testVectors/TV_cuphy_F14-DC-01_PRB273.h5',
    'testVectors/TV_fapi_F14-DC-02_PRB273.h5',
    'testVectors/TVnr_CP_F14_DS_01_PDSCH_gNB_CUPHY_s5p3.h5',
    'testVectors/multi-cell/testMac_config_params_F13.h5',
    'testVectors/multi-cell/launch_pattern_F13_3C.yaml',
    'testVectors/multi-cell/testMac_config_params_F13.h5',
    'testVectors/multi-cell/launch_pattern_F13_2C.yaml',
    'testVectors/multi-cell/launch_pattern_F13_4C.yaml',
    'testVectors/multi-cell/launch_pattern_F13_1C.yaml',
    'testVectors/multi-cell/launch_pattern_F13_1C_31.yaml',
]


def collect_cubb_artifacts(destination, cuBB_path):
    for artifact in ARTIFACTS_LIST:
        try:
            src = os.path.join(cuBB_path, artifact)
            dst = os.path.join(args.destination, artifact)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
                shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)

        except shutil.Error:
            pass 

        except Exception as e:
            print(e)


def add_tvs_to_artifacts(testVectors):
    if 'F08' in testVectors:
        ARTIFACTS_LIST.extend(F08_TVS)

    if 'F13' in testVectors:
        ARTIFACTS_LIST.extend(F13_TVS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="collect_cuBB_artifacts.py", description="Collect cuBB executables, libs, scripts and TVs into a separate directory")
    parser.add_argument("destination", help="destination directory", type=str)
    parser.add_argument("-d", "--dpdk", help="DPDK libs", action='store_true')
    parser.add_argument("-f", "--fh_generator", help="FH traffic generator config files", action='store_true')
    parser.add_argument("-t", "--testVectors", help="TVs and launch patterns", action="extend", nargs="+", type=str, choices=['F08', 'F13'])
    args = parser.parse_args()

    if args.dpdk:
        ARTIFACTS_LIST.append("gpu-dpdk/build/install/lib")

    if args.testVectors:
        add_tvs_to_artifacts(args.testVectors)

    if args.fh_generator:
        ARTIFACTS_LIST.append("cuPHY-CP/aerial-fh-driver/app/fh_generator/config")

    cuBB_path = os.getenv(CUBB_SDK_ENV_VAR)
    if not cuBB_path:
        raise Exception("{} environment variable is not defined!".format(CUBB_SDK_ENV_VAR))

    collect_cubb_artifacts(args.destination, cuBB_path)
    print("cuBB artifacts copied to: " + args.destination)
