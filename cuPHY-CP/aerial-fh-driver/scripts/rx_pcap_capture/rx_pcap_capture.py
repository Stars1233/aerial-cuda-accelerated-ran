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

CUBB_SDK_ENV_VAR = 'cuBB_SDK'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='rx_pcap_capture.py', description="Capture RX traffic from a running aerial_fh instance")
    parser.add_argument("-n", "--nic", help="NIC device name to capture on", type=str, required=True)
    parser.add_argument("-p", "--pcap", help="output PCAP filepath", type=str, required=True)
    parser.add_argument("-c", "--core", help="CPU core to use", type=int, default=0)
    parser.add_argument("-f", "--file-prefix", help="Primary DPDK process --file-prefix", type=str, default="cuphycontroller")

    args = parser.parse_args()

    cubb_path = os.getenv(CUBB_SDK_ENV_VAR)

    if not cubb_path:
        raise Exception("{} env var is not defined!".format(CUBB_SDK_ENV_VAR))

    command = "{}/gpu-dpdk/build/app/dpdk-pdump --file-prefix={} -a {} -l {} -- --pdump 'port=0,queue=0,rx-dev={}'".format(cubb_path, args.file_prefix, args.nic, args.core, args.pcap)
    print("Running:", command)

    if (os.system(command) == 256):
        command = "{}/gpu-dpdk/build/app/dpdk-pdump --file-prefix={} -a {} -l {} -- --pdump 'port=1,queue=0,rx-dev={}'".format(cubb_path, args.file_prefix, args.nic, args.core, args.pcap)
        print("Re-running previous command now with Port ID 1:", command)
        os.system(command)
