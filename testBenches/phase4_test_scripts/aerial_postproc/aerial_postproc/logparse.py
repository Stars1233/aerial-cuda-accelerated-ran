#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import re
import sys
import os

import pandas as pd
from aerial_postproc.timeconv import sfn_to_tai, tai_to_sfn, oran_fn_to_tai
from aerial_postproc.parsenator import Parsenator

from functools import partial

NUM_20SLOT_GROUPS = 4

CICD_SLOT_FORMAT = "%2i"
CICD_FLOAT_FORMAT = "%9.3f"
CICD_NONE_FORMAT = "     None"

#Example: F08_6C_87_MODCOMP_STT480000_EH_1P -> 6, 87, 0, True, True, False, False
#Format: F08_<cells>C_<pattern>_<BFP#|MODCOMP>_..._EH/SWDISABLEEH/GC_<ports>P
def parse_tc_info(tc_name):
    parts = tc_name.split("_")
    cell_count = int(parts[1].replace("C",""))
    pattern = parts[2]

    if(tc_name.find("_BFP") >= 0):
        bfp = int(parts[3].replace("BFP",""))
        modcomp = False
    else:
        bfp = 0
        modcomp = True

    eh = tc_name.find("_EH") >= 0
    swdisableeh = tc_name.find("_SWDISABLEEH") >= 0
    gc = tc_name.find("_GC") >= 0
    return cell_count,pattern,bfp,modcomp,eh,swdisableeh,gc

def timestamp2datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp / 1e9)

from enum import Enum, EnumMeta
class MetaEnum(EnumMeta):
    def __contains__(cls, item) -> bool:
        try:
            cls(item)
        except ValueError:
            return False
        return True

class Direction(Enum, metaclass=MetaEnum):
    UPLINK = 1     # RU -> DU
    DOWNLINK = 2   # DU -> RU

# The Plane enumeration refers to the Front Haul plane semantics; it is irrelevant to the 5G/NR PHY channels
class Plane(Enum, metaclass=MetaEnum):
    CONTROL = 1 
    USER = 2

class Statistic(Enum, metaclass=MetaEnum):
    AVERAGE = 1
    MINIMUM = 2

#Takes a list of parsers and results list of data frames
def easy_parse(input_log,parser_list,max_lines=None):
    print("Running easy_parse...")

    result_list = [[] for parser in parser_list]

    fid = open(input_log)
    line = fid.readline()
    line_idx = 0
    while(line != ""):

        for ii,parser in enumerate(parser_list):
            result_list[ii].extend(parser(line))

        #Enforce max lines if enabled
        line_idx += 1
        if(max_lines and line_idx >= max_lines):
            break

        #Print progress
        if(line_idx % 100000 == 0):
            print("Processed line %i of %s"%(line_idx,max_lines))

        #Move to next line
        line = fid.readline()

    return [pd.DataFrame(aa) for aa in result_list]

# There are four formats that need to be parsed

# For 'PDSCH Aggr', 'PUCCH Aggr', 'PRACH Aggr'
gpusetuprun_pattern1 = re.compile(
    r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV.MAP_.?L\] \[PHYDRV\] SFN (\d+)\.(\d+) (.*) .{16,} Map (\d+) times ===> (?:\[PDSCH H2D Copy\] { DURATION: (\d+.\d+) us } )?\[CPU CUDA Setup\] { START: (\d+) END: (\d+).*\[CPU CUDA Run\] { START: (\d+) END: (\d+).*\[GPU Setup\] { DURATION: (\d+.\d+) us }.*\[GPU Run] { DURATION: (\d+.\d+) us }.*\[Max GPU Time\] { START: (\d+) END: (\d+).*\[CPU wait GPU Time\] { START: (\d+) END: (\d+)"
)
# For 'Aggr PBCH', 'Aggr PDCCH DL', 'Aggr CSI-RS'
gpusetuprun_pattern2 = re.compile(
    r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV.MAP_DL\] \[PHYDRV\] SFN (\d+)\.(\d+) (.*) .{16,} Map (\d+) times ===> \[CPU CUDA time\] { START: (\d+) END: (\d+) .* \[GPU Setup\] { DURATION: (\d+.\d+) us } \[GPU Run] { DURATION: (\d+.\d+) us } \[Max GPU Time\] { START: (\d+) END: (\d+) .* \[CPU wait GPU Time\] { START: (\d+) END: (\d+)"
)
# For 'COMPRESSION DL'
gpusetuprun_pattern3 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV\.MAP_DL\] \[PHYDRV\] SFN (\d+)\.(\d+) COMPRESSION DL (\d+) bitwidths Cell (\d+) Map (\d+) times ===> \[CPU CUDA time\] { START: (\d+) END: (\d+) .* \[CPU wait GPU Time\] { START: (\d+) END: (\d+) .*\[GPU Execution Duration\] { DURATION: (\d+.\d+)")

# For 'PUSCH Aggr'
gpusetuprun_pattern4 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV.MAP_UL\] \[PHYDRV\] SFN (\d+)\.(\d+) (.*) .{16,} Map (\d+) times ===> \[CPU CUDA Setup\] { START: (\d+) END: (\d+) .* \[CPU CUDA Run\] { START: (\d+) END: (\d+) .* \[GPU Setup Ph1\] { DURATION: (\d+.\d+) us } \[GPU Setup Ph2\] { DURATION: (\d+.\d+) us } \[GPU Run] { DURATION: (\d+.\d+) us } \[GPU Run EH] { DURATION: (\d+.\d+) us } \[GPU Run Post EH] { DURATION: (\d+.\d+) us } \[GPU Run Phase 2] { DURATION: (\d+.\d+) us } \[GPU Run Gap] { DURATION: (\d+.\d+) us } \[Max GPU Time\] { START: (\d+) END: (\d+) .* \[CPU wait GPU Time\] { START: (\d+) END: (\d+)")

# For 'ORDER'
gpusetuprun_order_pattern1 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV.MAP_UL\] \[PHYDRV\] SFN (\d+)\.(\d+) (.*) .{16,} Cell (\d+) Map (\d+) times ===> \[CPU CUDA time\] { START: (\d+) END: (\d+) .* \[GPU Order Run\] { DURATION: (\d+.\d+) us .*}")
gpusetuprun_order_pattern2 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV.MAP_UL\] \[PHYDRV\] SFN (\d+)\.(\d+) (.*) .{16,} Cell (\d+) Map (\d+) times ===> \[CPU CUDA time\] { START: (\d+) END: (\d+) .* \[GPU Order Run\] { DURATION: (\d+.\d+) us .*} \[RX Packet Times] { EARLY: (\d+) ONTIME: (\d+) LATE: (\d+) }\[First Last RX Packet Times\] { FIRST: (\d+) LAST: (\d+) }")
order_format = None #global var used to lock in order format (value of 1 or 2 once locked in)

def parse_gpu_setup_and_run(phylog_line, condensed_format=False, gpu_task_filters=None):
    """
    Parses GPU setup/run times from slot_map_dl.cpp

    This function process a single phylog line, returns list with a single dictionary or []
    """
    global gpusetuprun_pattern1
    global gpusetuprun_pattern2
    global gpusetuprun_pattern3
    global gpusetuprun_pattern4
    global gpusetuprun_order_pattern1
    global gpusetuprun_order_pattern2
    global order_format

    result = None

    # Example:
    # pattern1: 19:43:08.885389 I [DRV.MAP_DL] [PHYDRV] SFN 156.13 PDSCH Aggr 1709760ed2ac9ada Map 45
    found = gpusetuprun_pattern1.match(phylog_line)
    if found:

        # print("pattern1: %s"%(phylog_line[0:85]))
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn % NUM_20SLOT_GROUPS) * 20
        task = found[4]
        map = int(found[5])

        # PDSCH H2D Copy duration (optional field, can be None)
        pdsch_h2d_copy_duration = float(found[6]) if found[6] is not None else 0.0

        cuda_setup_start_timestamp = int(found[7])
        cuda_setup_end_timestamp = int(found[8])
        cuda_run_start_timestamp = int(found[9])
        cuda_run_end_timestamp = int(found[10])

        gpu_setup_duration = float(found[11])
        gpu_run_duration = float(found[12])

        max_gpu_start_timestamp = int(found[13])
        max_gpu_end_timestamp = int(found[14])

        wait_gpu_start_timestamp = int(found[15])
        wait_gpu_end_timestamp = int(found[16])

        # Determine slot time
        t0_timestamp = sfn_to_tai(sfn, slot % 20, max_gpu_start_timestamp, 0, 0)

        if(not condensed_format):
            result = {
                "t0_timestamp": t0_timestamp,
                "log_timestamp": log_timestamp,
                "sfn": sfn,
                "slot": slot,
                "task": task,
                "map": map,
                "cuda_setup_start_timestamp": cuda_setup_start_timestamp,
                "cuda_setup_end_timestamp": cuda_setup_end_timestamp,
                "cuda_setup_start_datetime": timestamp2datetime(cuda_setup_start_timestamp),
                "cuda_setup_end_datetime": timestamp2datetime(cuda_setup_end_timestamp),
                "cuda_run_start_timestamp": cuda_run_start_timestamp,
                "cuda_run_end_timestamp": cuda_run_end_timestamp,
                "cuda_run_start_datetime": timestamp2datetime(cuda_run_start_timestamp),
                "cuda_run_end_datetime": timestamp2datetime(cuda_run_end_timestamp),
                "gpu_setup1_duration": gpu_setup_duration,
                "gpu_setup2_duration": 0,
                "gpu_run_duration": gpu_run_duration,
                "pdsch_h2d_copy_duration": pdsch_h2d_copy_duration,
                "max_gpu_start_timestamp": max_gpu_start_timestamp,
                "max_gpu_end_timestamp": max_gpu_end_timestamp,
                "wait_gpu_start_timestamp": wait_gpu_start_timestamp,
                "wait_gpu_end_timestamp": wait_gpu_end_timestamp,
            }
        else:
            result = {
                "t0_timestamp": t0_timestamp,
                "task": task,
                "slot": slot,
                "cpu_setup_duration": (cuda_setup_end_timestamp - cuda_setup_start_timestamp)/1e3,
                "cpu_run_duration": (cuda_run_end_timestamp - cuda_run_start_timestamp)/1e3,
                "gpu_setup1_duration": gpu_setup_duration,
                "gpu_setup2_duration": 0,
                "gpu_run_duration": gpu_run_duration,
                "pdsch_h2d_copy_duration": pdsch_h2d_copy_duration,
            }

    # Example:
    # pattern2: 19:43:07.929116 I [DRV.MAP_DL] [PHYDRV] SFN 61.0 Aggr CSI-RS 1709760ed6b29922 Map 461
    # pattern2: 19:43:07.929490 I [DRV.MAP_DL] [PHYDRV] SFN 61.1 Aggr PDCCH DL 1709760ed4ab373a Map 4
    # pattern2: 19:43:07.929497 I [DRV.MAP_DL] [PHYDRV] SFN 61.1 Aggr PBCH 1709760ed68e49cb Map 462 t
    found = gpusetuprun_pattern2.match(phylog_line)
    if found:

        # print("pattern2: %s"%(phylog_line[0:85]))
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn % NUM_20SLOT_GROUPS) * 20
        task = found[4]
        map = int(found[5])

        # These messages only have a cuda "time", assume setup=0
        cuda_setup_start_timestamp = int(found[6])
        cuda_setup_end_timestamp = int(found[6])
        cuda_run_start_timestamp = int(found[6])
        cuda_run_end_timestamp = int(found[7])

        gpu_setup_duration = float(found[8])  # Note - these durations are in usec
        gpu_run_duration = float(found[9])

        max_gpu_start_timestamp = int(found[10])
        max_gpu_end_timestamp = int(found[11])

        wait_gpu_start_timestamp = int(found[12])
        wait_gpu_end_timestamp = int(found[13])

        # Determine slot time
        t0_timestamp = sfn_to_tai(sfn, slot % 20, max_gpu_start_timestamp, 0, 0)

        if(not condensed_format):
            result = {
                "t0_timestamp": t0_timestamp,
                "log_timestamp": log_timestamp,
                "sfn": sfn,
                "slot": slot,
                "task": task,
                "map": map,
                "cuda_setup_start_timestamp": cuda_setup_start_timestamp,
                "cuda_setup_end_timestamp": cuda_setup_end_timestamp,
                "cuda_run_start_timestamp": cuda_run_start_timestamp,
                "cuda_run_end_timestamp": cuda_run_end_timestamp,
                "gpu_setup1_duration": gpu_setup_duration,
                "gpu_setup2_duration": 0,
                "gpu_run_duration": gpu_run_duration,
                "max_gpu_start_timestamp": max_gpu_start_timestamp,
                "max_gpu_end_timestamp": max_gpu_end_timestamp,
                "wait_gpu_start_timestamp": wait_gpu_start_timestamp,
                "wait_gpu_end_timestamp": wait_gpu_end_timestamp,
            }
        else:
            result = {
                "t0_timestamp": t0_timestamp,
                "task": task,
                "slot": slot,
                "cpu_setup_duration": (cuda_setup_end_timestamp - cuda_setup_start_timestamp)/1e3,
                "cpu_run_duration": (cuda_run_end_timestamp - cuda_run_start_timestamp)/1e3,
                "gpu_setup1_duration": gpu_setup_duration,
                "gpu_setup2_duration": 0,
                "gpu_run_duration": gpu_run_duration,
            }

    # Example:
    # pattern3: 00:08:36.214498 I [DRV.MAP_DL] [PHYDRV] SFN 57.10 COMPRESSION DL 14 bitwidths Cell 41
    found = gpusetuprun_pattern3.match(phylog_line)
    if found:

        # print("pattern3: %s"%(phylog_line[0:85]))
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn % NUM_20SLOT_GROUPS) * 20
        task = "COMPRESSION DL"
        bitwidth = int(found[4])
        cell = int(found[5])
        map = int(found[6])

        # These messages only have a cuda "time", assume setup=0
        cuda_setup_start_timestamp = int(found[7])
        cuda_setup_end_timestamp = int(found[7])
        cuda_run_start_timestamp = int(found[7])
        cuda_run_end_timestamp = int(found[8])

        gpu_setup_duration = 0.0#Note - these durations are in usec
        gpu_run_duration = float(found[11])

        max_gpu_start_timestamp = int(found[7])
        max_gpu_end_timestamp = int(found[10])

        wait_gpu_start_timestamp = int(found[9])
        wait_gpu_end_timestamp = int(found[10])

        # Determine slot time
        t0_timestamp = sfn_to_tai(sfn, slot % 20, max_gpu_start_timestamp, 0, 0)

        if(not condensed_format):
            result = {
                "t0_timestamp": t0_timestamp,
                "log_timestamp": log_timestamp,
                "sfn": sfn,
                "slot": slot,
                "task": task,
                "cell": cell,
                "map": map,
                "cuda_setup_start_timestamp": cuda_setup_start_timestamp,
                "cuda_setup_end_timestamp": cuda_setup_end_timestamp,
                "cuda_run_start_timestamp": cuda_run_start_timestamp,
                "cuda_run_end_timestamp": cuda_run_end_timestamp,
                "gpu_setup1_duration": gpu_setup_duration,
                "gpu_setup2_duration": 0,
                "gpu_run_duration": gpu_run_duration,
                "max_gpu_start_timestamp": max_gpu_start_timestamp,
                "max_gpu_end_timestamp": max_gpu_end_timestamp,
                "wait_gpu_start_timestamp": wait_gpu_start_timestamp,
                "wait_gpu_end_timestamp": wait_gpu_end_timestamp,
            }
        else:
            result = {
                "t0_timestamp": t0_timestamp,
                "task": task,
                "cell": cell,
                "slot": slot,
                "cpu_setup_duration": (cuda_setup_end_timestamp - cuda_setup_start_timestamp)/1e3,
                "cpu_run_duration": (cuda_run_end_timestamp - cuda_run_start_timestamp)/1e3,
                "gpu_setup1_duration": gpu_setup_duration,
                "gpu_setup2_duration": 0,
                "gpu_run_duration": gpu_run_duration,
            }

    # Example:
    # pattern4: 20:34:51.255288 INF 102602 0 [DRV.MAP_UL] [PHYDRV] SFN 121.4 PUSCH Aggr 1795716738ecf8aa Map 482 times ===> [CPU CUDA Setup] { START: 1699389291251437400 END: 1699389291251628640 DURATION: 191 us Setup STATUS: 1 } [CPU CUDA Run] { START: 1699389291251628640 END: 1699389291251660479 DURATION: 31 us Run STATUS: 1 } [GPU Setup Ph1] { DURATION: 17.22 us } [GPU Setup Ph2] { DURATION: 70.18 us } [GPU Run] { DURATION: 1507.20 us } [GPU Run EH] { DURATION: 323.58 us } [GPU Run Post EH] { DURATION: 1077.86 us } [GPU Run Phase 2] { DURATION: 86.66 us } [GPU Run Gap] { DURATION: 6.91 us } [Max GPU Time] { START: 1699389291251437400 END: 1699389291255062081 DURATION: 3624 us } [CPU wait GPU Time] { START: 1699389291253921185 END: 1699389291255062081 DURATION: 1140 us } [Callback] { START: 1699389291255094407 END: 1699389291255191297 DURATION: 96 us }
    found = gpusetuprun_pattern4.match(phylog_line)
    if found:
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn % NUM_20SLOT_GROUPS) * 20
        task = found[4]
        map = int(found[5])

        cuda_setup_start_timestamp = int(found[6])
        cuda_setup_end_timestamp = int(found[7])
        cuda_run_start_timestamp = int(found[8])
        cuda_run_end_timestamp = int(found[9])

        gpu_setup1_duration = float(found[10])
        gpu_setup2_duration = float(found[11])
        gpu_run_duration = float(found[12])

        gpu_run1_duration = float(found[13])# EH run
        gpu_run2_duration = float(found[14])# Non-EH run
        gpu_run3_duration = float(found[15])# Data copy
        gpu_run4_duration = float(found[16])# Gap (between phase 1 and phase 2)

        max_gpu_start_timestamp = int(found[17])
        max_gpu_end_timestamp = int(found[18])

        wait_gpu_start_timestamp = int(found[19])
        wait_gpu_end_timestamp = int(found[20])

        # Determine slot time
        t0_timestamp = sfn_to_tai(sfn, slot % 20, max_gpu_start_timestamp, 0, 0)

        if(not condensed_format):
            result = {
                "t0_timestamp": t0_timestamp,
                "log_timestamp": log_timestamp,
                "sfn": sfn,
                "slot": slot,
                "task": task,
                "map": map,
                "cuda_setup_start_timestamp": cuda_setup_start_timestamp,
                "cuda_setup_end_timestamp": cuda_setup_end_timestamp,
                "cuda_setup_start_datetime": timestamp2datetime(cuda_setup_start_timestamp),
                "cuda_setup_end_datetime": timestamp2datetime(cuda_setup_end_timestamp),
                "cuda_run_start_timestamp": cuda_run_start_timestamp,
                "cuda_run_end_timestamp": cuda_run_end_timestamp,
                "cuda_run_start_datetime": timestamp2datetime(cuda_run_start_timestamp),
                "cuda_run_end_datetime": timestamp2datetime(cuda_run_end_timestamp),
                "gpu_setup1_duration": gpu_setup1_duration,
                "gpu_setup2_duration": gpu_setup2_duration,
                "gpu_run_duration": gpu_run_duration,
                "gpu_run_eh_duration": gpu_run1_duration,
                "gpu_run_noneh_duration": gpu_run2_duration,
                "gpu_run_copy_duration": gpu_run3_duration,
                "gpu_run_gap_duration": gpu_run4_duration,
                "max_gpu_start_timestamp": max_gpu_start_timestamp,
                "max_gpu_end_timestamp": max_gpu_end_timestamp,
                "wait_gpu_start_timestamp": wait_gpu_start_timestamp,
                "wait_gpu_end_timestamp": wait_gpu_end_timestamp,
            }
        else:
            result = {
                "t0_timestamp": t0_timestamp,
                "task": task,
                "slot": slot,
                "cpu_setup_duration": (cuda_setup_end_timestamp - cuda_setup_start_timestamp)/1e3,
                "cpu_run_duration": (cuda_run_end_timestamp - cuda_run_start_timestamp)/1e3,
                "gpu_run_eh_duration": gpu_run1_duration,
                "gpu_run_noneh_duration": gpu_run2_duration,
                "gpu_run_copy_duration": gpu_run3_duration,
                "gpu_run_gap_duration": gpu_run4_duration,
                "gpu_setup1_duration": gpu_setup1_duration,
                "gpu_setup2_duration": gpu_setup2_duration,
                "gpu_run_duration": gpu_run_duration,
            }

    # Example:
    # pattern5: 19:51:01.864778 I [DRV.MAP_UL] [PHYDRV] SFN 350.4 ORDER 17463d65b02a28c9 Cell 41 Map 376 times ===> [CPU CUDA time] { START: 1677095461861016672 END: 1677095461861031724 DURATION: 15 us } [CPU RX U-plane] { START: 0 END: 0 DURATION: 0 us } [CPU Free U-plane] { START: 0 END: 0 DURATION: 0 us } [CPU Order Run] { START: 0 END: 0 DURATION: 0 us} [GPU Order Idle] { DURATION: 3.07 us } [GPU Order Run] { DURATION: 2116.61 us } [RX Packet Times] { EARLY: 0 ONTIME: 0 LATE: 0 }
    # pattern5: 15:25:07.968812 I [DRV.MAP_UL] [PHYDRV] SFN 864.15 ORDER 1747b7bec9055a51 Cell 41 Map 387 times ===> [CPU CUDA time] { START: 1677511507966418478 END: 1677511507966429898 DURATION: 11 us } [CPU RX U-plane] { START: 0 END: 0 DURATION: 0 us } [CPU Free U-plane] { START: 0 END: 0 DURATION: 0 us } [CPU Order Run] { START: 0 END: 0 DURATION: 0 us} [GPU Order Idle] { DURATION: 3.07 us } [GPU Order Run] { DURATION: 447.49 us } [RX Packet Times] { EARLY: 3 ONTIME: 0 LATE: 0 }[First Last RX Packet Times] { FIRST: 1677511507967730709 LAST: 1677511507967945453 }
    found = False
    if(order_format is None):
        found1 = gpusetuprun_order_pattern1.match(phylog_line)
        found2 = gpusetuprun_order_pattern2.match(phylog_line)
        if(found2):
            order_format = 2
            found = found2
        elif(found1):
            order_format = 1
            found = found1
    elif(order_format == 1):
        found = gpusetuprun_order_pattern1.match(phylog_line)
    else:
        found = gpusetuprun_order_pattern2.match(phylog_line)
    if found:
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn % NUM_20SLOT_GROUPS) * 20
        task = found[4]
        cell = int(int(found[5]) - 41)
        map = int(found[6])

        #Setting values for setup/run to same
        cuda_setup_start_timestamp = int(found[7])
        cuda_setup_end_timestamp = int(found[8])
        cuda_run_start_timestamp = int(found[7])
        cuda_run_end_timestamp = int(found[8])

        gpu_setup1_duration = 0.0
        gpu_setup2_duration = 0.0
        gpu_run_duration = float(found[9])

        #This is bogus (does not apply)
        max_gpu_start_timestamp = int(found[7])
        max_gpu_end_timestamp = int(found[8])

        #This is bogus (does not apply)
        wait_gpu_start_timestamp = int(found[7])
        wait_gpu_end_timestamp = int(found[8])

        if(order_format == 2):
            early_count = int(found[10])
            ontime_count = int(found[11])
            late_count = int(found[12])

            first_hw_timestamp = int(found[13])
            last_hw_timestamp = int(found[14])

        # Determine slot time
        t0_timestamp = sfn_to_tai(sfn, slot % 20, max_gpu_start_timestamp, 0, 0)

        if(not condensed_format):
            result = {
                "t0_timestamp": t0_timestamp,
                "log_timestamp": log_timestamp,
                "sfn": sfn,
                "slot": slot,
                "task": task,
                "cell": cell,
                "map": map,
                "cuda_setup_start_timestamp": cuda_setup_start_timestamp,
                "cuda_setup_end_timestamp": cuda_setup_end_timestamp,
                "cuda_setup_start_datetime": timestamp2datetime(cuda_setup_start_timestamp),
                "cuda_setup_end_datetime": timestamp2datetime(cuda_setup_end_timestamp),
                "cuda_run_start_timestamp": cuda_run_start_timestamp,
                "cuda_run_end_timestamp": cuda_run_end_timestamp,
                "cuda_run_start_datetime": timestamp2datetime(cuda_run_start_timestamp),
                "cuda_run_end_datetime": timestamp2datetime(cuda_run_end_timestamp),
                "gpu_setup1_duration": gpu_setup1_duration,
                "gpu_setup2_duration": gpu_setup2_duration,
                "gpu_run_duration": gpu_run_duration,
                "max_gpu_start_timestamp": max_gpu_start_timestamp,
                "max_gpu_end_timestamp": max_gpu_end_timestamp,
                "wait_gpu_start_timestamp": wait_gpu_start_timestamp,
                "wait_gpu_end_timestamp": wait_gpu_end_timestamp,
            }
            if(order_format == 2):
                result["first_hw_timestamp"] = first_hw_timestamp
                result["last_hw_timestamp"] =  last_hw_timestamp
        else:
            result = {
                "t0_timestamp": t0_timestamp,
                "task": task,
                "slot": slot,
                "cpu_setup_duration": (cuda_setup_end_timestamp - cuda_setup_start_timestamp)/1e3,
                "cpu_run_duration": (cuda_run_end_timestamp - cuda_run_start_timestamp)/1e3,
                "gpu_setup1_duration": gpu_setup1_duration,
                "gpu_setup2_duration": gpu_setup2_duration,
                "gpu_run_duration": gpu_run_duration,
            }

    if(result is not None and (gpu_task_filters is None or task in gpu_task_filters)):
        if(not condensed_format):
            result['start_timestamp'] = result['max_gpu_start_timestamp']
            result['end_timestamp'] = result['max_gpu_end_timestamp']
            result['cuda_setup_duration'] = (result['cuda_setup_end_timestamp'] - result['cuda_setup_start_timestamp'])/1000.
            result['cuda_run_duration'] = (result['cuda_run_end_timestamp'] - result['cuda_run_start_timestamp'])/1000.
            result['max_gpu_duration'] = (result['max_gpu_end_timestamp'] - result['max_gpu_start_timestamp'])/1000.
            result['wait_gpu_duration'] = (result['wait_gpu_end_timestamp'] - result['wait_gpu_start_timestamp'])/1000.
            result['execution_diff'] = result['max_gpu_duration'] - (result["gpu_setup1_duration"] + result["gpu_setup2_duration"] + result['gpu_run_duration'])
            result['max_gpu_start_deadline'] = (result['max_gpu_start_timestamp'] - result['t0_timestamp'])/1000.
            result['max_gpu_end_deadline'] = (result['max_gpu_end_timestamp'] - result['t0_timestamp'])/1000.
            result['cuda_setup_start_deadline'] = (result['cuda_setup_start_timestamp'] - result['t0_timestamp'])/1000.
            result['cuda_setup_end_deadline'] = (result['cuda_setup_start_timestamp'] - result['t0_timestamp'])/1000.
            result['wait_gpu_start_deadline'] = (result['wait_gpu_start_timestamp'] - result['t0_timestamp'])/1000.
            result['wait_gpu_end_deadline'] = (result['wait_gpu_end_timestamp'] - result['t0_timestamp'])/1000.

        result["gpu_total_duration"] =  result["gpu_setup1_duration"]+result["gpu_setup2_duration"]+result["gpu_run_duration"]

        # Return as list (to match formatting of other parsers)
        return [result]
    else:
        return []

compression_pattern = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV\.MAP_DL\] \[PHYDRV\] SFN (\d+)\.(\d+) COMPRESSION DL (\d+) bitwidths Cell (\d+) Map (\d+) times ===> \[CPU CUDA time\] { START: (\d+) END: (\d+) .* \[CPU wait GPU Time\] { START: (\d+) END: (\d+) .* \[S0 Prepare Execution Duration 1\] { DURATION: (\d+.\d+) .* \[S0 Prepare Execution Duration 2\] { DURATION: (\d+.\d+) .* \[S0 Prepare Execution Duration 3\] { DURATION: (\d+.\d+) .* \[Channel To Compression Gap\] { DURATION: (\d+.\d+) .* \[GPU Execution Duration\] { DURATION: (\d+.\d+)")

def parse_new_compression_message(phylog_line):
    global compression_pattern
    result = []

    # Example:
    # pattern3: 00:08:36.214498 I [DRV.MAP_DL] [PHYDRV] SFN 57.10 COMPRESSION DL 14 bitwidths Cell 41
    found = compression_pattern.match(phylog_line)
    if found:

        #print("pattern3: %s"%(phylog_line[0:85]))
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn%NUM_20SLOT_GROUPS)*20
        bitwidth = int(found[4])
        task = "COMPRESSION DL"
        cell = int(found[5])
        map = int(found[6])

        # Parse times and durations
        cuda_run_start_timestamp = int(found[7])
        cuda_run_end_timestamp = int(found[8])
        cpu_wait_start_timestamp = int(found[9])
        cpu_wait_end_timestamp = int(found[10])
        gpu_prepare_duration1 = float(found[11])
        gpu_prepare_duration2 = float(found[12])
        gpu_prepare_duration3 = float(found[13])
        gpu_channel_to_compression_gap = float(found[14])
        gpu_compression_duration = float(found[15])

        #Determine slot time
        t0_timestamp = sfn_to_tai(sfn, slot%20, cuda_run_start_timestamp, 0, 0)

        # Add result for cuda_run
        result.append({"t0_timestamp": t0_timestamp,
                       "log_timestamp": log_timestamp,
                       "sfn": sfn,
                       "slot": slot,
                       "task": task,
                       "cell": cell,
                       "map": map,
                       "event": "cpu_cuda_run",
                       "duration": (cuda_run_end_timestamp - cuda_run_start_timestamp)/1000.0
                       })
        # Add result for cpu_wait
        result.append({"t0_timestamp": t0_timestamp,
                       "log_timestamp": log_timestamp,
                       "sfn": sfn,
                       "slot": slot,
                       "task": task,
                       "cell": cell,
                       "map": map,
                       "event": "cpu_wait_on_gpu",
                       "duration": (cpu_wait_end_timestamp - cpu_wait_start_timestamp)/1000.0
                       })
        # Add result for prepare_memsets
        result.append({"t0_timestamp": t0_timestamp,
                       "log_timestamp": log_timestamp,
                       "sfn": sfn,
                       "slot": slot,
                       "task": task,
                       "cell": cell,
                       "map": map,
                       "event": "prepare_memsets",
                       "duration": gpu_prepare_duration1
                       })
        # Add result for pre_prepare_kernel
        result.append({"t0_timestamp": t0_timestamp,
                       "log_timestamp": log_timestamp,
                       "sfn": sfn,
                       "slot": slot,
                       "task": task,
                       "cell": cell,
                       "map": map,
                       "event": "pre_prepare_kernel",
                       "duration": gpu_prepare_duration2
                       })
        # Add result for prepare_kernel
        result.append({"t0_timestamp": t0_timestamp,
                       "log_timestamp": log_timestamp,
                       "sfn": sfn,
                       "slot": slot,
                       "task": task,
                       "cell": cell,
                       "map": map,
                       "event": "prepare_kernel",
                       "duration": gpu_prepare_duration3
                       })
        # Add result for prepare_total
        result.append({"t0_timestamp": t0_timestamp,
                       "log_timestamp": log_timestamp,
                       "sfn": sfn,
                       "slot": slot,
                       "task": task,
                       "cell": cell,
                       "map": map,
                       "event": "prepare_total",
                       "duration": gpu_prepare_duration1+gpu_prepare_duration2+gpu_prepare_duration3
                       })
        # Add result for channel_to_compression_gap
        result.append({"t0_timestamp": t0_timestamp,
                       "log_timestamp": log_timestamp,
                       "sfn": sfn,
                       "slot": slot,
                       "task": task,
                       "cell": cell,
                       "map": map,
                       "event": "channel_to_compression_gap",
                       "duration": gpu_channel_to_compression_gap
                       })
        # Add result for compression_kernel
        result.append({"t0_timestamp": t0_timestamp,
                       "log_timestamp": log_timestamp,
                       "sfn": sfn,
                       "slot": slot,
                       "task": task,
                       "cell": cell,
                       "map": map,
                       "event": "compression_kernel",
                       "duration": gpu_compression_duration
                       })

    return result

def parse_last_n_bytes(filename,nn,parser_list,column_filter_list):
    """
    Seeks to the last 'nn' bytes and run desired parsing

    parser_list is list of parsers
    column_filter_list is list of columns to keep for each parser
    (Note this matches interface for Parsenator)
    """

    #Must have a definition for column filters for every parser
    assert(len(parser_list)==len(column_filter_list))

    #Limit search to entire file
    NN = min(os.path.getsize(filename),nn)

    with open(filename, 'rb') as fid:
        #Seek to end of file
        segment = None
        offset = 0
        fid.seek(-NN, os.SEEK_END)
        buffer = fid.read().decode(encoding='utf-8')
        lines = buffer.split("\n")
        lines = lines[1:]

        def parsing_func(line_list, parser_list, column_filter_list):
            output_results = [[] for ii in range(len(parser_list))]

            #Process all lines
            for line in line_list:
                for ii,parser in enumerate(parser_list):
                    current_column_filter = column_filter_list[ii]
                    results = parser(line)
                    for res in results:
                        if(current_column_filter is not None):
                            #Down-select only keys that we need
                            temp_dict = {key: val for key, val in res.items() if key in column_filter_list[ii]}
                            output_results[ii].append(temp_dict)
                        else:
                            output_results[ii].append(res)

            return output_results

        data_list = parsing_func(lines,parser_list,column_filter_list)

    return [pd.DataFrame(dd) for dd in data_list]


ul_packet_summary_re = re.compile(r"(\d+\:\d+\:\d+\.\d+) .*\[DRV\.UL_PACKET_SUMMARY\] Slot (\d+) \| (.*) \|")

def parse_ul_packet_summary(line):
    global ul_packet_summary_re

    found = ul_packet_summary_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        slot = int(found[2])

        data_string = found[3]
        for ii,timing_data in enumerate(data_string.split("|")):
            vals = timing_data.split(",")
            if(vals[0].rstrip().lstrip().isdigit()):
                #Note: there are messages with both counts and percentages, we only process the counts
                early_count = int(vals[0])
                ontime_count = int(vals[1])
                late_count = int(vals[2])
            else:
                #This is a percentage message, do not process it
                break
            
            total_count = early_count+ontime_count+late_count
            if(total_count > 0):
                #Note: Ignoring lines that are present but contain all 0s
                #This can happen for MMIMO test cases

                ul_ontime_percentage = 100.0*(float(ontime_count) / total_count)
                result.append({'log_timestamp': log_timestamp,
                               'slot': slot,
                               'cell': ii,
                               'early_count': early_count,
                               'ontime_count': ontime_count,
                               'late_count': late_count,
                               'ulu_ontime_percentage': ul_ontime_percentage})

    return result

srs_packet_summary_re = re.compile(r"(\d+\:\d+\:\d+\.\d+) .*\[DRV\.SRS_PACKET_SUMMARY\] Slot (\d+) \| (.*) \|")

def parse_srs_packet_summary(line):
    global srs_packet_summary_re

    found = srs_packet_summary_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        slot = int(found[2])

        data_string = found[3]
        for ii,timing_data in enumerate(data_string.split("|")):
            vals = timing_data.split(",")
            if(vals[0].rstrip().lstrip().isdigit()):
                #Note: there are messages with both counts and percentages, we only process the counts
                early_count = int(vals[0])
                ontime_count = int(vals[1])
                late_count = int(vals[2])
            else:
                #This is a percentage message, do not process it
                break

            total_count = early_count+ontime_count+late_count
            if(total_count > 0):
                #Note: Ignoring lines that are present but contain all 0s
                
                srs_ontime_percentage = 100.0*(float(ontime_count) / total_count)
                result.append({'log_timestamp': log_timestamp,
                               'slot': slot,
                               'cell': ii,
                               'early_count': early_count,
                               'ontime_count': ontime_count,
                               'late_count': late_count,
                               'srs_ontime_percentage': srs_ontime_percentage})

    return result


FH_PACKET_SUMMARY_TYPE_MAP = {
    'DLU': 'dlu',
    'DLC': 'dlc',
    'ULC': 'ulc',
    'ULU TX PRACH': 'ulutx_prach',
    'ULU TX PUCCH': 'ulutx_pucch',
    'ULU TX PUSCH': 'ulutx_pusch',
    'ULU TX SRS': 'ulutx_srs',
}

fh_packet_summary_re = re.compile(
    r"(\d+\:\d+\:\d+\.\d+) .*\[FH\.PACKET_SUMMARY\] "
    r"(ULC|DLC|DLU|ULU TX PRACH|ULU TX PUCCH|ULU TX PUSCH|ULU TX SRS) "
    r"Slot (\d+) \| (.*) \|"
)

def parse_fh_packet_summary(line):
    global fh_packet_summary_re

    found = fh_packet_summary_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        fh_type = FH_PACKET_SUMMARY_TYPE_MAP[found[2]]
        slot = int(found[3])

        data_string = found[4]
        for ii, timing_data in enumerate(data_string.split("|")):
            vals = timing_data.split(",")
            if vals[0].rstrip().lstrip().isdigit():
                early_count = int(vals[0])
                ontime_count = int(vals[1])
                late_count = int(vals[2])
            else:
                break

            total_count = early_count + ontime_count + late_count
            if total_count > 0:
                ontime_percentage = 100.0 * (float(ontime_count) / total_count)
                result.append({'log_timestamp': log_timestamp,
                               'type': fh_type,
                               'slot': slot,
                               'cell': ii,
                               'ontime_percentage': ontime_percentage})

    return result


_ONTIME_COLUMNS = ['log_timestamp', 'slot', 'cell', 'type', 'ontime_percentage']


def build_df_du_ontime(phy_filename, bytes_to_read=1000000):
    parsed_result = parse_last_n_bytes(phy_filename, bytes_to_read,
                                       [parse_ul_packet_summary, parse_srs_packet_summary],
                                       [None, None])
    df_ul = parsed_result[0]
    df_srs = parsed_result[1]

    if len(df_ul) > 0:
        df_ul['type'] = 'ulu'
        df_ul = df_ul.rename(columns={'ulu_ontime_percentage': 'ontime_percentage'})
    if len(df_srs) > 0:
        df_srs['type'] = 'srs'
        df_srs = df_srs.rename(columns={'srs_ontime_percentage': 'ontime_percentage'})

    result = pd.concat([df_ul, df_srs], ignore_index=True)
    if len(result) == 0:
        return pd.DataFrame(columns=_ONTIME_COLUMNS)
    return result


def build_df_ru_ontime(ru_filename, bytes_to_read=1000000):
    parsed_result = parse_last_n_bytes(ru_filename, bytes_to_read,
                                       [parse_fh_packet_summary],
                                       [None])
    result = parsed_result[0]
    if len(result) == 0:
        return pd.DataFrame(columns=_ONTIME_COLUMNS)
    return result


map_re_ru_aggregate = {
    (Direction.DOWNLINK, Plane.CONTROL, Statistic.AVERAGE): re.compile('(\d+\:\d+\:\d+\.\d+).*\[RU\].*DL C AVG ON TIME SLOTS %\s+\|\s+(\d+(?:\.\d+)?)%'),
    (Direction.DOWNLINK, Plane.CONTROL, Statistic.MINIMUM): re.compile('(\d+\:\d+\:\d+\.\d+).*\[RU\].*DL C MIN ON TIME SLOTS %\s+\|\s+(\d+(?:\.\d+)?)%'),
    (Direction.DOWNLINK, Plane.USER,    Statistic.AVERAGE): re.compile('(\d+\:\d+\:\d+\.\d+).*\[RU\].*DL U AVG ON TIME SLOTS %\s+\|\s+(\d+(?:\.\d+)?)%'),
    (Direction.DOWNLINK, Plane.USER,    Statistic.MINIMUM): re.compile('(\d+\:\d+\:\d+\.\d+).*\[RU\].*DL U MIN ON TIME SLOTS %\s+\|\s+(\d+(?:\.\d+)?)%'),
    (Direction.UPLINK,   Plane.CONTROL, Statistic.AVERAGE): re.compile('(\d+\:\d+\:\d+\.\d+).*\[RU\].*UL C AVG ON TIME SLOTS %\s+\|\s+(\d+(?:\.\d+)?)%'),
    (Direction.UPLINK,   Plane.CONTROL, Statistic.MINIMUM): re.compile('(\d+\:\d+\:\d+\.\d+).*\[RU\].*UL C MIN ON TIME SLOTS %\s+\|\s+(\d+(?:\.\d+)?)%')
}


def parse_ru_aggregate_stat(line, direction: Direction, plane: Plane, statistic: Statistic):
    if not direction in Direction:
        print(f'ERROR:: {direction} not valid.')
        return False
    if not plane in Plane:
        print(f'ERROR:: {plane} not valid.')
        return False
    if not statistic in Statistic:
        print(f'ERROR:: {statistic} not valid.')
    map_re_key = (direction, plane, statistic)
    regex = map_re_ru_aggregate[map_re_key]    
    found = regex.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        percentage    = float(found[2])
        result = [{
            'log_timestamp': log_timestamp,
            'statistic'    : percentage
        }]    
    return result

# example
# 13:05:20.143472 WRN 2516 0 [RU] | AVG ONTIME U Slot 32 %         |     100.00% |
re_u_avg_ontime         = re.compile('(\d+\:\d+\:\d+\.\d+).*\[RU\].*AVG ONTIME U Slot (\d+) %\s+\|\s+(\d+(?:\.\d+)?)')
def parse_ru_aggregate_uplink_average_ontime(line):
    found = re_u_avg_ontime.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        slot_id       = int(found[2])
        statistic     = float(found[3])
        result = [{
            'log_timestamp': log_timestamp,
            'slot_id'      : slot_id,
            'average'      : statistic
        }]
    return result


timingdebug_pattern1 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV\.FUNC_DL\] (.{6,8}) {packet timing debug} cell=(\d+) desired_start=(\d+) cpu=(\d+) sym=(\d+) actual_start=(\d+) wqe_idx=(\d+)")

def parse_packet_timing_debug_messages(line,packet_timing_cache):

    found = timingdebug_pattern1.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        slot_string = found[2]
        (oran_frame,oran_subframe,oran_slot) = parse_oran_string(slot_string)
        frame = oran_frame
        slot = parse_slot(slot_string) + (frame%NUM_20SLOT_GROUPS)*20
        cell = int(found[3])
        desired_start_time = int(found[4])
        cpu_time = int(found[5])
        symbol = int(found[6])
        actual_start_time = int(found[7])
        wqe_idx = int(found[8])

        t0_timestamp = oran_fn_to_tai(oran_frame,oran_subframe,oran_slot,desired_start_time,0,0)

        #Find difference between expected and actual
        expected_window = (GH_4TR_DLU_WINDOW_OFFSET1 + symbol*SYMBOL_TIME) * 1e3
        desired_window =  desired_start_time - t0_timestamp
        actual_window =  actual_start_time - t0_timestamp
        desired_offset_usec = (desired_window - expected_window) / 1e3
        actual_offset_usec = (actual_window - expected_window) / 1e3

        #Get sequence numbers (these messages occur in a sequence for each symbol)
        key = (t0_timestamp,symbol,cell)
        if(key not in packet_timing_cache.keys()):
            packet_timing_cache[key] = 0
        else:
            packet_timing_cache[key] += 1

        seq_num = packet_timing_cache[key]

        result.append({'log_timestamp': log_timestamp,
                       't0_timestamp': t0_timestamp,
                       'symbol_window_timestamp': expected_window + t0_timestamp,
                       'frame': frame,
                       'slot': slot,
                       'cell': cell,
                       'desired_start_time': desired_start_time,
                       'cpu_time': cpu_time,
                       'symbol': symbol,
                       'actual_start_time': actual_start_time,
                       'wqe_idx': wqe_idx,
                       'seq_num': seq_num,
                       'desired_offset_usec': desired_offset_usec,
                       'actual_offset_usec': actual_offset_usec})

    return result


ul_timing_message_re = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[(DRV|FH).SYMBOL_TIMINGS\] SFN (\d+)\.(\d+) Cell (\d+) early\/late ts per sym: (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+)")
srs_timing_message_re = re.compile("(\d+\:\d+\:\d+\.\d+) I.*\[(DRV|FH).SYMBOL_TIMINGS_SRS\] .*SFN (\d+)\.(\d+) Cell (\d+) early\/late ts per sym: (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+) (\d+):(\d+)")
def parse_du_symbol_timings(line,mmimo_enable=False,parse_srs=False):

    global ul_timing_message_re
    global srs_timing_message_re

    if parse_srs:
        current_re = srs_timing_message_re
        tc_type = getTCType(mmimo_enable)
        start_offset = getReceptionWindow(TrafficType.DTT_SRS, tc_type)[0]
        end_offset = getReceptionWindow(TrafficType.DTT_SRS, tc_type)[1]
    else:
        current_re = ul_timing_message_re
        tc_type = getTCType(mmimo_enable)
        start_offset = getReceptionWindow(TrafficType.DTT_ULU, tc_type)[0]
        end_offset = getReceptionWindow(TrafficType.DTT_ULU, tc_type)[1]

    found = current_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        module = found[2]
        sfn = int(found[3])
        slot = int(found[4]) + (sfn%NUM_20SLOT_GROUPS)*20
        cell = int(found[5])

        for ii in range(14):
            earliest_timestamp = int(found[6+2*ii])
            latest_timestamp = int(found[6+2*ii+1])

            if(earliest_timestamp != 18446744073709551615 and latest_timestamp != 0):
                if(parse_srs):
                    #Note: SRS window is not a function of symbol
                    window_start = start_offset
                    window_end = end_offset
                else:
                    window_start = start_offset + ii*SYMBOL_TIME
                    window_end = end_offset + ii*SYMBOL_TIME
                t0_timestamp = sfn_to_tai(sfn, slot%20, earliest_timestamp, 0, 0)
                start_deadline = (earliest_timestamp - t0_timestamp)/1000.
                end_deadline = (latest_timestamp - t0_timestamp)/1000.
                result.append({'log_timestamp': log_timestamp,
                                't0_timestamp': t0_timestamp,
                                'sfn': sfn,
                                'slot': slot,
                                'cell': cell,
                                'symbol': ii,
                                'start_timestamp': earliest_timestamp,
                                'stop_timestamp': latest_timestamp,
                                'start_deadline': start_deadline,
                                'end_deadline': end_deadline,
                                'start_offset': start_deadline - window_start,
                                'end_offset': end_deadline - window_start,
                                'window_start_deadline': window_start,
                                'window_end_deadline': window_end
                                })

    return result

def parse_frame(slot_string):
    """
    Returns the frame from ORAN string
    example: F74S2S1 --> 74
    """
    return int(slot_string.split("F")[1].split("S")[0])


def parse_slot(slot_string):
    """
    Returns the slot from ORAN string
    example: F74S2S1 --> 2*2 + 1 = 5
    """
    temp = slot_string.split("S")
    return int(temp[1]) * 2 + int(temp[2])

def parse_oran_string(slot_string):
    """
    Returns the oran frame/oran subframe/oran slot from the ORAN string
    example: F74S2S1 --> (74,2,1)
    """

    data = slot_string.split("F")[1].split("S")

    return (int(data[0]),int(data[1]),int(data[2]))


ru_tx_time_pattern1 = re.compile(r"(\d+\:\d+\:\d+\.\d+) .*\[(.*).TX_TIMINGS\] \[(.*)\] (?:(\w+) )?(.{6,8}) Cell (\d+) TX Time (\d+) Enqueue Time (\d+) Sym (\d+) Num Packets (\d+) (\d+) Queue (\d+)")

def parse_ru_tx_times(ru_line,mmimo_enable=False):
    global ru_tx_time_pattern1

    tc_type = getTCType(mmimo_enable)
    
    result = []

    # Example:
    # pattern 1: 20:34:59.222458 [RU.UL_TIMINGS] F150S2S1 Cell 0 Symbol Time 1677184499223116714
    found = ru_tx_time_pattern1.match(ru_line)
    if found:

        #print("pattern1: %s"%(ru_line[0:85]))
        log_timestamp = found[1]
        type = found[2]
        old_channel_field = found[3]
        new_channel_field = found[4]  # This could be None for old format
        # Use new field if present, otherwise use old field
        channel = new_channel_field if new_channel_field is not None else old_channel_field
        slot_string = found[5]
        cell = int(found[6])
        tx_time = int(found[7])
        enqueue_time = int(found[8])
        symbol = int(found[9])
        num_packets1 = int(found[10])
        num_packets2 = int(found[11])
        que_index = int(found[12])

        # Use SRS reception window if channel is SRS, otherwise use ULU window
        if channel == 'SRS':
            start_offset, end_offset = getReceptionWindow(TrafficType.DTT_SRS, tc_type)
        else:
            start_offset, end_offset = getReceptionWindow(TrafficType.DTT_ULU, tc_type)
        
        window_start = start_offset + symbol*SYMBOL_TIME
        window_end = end_offset + symbol*SYMBOL_TIME
        #t0_timestamp = sfn_to_tai(frame, slot%20, tx_time, 0, 0)

        (oran_frame,oran_subframe,oran_slot) = parse_oran_string(slot_string)
        t0_timestamp = oran_fn_to_tai(oran_frame,oran_subframe,oran_slot,tx_time,0,0)

        sfn_t = tai_to_sfn(t0_timestamp, 0,0)
        sfn = sfn_t.sfn
        slot = sfn_t.slot+ (sfn%NUM_20SLOT_GROUPS)*20

        # Add result for compression_kernel
        result.append({"log_timestamp": log_timestamp,
                       "t0_timestamp": t0_timestamp,
                       "type": type,
                       "channel": channel,
                       "sfn": sfn,
                       "slot": slot,
                       "cell": cell,
                       "tx_time": tx_time,
                       "enqueue_time": enqueue_time,
                       "tx_deadline": (tx_time - t0_timestamp)/1e3,
                       "enqueue_deadline": (enqueue_time - t0_timestamp)/1e3,
                       "symbol": symbol,
                       "num_packets": num_packets2,
                       "que_index": que_index,
                       "window_start_deadline": window_start,
                       "window_end_deadline": window_end
                       })

    return result

ru_tx_time_sum_pattern1 = re.compile(r"(\d+\:\d+\:\d+\.\d+) .*\[(.*).TX_TIMINGS_SUM\] \[(.*)\] (?:(\w+) )?(.{6,8}) Cell (\d+) Enqueue Time (\d+)")

def parse_ru_tx_sum_times(ru_line,mmimo_enable=False):
    global ru_tx_time_sum_pattern1

    tc_type = getTCType(mmimo_enable)
    
    result = []

    # Example:
    # pattern 1: 20:34:59.222458 [RU.UL_TIMINGS] F150S2S1 Cell 0 Symbol Time 1677184499223116714
    found = ru_tx_time_sum_pattern1.match(ru_line)
    if found:

        #print("pattern1: %s"%(ru_line[0:85]))
        log_timestamp = found[1]
        type = found[2]
        old_channel_field = found[3]
        new_channel_field = found[4]  # This could be None for old format
        # Use new field if present, otherwise use old field
        channel = new_channel_field if new_channel_field is not None else old_channel_field
        slot_string = found[5]
        cell = int(found[6])
        enqueue_time = int(found[7])

        # Use SRS reception window if channel is SRS, otherwise use ULU window
        if channel == 'SRS':
            start_offset, end_offset = getReceptionWindow(TrafficType.DTT_SRS, tc_type)
        else:
            start_offset, end_offset = getReceptionWindow(TrafficType.DTT_ULU, tc_type)

        (oran_frame,oran_subframe,oran_slot) = parse_oran_string(slot_string)
        t0_timestamp = oran_fn_to_tai(oran_frame,oran_subframe,oran_slot,enqueue_time,0,0)

        sfn_t = tai_to_sfn(t0_timestamp, 0,0)
        sfn = sfn_t.sfn
        slot = sfn_t.slot+ (sfn%NUM_20SLOT_GROUPS)*20

        # Add result for compression_kernel
        result.append({"log_timestamp": log_timestamp,
                       "t0_timestamp": t0_timestamp,
                       "type": type,
                       "channel": channel,
                       "sfn": sfn,
                       "slot": slot,
                       "cell": cell,
                       "enqueue_timestamp": enqueue_time,
                       "enqueue_deadline": (enqueue_time-t0_timestamp)/1000.,
                       "window_start_sym0_deadline": (start_offset),
                       "window_start_sym13_deadline": (start_offset + 13*SYMBOL_TIME),
                       })

    return result

latepacketsymbollevel_pattern1 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[RU.LATE_PACKETS\] (.{6,8}) Cell (\d+) (.*) T0 (\d+) .* Sym 0 (\d+) Sym 1 (\d+) Sym 2 (\d+) Sym 3 (\d+) Sym 4 (\d+) Sym 5 (\d+) Sym 6 (\d+) Sym 7 (\d+) Sym 8 (\d+) Sym 9 (\d+) Sym 10 (\d+) Sym 11 (\d+) Sym 12 (\d+) Sym 13 (\d+)")

def parse_late_packets_symbol_level(ru_line):
    """
    Grabs per-symbol late packet counts

    This function parses one ru line at a time, returns a list of dictionaries
    """

    results = []

    found = latepacketsymbollevel_pattern1.match(ru_line)
    if found:
        log_timestamp = found[1]
        slot_string = found[2]
        frame = parse_frame(slot_string)
        slot = parse_slot(slot_string) + (frame%NUM_20SLOT_GROUPS)*20
        cell = int(found[3])
        task = found[4]
        t0_timestamp = int(found[5])
        sym_counts = []
        for ii in range(14):
            sym_counts.append(int(found[6 + ii]))
        for ii in range(14):
            results.append(
                {
                    "log_timestamp": log_timestamp,
                    "t0_timestamp": t0_timestamp,
                    "task": task,
                    "cell": cell,
                    "frame": frame,
                    "slot": slot,
                    "symbol": ii,
                    "late_packet_count": sym_counts[ii],
                }
            )

    return results

latepacketslotlevel_pattern1 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[RU.LATE_PACKETS\] (.{6,8}) Cell (\d+) (.*) T0 (\d+) .* Sym 0 (\d+) Sym 1 (\d+) Sym 2 (\d+) Sym 3 (\d+) Sym 4 (\d+) Sym 5 (\d+) Sym 6 (\d+) Sym 7 (\d+) Sym 8 (\d+) Sym 9 (\d+) Sym 10 (\d+) Sym 11 (\d+) Sym 12 (\d+) Sym 13 (\d+)")

def parse_late_packets_slot_level(ru_line):
    """
    Grabs per-symbol late packet counts

    This function parses one ru line at a time, returns a list of dictionaries
    """

    results = []

    found = latepacketslotlevel_pattern1.match(ru_line)
    if found:
        log_timestamp = found[1]
        slot_string = found[2]
        frame = parse_frame(slot_string)
        slot = parse_slot(slot_string) + (frame%NUM_20SLOT_GROUPS)*20
        cell = int(found[3])
        task = found[4]
        t0_timestamp = int(found[5])
        sym_count = 0
        for ii in range(14):
            sym_count += int(found[6+ii])

        results.append({'log_timestamp': log_timestamp,
                        't0_timestamp': t0_timestamp,
                        'task': task,
                        'cell': cell,
                        'frame': frame,
                        'slot': slot,
                        'late_packet_count': sym_count})

    return results


# Saved these here as reference
SLOT_TIME = 500.
SYMBOL_TIME = SLOT_TIME / 14.

#4TR parameters
GH_4TR_DLC_WINDOW_OFFSET1_BFW = -470    #Note: we do not have this concept for 4TR, but just define these deadlines as the same window
GH_4TR_DLC_WINDOW_OFFSET2_BFW = -419
GH_4TR_DLC_WINDOW_OFFSET1_NONBFW = -470
GH_4TR_DLC_WINDOW_OFFSET2_NONBFW = -419
GH_4TR_DLU_WINDOW_OFFSET1 = -345
GH_4TR_DLU_WINDOW_OFFSET2 = -294
GH_4TR_ULC_WINDOW_OFFSET1_BFW = -336
GH_4TR_ULC_WINDOW_OFFSET2_BFW = -285
GH_4TR_ULC_WINDOW_OFFSET1_NONBFW = -336
GH_4TR_ULC_WINDOW_OFFSET2_NONBFW = -285
GH_4TR_ULU_WINDOW_OFFSET1 = 280
GH_4TR_ULU_WINDOW_OFFSET2 = 331
GH_4TR_SRS_WINDOW_OFFSET1 = 512
GH_4TR_SRS_WINDOW_OFFSET2 = 1831

#MMIMO parameters
GH_64TR_DLC_WINDOW_OFFSET1_BFW = -669
GH_64TR_DLC_WINDOW_OFFSET2_BFW = -419
GH_64TR_DLC_WINDOW_OFFSET1_NONBFW = -470
GH_64TR_DLC_WINDOW_OFFSET2_NONBFW = -419
GH_64TR_DLU_WINDOW_OFFSET1 = -345
GH_64TR_DLU_WINDOW_OFFSET2 = -294
GH_64TR_ULC_WINDOW_OFFSET1_BFW = -535
GH_64TR_ULC_WINDOW_OFFSET2_BFW = -285
GH_64TR_ULC_WINDOW_OFFSET1_NONBFW = -336
GH_64TR_ULC_WINDOW_OFFSET2_NONBFW = -285
GH_64TR_ULU_WINDOW_OFFSET1 = 280
GH_64TR_ULU_WINDOW_OFFSET2 = 331
GH_64TR_SRS_WINDOW_OFFSET1 = 512
GH_64TR_SRS_WINDOW_OFFSET2 = 1831

from enum import Enum
class TrafficType(Enum):
    DTT_DLC_BFW = 0
    DTT_DLC_NONBFW = 1
    DTT_DLU = 2
    DTT_ULC_BFW = 3
    DTT_ULC_NONBFW = 4
    DTT_ULU = 5
    DTT_SRS = 6

class TestCaseType(Enum):
    GH_4TR = 0
    GH_64TR = 1

window_dict = {
    TestCaseType.GH_4TR: {
                    TrafficType.DTT_DLC_BFW: (GH_4TR_DLC_WINDOW_OFFSET1_BFW,GH_4TR_DLC_WINDOW_OFFSET2_BFW),
                    TrafficType.DTT_DLC_NONBFW: (GH_4TR_DLC_WINDOW_OFFSET1_NONBFW,GH_4TR_DLC_WINDOW_OFFSET2_NONBFW),
                    TrafficType.DTT_DLU: (GH_4TR_DLU_WINDOW_OFFSET1,GH_4TR_DLU_WINDOW_OFFSET2),
                    TrafficType.DTT_ULC_BFW: (GH_4TR_ULC_WINDOW_OFFSET1_BFW,GH_4TR_ULC_WINDOW_OFFSET2_BFW),
                    TrafficType.DTT_ULC_NONBFW: (GH_4TR_ULC_WINDOW_OFFSET1_NONBFW,GH_4TR_ULC_WINDOW_OFFSET2_NONBFW),
                    TrafficType.DTT_ULU: (GH_4TR_ULU_WINDOW_OFFSET1,GH_4TR_ULU_WINDOW_OFFSET2),
                    TrafficType.DTT_SRS: (GH_4TR_SRS_WINDOW_OFFSET1,GH_4TR_SRS_WINDOW_OFFSET2)
    },
    TestCaseType.GH_64TR: {
                    TrafficType.DTT_DLC_BFW: (GH_64TR_DLC_WINDOW_OFFSET1_BFW,GH_64TR_DLC_WINDOW_OFFSET2_BFW),
                    TrafficType.DTT_DLC_NONBFW: (GH_64TR_DLC_WINDOW_OFFSET1_NONBFW,GH_64TR_DLC_WINDOW_OFFSET2_NONBFW),
                    TrafficType.DTT_DLU: (GH_64TR_DLU_WINDOW_OFFSET1,GH_64TR_DLU_WINDOW_OFFSET2),
                    TrafficType.DTT_ULC_BFW: (GH_64TR_ULC_WINDOW_OFFSET1_BFW,GH_64TR_ULC_WINDOW_OFFSET2_BFW),
                    TrafficType.DTT_ULC_NONBFW: (GH_64TR_ULC_WINDOW_OFFSET1_NONBFW,GH_64TR_ULC_WINDOW_OFFSET2_NONBFW),
                    TrafficType.DTT_ULU: (GH_64TR_ULU_WINDOW_OFFSET1,GH_64TR_ULU_WINDOW_OFFSET2),
                    TrafficType.DTT_SRS: (GH_64TR_SRS_WINDOW_OFFSET1,GH_64TR_SRS_WINDOW_OFFSET2)
    }
}

# Returns a tuple containing times relative to T0 (start,end)
def getReceptionWindow(dl_traffic_type, tc_type):
    return window_dict[tc_type][dl_traffic_type]

def getTCType(mmimo_enable):
    if(mmimo_enable):
        return TestCaseType.GH_64TR
    else:
        return TestCaseType.GH_4TR

symboltiming_pattern1 = re.compile(r"(\d+\:\d+\:\d+\.\d+) .*\[(RU|FH).SYMBOL_TIMINGS\] (.{6,8}) Cell (\d+) (.*) T0 (\d+) .* Sym 0 (\d+) (\d+) Sym 1 (\d+) (\d+) Sym 2 (\d+) (\d+) Sym 3 (\d+) (\d+) Sym 4 (\d+) (\d+) Sym 5 (\d+) (\d+) Sym 6 (\d+) (\d+) Sym 7 (\d+) (\d+) Sym 8 (\d+) (\d+) Sym 9 (\d+) (\d+) Sym 10 (\d+) (\d+) Sym 11 (\d+) (\d+) Sym 12 (\d+) (\d+) Sym 13 (\d+) (\d+)")

def parse_ru_symbol_timings(line, mmimo_enable=False):
    """
    Parses symbol timing messages from RU log line
    """

    results = []

    dlc_offset1,dlc_offset2 = getReceptionWindow(TrafficType.DTT_DLC_BFW, getTCType(mmimo_enable))
    dlu_offset1,dlu_offset2 = getReceptionWindow(TrafficType.DTT_DLU, getTCType(mmimo_enable))
    ulc_offset1,ulc_offset2 = getReceptionWindow(TrafficType.DTT_ULC_BFW, getTCType(mmimo_enable))

    found = symboltiming_pattern1.match(line)
    if found:
        log_timestamp = found[1]
        tag_name = found[2]+".SYMBOL_TIMINGS"
        slot_string = found[3]
        sfn = parse_frame(slot_string)
        slot = parse_slot(slot_string) + (sfn%NUM_20SLOT_GROUPS)*20
        cell = int(found[4])
        task = found[5]
        t0_timestamp = int(found[6])

        #At this point we have earliest/latest timestamp alternating
        earliest_timestamp = []
        latest_timestamp = []
        for ii in range(14):
            earliest_timestamp.append(int(found[7+2*ii]))
            latest_timestamp.append(int(found[7+2*ii+1]))

        for symbol in range(14):
            if(task=='DL C Plane'):
                bound1 = dlc_offset1
                bound2 = dlc_offset2
            elif(task=='DL U Plane'):
                bound1 = dlu_offset1
                bound2 = dlu_offset2
            elif(task=='UL C Plane'):
                bound1 = ulc_offset1
                bound2 = ulc_offset2
            else:
                print("WARNING :: unknown task %s in %s"%(task,"parse_ru_symbol_timings"))

            #Establish window times relative to T0 (in ns)
            window_start = (bound1 + SYMBOL_TIME*symbol)
            window_end = (bound2 + SYMBOL_TIME*symbol)

            earliest_ts = earliest_timestamp[symbol]
            latest_ts = latest_timestamp[symbol]

            if(earliest_ts != 18446744073709551615 and latest_ts != 0):
                start_deadline = (earliest_ts - t0_timestamp)/1000.
                end_deadline = (latest_ts - t0_timestamp)/1000.

                results.append({'log_timestamp': log_timestamp,
                                't0_timestamp': t0_timestamp,
                                'sfn': sfn,
                                'slot': slot,
                                'cell': cell,
                                'symbol': symbol,
                                'start_timestamp': earliest_ts,
                                'stop_timestamp': latest_ts,
                                'start_deadline': start_deadline,
                                'end_deadline': end_deadline,
                                'start_offset': start_deadline - window_start,
                                'end_offset': end_deadline - window_start,
                                'window_start_deadline': window_start,
                                'window_end_deadline': window_end,
                                'task': task
                                })
            # else:
            #     if(task in ['DL U Plane','DL C Plane']):
            #         print("%s: %i %i %i"%(task,sfn,slot,cell))


    return results

earlylate_pattern1 = re.compile(r"(\d+\:\d+\:\d+\.\d+) .* F(\d+)S(\d+)S(\d+) Cell (\d+) (.*) T0 (\d+) num pkts (\d+) max (-?\d+) min (-?\d+) early (\d+), ontime (\d+), late (\d+)")

def parse_early_late_message(ru_line):
    results = []

    found = earlylate_pattern1.match(ru_line)
    if found:
        log_timestamp = found[1]

        # Calculate the SFN.slot directly from T0.
        oran_fn = int(found[2])
        oran_sf = int(found[3])
        oran_slot = int(found[4])

        cell = int(found[5])
        task = 'RU '+found[6]
        t0_timestamp = int(found[7])
        num_pkts = int(found[8])
        start_timestamp = t0_timestamp + int(found[10])
        end_timestamp = t0_timestamp + int(found[9])
        num_early = int(found[11])
        num_ontime = int(found[12])
        num_late = int(found[13])

        sfn_t = tai_to_sfn(t0_timestamp, 0,0)
        sfn = sfn_t.sfn
        slot = sfn_t.slot+ (sfn%NUM_20SLOT_GROUPS)*20

        results.append({'t0_timestamp': t0_timestamp,
                        'log_timestamp': log_timestamp,
                        'sfn': sfn,
                        'slot': slot,
                        'oran_fn': oran_fn,
                        'oran_sf': oran_sf,
                        'oran_slot': oran_slot,
                        'task': task,
                        'cell': cell,
                        'start_timestamp': start_timestamp,
                        'end_timestamp': end_timestamp,
                        'num_pkts': num_pkts,
                        'num_early': num_early,
                        'num_ontime': num_ontime,
                        'num_late': num_late,
                        })

    return results

def add_pmu_metrics_fields(io_dict,pmu_metrics_str):
    pmu_metrics_str = pmu_metrics_str.strip().lstrip("<").rstrip(">")
    values = pmu_metrics_str.split(",")
    pmu_metrics_type = int(values[0])
    if(pmu_metrics_type == 3):
        #Cache metrics
        io_dict["l1i_miss_pki"] = float(values[1])
        io_dict["l1d_miss_pki"] = float(values[2])
        io_dict["l2d_miss_pki"] = float(values[3])
        io_dict["l3d_miss_pki"] = float(values[4])
        io_dict["mem_access_pki"] = float(values[5])
        io_dict["ipc"] = float(values[6])
    elif(pmu_metrics_type == 4):
        #Cache metrics
        io_dict["total_clock"] = int(values[1])
        io_dict["task_clock"] = int(values[2])
        io_dict["migrations"] = int(values[3])
        io_dict["context_switches"] = int(values[4])
        io_dict["total_exc"] = int(values[5])
        io_dict["cycle_count"] = int(values[6])
        io_dict["instr_count"] = int(values[7])
    else:
        print("WARNING :: pmu_metrics_type=%i not recognized"%pmu_metrics_type)

ti_re = re.compile('(\d+\:\d+\:\d+\.\d+) INF (?:([a-zA-Z0-9]+)\s*([0-9]+)\s*)?.*\[.*] {TI} <([a-zA-Z0-9 -]+),(\d+),(\d+),(\d+),(\d+)> (<.*> )?(.*)')
def parse_ti_line(line, subtask_filters=None, enable_pmu_metrics=False, just_full_task=False):
    global ti_re
    found = ti_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        if(found[2] is None):
            thread_name = ""
        else:
            thread_name = found[2]
        if(found[3] is None):
            dropped_messages = ""
        else:
            dropped_messages = int(found[3])
        task = found[4]
        sfn = int(found[5])
        slot = int(found[6]) + (sfn%NUM_20SLOT_GROUPS)*20
        map = int(found[7])
        cpu = int(found[8])

        #PMU Metric Information
        if(found[9] is None):
            pmu_metrics_str = ""
        else:
            pmu_metrics_str = found[9]

        #Parse subtask
        subtask_str = found[10]
        subtask_split = subtask_str.split(",")
        subtask_split.pop() #Remove blank data at the end
        first_start_timestamp = None
        last_end_timestamp = None
        num_sections = len(subtask_split)-1
        for ii in range(num_sections):
            subtask1,subtask_timestamp1 = subtask_split[ii].split(":")
            subtask2,subtask_timestamp2 = subtask_split[ii+1].split(":")
            start_timestamp = int(subtask_timestamp1)
            end_timestamp = int(subtask_timestamp2)

            # Grab t0 based on first subtask
            if(ii==0):
                t0_timestamp = sfn_to_tai(sfn, slot%20, start_timestamp, 0, 0)
                first_start_timestamp = start_timestamp

            if(ii==(num_sections-1)):
                last_end_timestamp = end_timestamp

            # Add separate result per subtask
            if(not just_full_task and (subtask_filters is None or (task,subtask1) in subtask_filters)):
                result.append({
                    'log_timestamp': log_timestamp,
                    'thread_name': thread_name,
                    'dropped_messages': dropped_messages,
                    't0_timestamp': t0_timestamp,
                    'task': task,
                    'sfn': sfn,
                    'slot': slot,
                    'map': map,
                    'cpu': cpu,
                    'subtask': subtask1,
                    'sequence': ii,
                    'start_timestamp': start_timestamp,
                    'end_timestamp': end_timestamp})

        # Add result for full task
        if(first_start_timestamp is not None and
           last_end_timestamp is not None and
           (subtask_filters is None or (task,'Full Task') in subtask_filters)):
            full_task_dict = {
                    'log_timestamp': log_timestamp,
                    'thread_name': thread_name,
                    'dropped_messages': dropped_messages,
                    't0_timestamp': t0_timestamp,
                    'task': task,
                    'sfn': sfn,
                    'slot': slot,
                    'map': map,
                    'cpu': cpu,
                    'subtask': 'Full Task',
                    'sequence': -1,
                    'start_timestamp': first_start_timestamp,
                    'end_timestamp': last_end_timestamp}
            if(enable_pmu_metrics):
                add_pmu_metrics_fields(full_task_dict,pmu_metrics_str)
            result.append(full_task_dict)
            
    #Add basic time fields
    for data_dict in result:
        add_basic_time_fields_to_dict(data_dict)

    return result

mti_re = re.compile('(\d+\:\d+\:\d+\.\d+) INF (?:([a-zA-Z0-9]+)\s*([0-9]+)\s*)?.*\[.*] {mTI} <([a-zA-Z0-9 -]+),(\d+),(\d+),(\d+),(\d+)> (<.*> )?(.*)')
def parse_mti_line(line, task_filters=None, enable_pmu_metrics=False):
    global mti_re
    found = mti_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        if(found[2] is None):
            thread_name = ""
        else:
            thread_name = found[2]
        if(found[3] is None):
            dropped_messages = ""
        else:
            dropped_messages = int(found[3])
        task = found[4]
        sfn = int(found[5])
        slot = int(found[6]) + (sfn%NUM_20SLOT_GROUPS)*20
        map = int(found[7])
        cpu = int(found[8])

        #PMU Metric Information
        if(found[9] is None):
            pmu_metrics_str = ""
        else:
            pmu_metrics_str = found[9]

        #Parse subtask
        subtask_str = found[10]
        subtask_str = subtask_str.rstrip(',')
        if(subtask_str.find("s:")>=0):
            subtask = "start"
        else:
            subtask = "end"
        timestamp = int(subtask_str.split(":")[1])
        t0_timestamp = sfn_to_tai(sfn, slot%20, timestamp, 0, 0)

        if(task_filters is None or task in task_filters):

            result_dict = {
                'log_timestamp': log_timestamp,
                'thread_name': thread_name,
                'dropped_messages': dropped_messages,
                't0_timestamp': t0_timestamp,
                'task': task,
                'sfn': sfn,
                'slot': slot,
                'map': map,
                'cpu': cpu,
                'subtask': subtask,
                'timestamp': timestamp}
        
            if(enable_pmu_metrics and subtask == "end"):
                add_pmu_metrics_fields(result_dict,pmu_metrics_str)

            result.append(result_dict)


    return result

tick_times_re = re.compile('(\d+\:\d+\:\d+\.\d+) I.*\[L2A.TICK_TIMES\] SFN (\d+)\.(\d+) current_time=(\d+), tick=(\d+)')
def parse_tick_times(line):
    global tick_times_re
    found = tick_times_re.match(line)
    result = []
    if found:
        #print(line)
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn%NUM_20SLOT_GROUPS)*20
        cpu_timestamp = int(found[4])
        tick_timestamp = int(found[5])

        t0_timestamp = sfn_to_tai(sfn, slot%20, tick_timestamp, 0, 0)

        result.append({'log_timestamp':log_timestamp,
                       't0_timestamp':t0_timestamp,
                       'sfn':sfn,
                       'slot':slot,
                       'cpu_timestamp': cpu_timestamp,
                       'tick_timestamp': tick_timestamp,
                       'cpu_deadline':(cpu_timestamp - t0_timestamp) / 1000.,
                       'tick_deadline':(tick_timestamp - t0_timestamp) / 1000.})
        
    return result

# example
# [test_mac]: 02:15:44.200480 WRN 3746 0 [MAC.FAPI] Cell  0 | DL 1469.14 Mbps 1400 Slots | UL  196.70 Mbps
testmac_throughput_re = re.compile('(\d+\:\d+\:\d+\.\d+) .* \[MAC.FAPI\] Cell\s*(\d+) \| DL (\d+(?:\.\d+)?) Mbps .* \| UL\s*(\d+(?:\.\d+)?) Mbps')
def parse_testmac_observed_throughput_per_second(line):
    found = testmac_throughput_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        cell_id       = found[2]
        thr_dl        = float(found[3])
        thr_ul        = float(found[4])
        result.append({
            'log_timestamp': log_timestamp,
            'cell_id'      : cell_id,
            'throughput_dl': thr_dl,
            'throughput_ul': thr_ul
        })
    return result

# example
# [test_mac]: 02:15:27.898318 WRN 3387 0 [MAC.LP] ExpectedData: Cell=0 DL=1469.138400 UL=196.699200 
testmac_expected_throughput_re = re.compile('(\d+\:\d+\:\d+\.\d+) .* \[MAC.LP\] ExpectedData: Cell=(\d+) DL=(\d+(?:\.\d+)?) UL=(\d+(?:\.\d+)?)')
def parse_testmac_expected_throughput(line):
    found = testmac_expected_throughput_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        cell_id       = found[2]
        thr_dl        = float(found[3])
        thr_ul        = float(found[4])
        result.append({
            'log_timestamp': log_timestamp,
            'cell_id'      : cell_id,
            'throughput_dl': thr_dl,
            'throughput_ul': thr_ul
        })
    return result

# example
# 04:08:32.500537 WRN 2504 0 [RU] Cell  0 DL    0.00 Mbps    0 Slots | UL    0.00 Mbps
ru_throughput_re = re.compile('(\d+\:\d+\:\d+\.\d+) .* \[RU\] Cell\s*(\d+) DL\s*(\d+(?:\.\d+)?) Mbps.*UL\s*(\d+(?:\.\d+)?) Mbps')
def parse_ru_observed_throughput_per_second(line):
    found = ru_throughput_re.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        cell_id       = found[2]
        thr_dl        = float(found[3])
        thr_ul        = float(found[4])
        result.append({
            'log_timestamp': log_timestamp,
            'cell_id'      : cell_id,
            'throughput_dl': thr_dl,
            'throughput_ul': thr_ul
        })
    return result


testmac_times_re = re.compile('(\d+\:\d+\:\d+\.\d+) I.*\[MAC.PROCESSING_TIMES\] SFN (\d+)\.(\d+) .*tick=(\d+) slot_indication=(\d+) fapi1_start=(\d+) fapi1_stop=(\d+) fapi1_count=(\d+) sleep_time=(\d+) fapi2_start=(\d+) fapi2_stop=(\d+) fapi2_count=(\d+) notify_start=(\d+) notify_stop=(\d+)')

def parse_testmac_times(line):

    global testmac_times_re
    found = testmac_times_re.match(line)
    result = []
    if found:
        #print(line)
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn%NUM_20SLOT_GROUPS)*20
        tick_timestamp = int(found[4])
        slot_indication_timestamp = int(found[5])
        fapi1_start_timestamp = int(found[6])
        fapi1_stop_timestamp = int(found[7])
        fapi1_count = int(found[8])
        sleep_time_ns = int(found[9])
        fapi2_start_timestamp = int(found[10])
        fapi2_stop_timestamp = int(found[11])
        fapi2_count = int(found[12])
        notify_start_timestamp = int(found[13])
        notify_stop_timestamp = int(found[14])

        # Calculate t0
        t0_timestamp = sfn_to_tai(sfn, slot%20, tick_timestamp, 0, 0)

        # Add separate result per subtask
        result.append({
            'log_timestamp':log_timestamp,
            't0_timestamp': t0_timestamp,
            'sfn': sfn,
            'slot': slot,
            'tick_timestamp': tick_timestamp,
            'slot_indication_timestamp': slot_indication_timestamp,
            'fapi1_start_timestamp': fapi1_start_timestamp,
            'fapi1_stop_timestamp': fapi1_stop_timestamp,
            'fapi1_count': fapi1_count,
            'sleep_time_ns': sleep_time_ns,
            'fapi2_start_timestamp': fapi2_start_timestamp,
            'fapi2_stop_timestamp': fapi2_stop_timestamp,
            'fapi2_count': fapi2_count,
            'notify_start_timestamp': notify_start_timestamp,
            'notify_stop_timestamp': notify_stop_timestamp,
            'start_timestamp': slot_indication_timestamp,
            'first_fapi_timestamp': fapi1_stop_timestamp,
            'end_timestamp': fapi2_stop_timestamp,
            })
        
    #Add basic time fields
    for data_dict in result:
        #Add deadline times for start/end
        add_basic_time_fields_to_dict(data_dict)

        #Also first fapi deadline
        data_dict['first_fapi_deadline'] = (data_dict['first_fapi_timestamp'] - data_dict['t0_timestamp'])/1000.

    return result

#L2A correction variables
ENABLE_L2A_VERIFY = False
ENABLE_L2A_CORRECTION = False
last_slot = None
last_sfn = None
first_correction_since_reset = True

l2times_re = re.compile('(\d+\:\d+\:\d+\.\d+) I.*\[L2A.PROCESSING_TIMES\] SFN (\d+)\.(\d+) .*l1_slot_ind_tick=(\d+) l2a_start_time=(\d+) l2a_end_time=(\d+) last_fapi_msg_tick=(\d+) l1_enqueue_complete_time=(\d+) UL=(\d+) DL=(\d+) CSIRS=(\d+)')

def reset_l2_slot():
    global last_slot
    global last_sfn
    global first_correction_since_reset
    last_slot = None
    last_sfn = None
    first_correction_since_reset = True

def parse_l2_times(line):
    #Note - raises an expection if this function detects bad L2A information

    global l2times_re
    global last_slot
    global last_sfn
    global first_correction_since_reset
    found = l2times_re.match(line)
    result = []
    if found:
        #print(line)
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn%NUM_20SLOT_GROUPS)*20
        l1_slot_ind_tick_timestamp = int(found[4])
        l2a_start_time_timestamp = int(found[5])
        l2a_end_time_timestamp = int(found[6])
        last_fapi_msg_tick_timestamp = int(found[7])
        l1_enqueue_complete_timestamp = int(found[8])
        ul_slot = bool(int(found[9]))
        dl_slot = bool(int(found[10]))
        csirs_slot = bool(int(found[11]))

        # Correct L2A bug
        if(last_slot is not None):
            #Determine our expected slot/frame
            expected_slot = (last_slot + 1)%(NUM_20SLOT_GROUPS*20)
            expected_sfn = last_sfn
            if(expected_slot % 20 == 0):
                expected_sfn += 1
            expected_sfn %= 1024

            if(slot != expected_slot or sfn != expected_sfn):
                if(ENABLE_L2A_VERIFY):
                    print("ERROR: L2A correction disabled and incorrect sfn.slot detected in the following line:")
                    print("ERROR: %s"%line)
                    raise Exception("Invalid L2A information detected")
                if(ENABLE_L2A_CORRECTION):
                    if(first_correction_since_reset):
                        print("WARNING: Fixing L2A bug starting at this line:")
                        print("WARNING: %s"%line)
                        first_correction_since_reset = False

                    slot = expected_slot
                    sfn = expected_sfn
        
        last_slot = slot
        last_sfn = sfn

        # Calculate t0
        t0_timestamp = sfn_to_tai(sfn, slot%20, l1_slot_ind_tick_timestamp, 0, 0)

        # Add separate result per subtask
        result.append({
            'log_timestamp':log_timestamp,
            't0_timestamp': t0_timestamp,
            'sfn': sfn,
            'slot': slot,
            'l1_slot_indication_timestamp': l1_slot_ind_tick_timestamp,
            'first_fapi_timestamp': l2a_start_time_timestamp,
            'last_fapi_timestamp': last_fapi_msg_tick_timestamp,
            'fapi_processing_complete_timestamp': l2a_end_time_timestamp,
            'l1_enqueue_complete_timestamp': l1_enqueue_complete_timestamp,
            'start_timestamp': l2a_start_time_timestamp,
            'end_timestamp': l1_enqueue_complete_timestamp,
            'ul_slot': ul_slot,
            'dl_slot': dl_slot,
            'csirs_slot': csirs_slot})
        
    #Add basic time fields
    for data_dict in result:
        add_basic_time_fields_to_dict(data_dict)

        data_dict['first_fapi_deadline'] = (data_dict['first_fapi_timestamp'] - data_dict['t0_timestamp'])/1000.
        data_dict['last_fapi_deadline'] = (data_dict['last_fapi_timestamp'] - data_dict['t0_timestamp'])/1000.

    return result

order_kernel_debug_re = re.compile('(\d+\:\d+\:\d+\.\d+) I.* \{ORDER API DEBUG\} SFN (\d+)\.(\d+) (.*)')

def parse_order_kernel_debug(line,closest_t0_timestamp):
    #Note - raises an excpetion if this function detects bad L2A information

    global order_kernel_debug_re
    found = order_kernel_debug_re.match(line)
    if found:
        #print(line)
        log_timestamp = found[1]
        sfn = int(found[2])
        slot = int(found[3]) + (sfn%NUM_20SLOT_GROUPS)*20
        data_str = found[4]

        # Calculate t0
        t0_timestamp = sfn_to_tai(sfn, slot%20, closest_t0_timestamp, 0, 0)

        # Calculate slot 4 t0
        if(slot %10 == 4):
            t0_timestamp_slot4 = t0_timestamp
        else:
            t0_timestamp_slot4 = sfn_to_tai(sfn, (slot-1)%20, closest_t0_timestamp, 0, 0)

        #Parse packet count/clock data
        call_split = data_str.split(",")

        GLOBAL_TIMER_CLOCK = 1e9
        packet_info = []
        rate_info = []
        current_gpu_time = 0.0
        for ii in range(len(call_split)):
            if(call_split[ii] != ""):
                num_packets,gpu_clock = call_split[ii].split(":")
                num_packets = int(num_packets)
                gpu_clock = int(gpu_clock)

                if(num_packets > 0):

                    if(ii > 0):
                        clock_delta = gpu_clock - last_gpu_clock
                        duration = (clock_delta / GLOBAL_TIMER_CLOCK) * 1e6
                        current_gpu_time += duration
        
                    packet_info.append({
                        'log_timestamp':log_timestamp,
                        't0_timestamp': t0_timestamp,
                        't0_timestamp_slot4': t0_timestamp_slot4,
                        'sfn': sfn,
                        'slot': slot,
                        'packet_count': num_packets,
                        'gpu_clock': gpu_clock,
                        'api_call': ii,
                        'gpu_time': current_gpu_time,
                        })
                
                    if(ii > 0):
                        rate_info.append({
                            'log_timestamp':log_timestamp,
                            't0_timestamp': t0_timestamp,
                            't0_timestamp_slot4': t0_timestamp_slot4,
                            'sfn': sfn,
                            'slot': slot,
                            'packet_count': num_packets,
                            'gpu_duration': duration,
                            'packet_rate': num_packets / duration,
                            'api_call': ii,
                            'gpu_time': current_gpu_time,
                            })
                    last_gpu_clock = gpu_clock

        return packet_info,rate_info
    else:
        return [],[]
    
def add_basic_time_fields_to_dict(data_dict):
    '''
    Function that appends basic commonly used time fields assuming input dict has:
    1.) t0_timestamp
    2.) start_timestamp
    3.) end_timestamp
    '''
    data_dict['start_deadline'] = (data_dict['start_timestamp'] - data_dict['t0_timestamp'])/1000.
    data_dict['end_deadline'] = (data_dict['end_timestamp'] - data_dict['t0_timestamp'])/1000.
    data_dict['duration'] = data_dict['end_deadline'] - data_dict['start_deadline']

def add_time_fields(df,ref_t0):
    '''
    Function that adds several fields to the dataframe assuming the input dataframe has:
    1.) t0_timestamp
    2.) start_timestamp
    3.) end_timestamp
    '''
    for ii in range(len(df)):

        # Add Time In Run (tir) field
        df[ii]['tir'] = (df[ii]['t0_timestamp'] - ref_t0) / 1e9
        df[ii]['start_tir'] = (df[ii]['start_timestamp'] - ref_t0) / 1e9
        df[ii]['end_tir'] = (df[ii]['end_timestamp'] - ref_t0) / 1e9

        # Add deadline fields (in usec)
        df[ii]['start_deadline'] = (df[ii]['start_timestamp'] - df[ii]['t0_timestamp'])/1000.
        df[ii]['end_deadline'] = (df[ii]['end_timestamp'] - df[ii]['t0_timestamp'])/1000.

        # Add duration field (in usec)
        df[ii]['duration'] = df[ii]['end_deadline'] - df[ii]['start_deadline']

        # Add datetime fields (in sec)
        df[ii]['t0_datetime'] = datetime.datetime.fromtimestamp(df[ii]['t0_timestamp']/1e9)
        df[ii]['start_datetime'] = datetime.datetime.fromtimestamp(df[ii]['start_timestamp']/1e9)
        df[ii]['end_datetime'] = datetime.datetime.fromtimestamp(df[ii]['end_timestamp']/1e9)

        # Add formatted t0
        df[ii]['t0_formatted'] = df[ii]['t0_datetime'].strftime('%Y-%m-%d %H:%M:%S.%f')

def parse_rulog(rulog, max_duration=None):
    print(f"Parsing ru log %s"%rulog)

    early_late_results = []
    lp_sym_results = []
    lp_slot_results = []
    symbol_timing_results = []
    ul_tx_timing_results = []
    ref_t0 = None

    fid = open(rulog)
    line = fid.readline()
    current_duration = 0.0
    line_idx = 0
    while(line != ""):

        # Pare early late information
        early_late_results.extend(parse_early_late_message(line))

        # Parse late packet information
        lp_sym_results.extend(parse_late_packets_symbol_level(line))

        # Parse late packet information
        lp_slot_results.extend(parse_late_packets_slot_level(line))

        # Parse symbol timing information
        symbol_timing_results.extend(parse_ru_symbol_timings(line))

        # Parse symbol timing information
        ul_tx_timing_results.extend(parse_ru_tx_times(line))

        if ref_t0 is None:
            # Set reference time if not set
            #Note: all messsages have a t0 timestamp
            if(len(lp_sym_results) > 0):
                ref_t0 = lp_sym_results[0]['t0_timestamp']
            elif(len(lp_slot_results) > 0):
                ref_t0 = lp_slot_results[0]['t0_timestamp']
            elif(len(early_late_results) > 0):
                ref_t0 = early_late_results[0]['t0_timestamp']
            elif(len(symbol_timing_results) > 0):
                ref_t0 = symbol_timing_results[0]['t0_timestamp']
            elif(len(ul_tx_timing_results) > 0):
                ref_t0 = ul_tx_timing_results[0]['t0_timestamp']

        elif max_duration is not None:
            # Determine how far we are in the run
            last_time_list = []
            if(len(lp_sym_results) > 0):
                last_time_list.append(lp_sym_results[-1]['t0_timestamp'])
            elif(len(lp_slot_results) > 0):
                last_time_list.append(lp_slot_results[-1]['t0_timestamp'])
            elif(len(early_late_results) > 0):
                last_time_list.append(early_late_results[-1]['t0_timestamp'])
            elif(len(symbol_timing_results) > 0):
                last_time_list.append(symbol_timing_results[-1]['t0_timestamp'])
            elif(len(ul_tx_timing_results) > 0):
                last_time_list.append(ul_tx_timing_results[-1]['t0_timestamp'])

            current_duration = (max(last_time_list) - ref_t0) / 1e9

        # Determine how far we are in run time
        if(max_duration is not None and current_duration > max_duration):
            break

        # Print progress periodically
        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))

        # Read new line
        line = fid.readline()

    # Add standard time fields
    common_results = [early_late_results,symbol_timing_results]
    for data in common_results:
        add_time_fields(data,ref_t0)

    # Format late packet dataframe
    early_late_df = pd.DataFrame(early_late_results)
    lp_sym_df = pd.DataFrame(lp_sym_results)
    lp_slot_df = pd.DataFrame(lp_slot_results)
    symbol_timing_df = pd.DataFrame(symbol_timing_results)
    ul_tx_timing_df = pd.DataFrame(ul_tx_timing_results)

    return (early_late_df,lp_sym_df,lp_slot_df,symbol_timing_df,ul_tx_timing_df)

def parse_phylog(phylog, max_duration=None):
    print(f"Parsing phy log %s"%phylog)

    reset_l2_slot()

    ti_results = []
    gpu_results = []
    compression_results = []
    l2times_results = []
    packet_timing_debug_results = []
    packet_timing_cache = {}
    ref_t0 = None

    fid = open(phylog)
    line = fid.readline()
    current_duration = 0.0
    line_idx = 0
    while(line != ""):
        # Parse task instrumentation messages
        ti_results.extend(parse_ti_line(line))

        # Parse GPU setup/run/duration messages
        gpu_results.extend(parse_gpu_setup_and_run(line))

        # Parse L2 times
        l2times_results.extend(parse_l2_times(line))

        # Parse compression messages (these ones are not part of the common format)
        compression_results.extend(parse_new_compression_message(line))

        # Parse DU packet send times
        packet_timing_debug_results.extend(parse_packet_timing_debug_messages(line,packet_timing_cache))

        if ref_t0 is None:
            # Set reference time if not set
            #Note: all messsages have a t0 timestamp
            if(len(ti_results) > 0):
                ref_t0 = ti_results[0]['t0_timestamp']
            elif(len(gpu_results) > 0):
                ref_t0 = gpu_results[0]['t0_timestamp']
            elif(len(l2times_results) > 0):
                ref_t0 = l2times_results[0]['t0_timestamp']
            elif(len(compression_results) > 0):
                ref_t0 = compression_results[0]['t0_timestamp']

        elif max_duration is not None:
            # Determine how far we are in the run
            last_time_list = []
            if(len(ti_results) > 0):
                last_time_list.append(ti_results[-1]['t0_timestamp'])
            elif(len(gpu_results) > 0):
                last_time_list.append(gpu_results[-1]['t0_timestamp'])
            elif(len(l2times_results) > 0):
                last_time_list.append(l2times_results[-1]['t0_timestamp'])
            elif(len(compression_results) > 0):
                last_time_list.append(compression_results[-1]['t0_timestamp'])

            current_duration = (max(last_time_list) - ref_t0) / 1e9

        # Determine how far we are in run time
        if(max_duration is not None and current_duration > max_duration):
            break

        # Print progress periodically
        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))

        line = fid.readline()

    # Add standard time fields
    common_results = [ti_results,gpu_results,l2times_results]
    for data in common_results:
        add_time_fields(data,ref_t0)


    return (pd.DataFrame(ti_results),
            pd.DataFrame(gpu_results),
            pd.DataFrame(compression_results),
            pd.DataFrame(l2times_results),
            pd.DataFrame(packet_timing_debug_results))

def parse_phylog_ti_gpu_l2(phylog, max_duration=None, subtask_filters=None, gpu_task_filters=None):
    print(f"Parsing phy log %s"%phylog)

    reset_l2_slot()

    ti_results = []
    gpu_results = []
    l2times_results = []
    ref_t0 = None

    with open(phylog) as fid:
        line = fid.readline()
        current_duration = 0.0
        line_idx = 0
        while(line != ""):
            # Parse task instrumentation messages
            ti_results.extend(parse_ti_line(line,subtask_filters=subtask_filters))

            # Parse GPU setup/run/duration messages
            gpu_results.extend(parse_gpu_setup_and_run(line,gpu_task_filters=gpu_task_filters))

            # Parse L2 times
            l2times_results.extend(parse_l2_times(line))

            if ref_t0 is None:
                # Set reference time if not set
                #Note: all messsages have a t0 timestamp
                if(len(ti_results) > 0):
                    ref_t0 = ti_results[0]['t0_timestamp']
                elif(len(gpu_results) > 0):
                    ref_t0 = gpu_results[0]['t0_timestamp']
                elif(len(l2times_results) > 0):
                    ref_t0 = l2times_results[0]['t0_timestamp']

            elif max_duration is not None:
                # Determine how far we are in the run
                last_time_list = []
                if(len(ti_results) > 0):
                    last_time_list.append(ti_results[-1]['t0_timestamp'])
                elif(len(gpu_results) > 0):
                    last_time_list.append(gpu_results[-1]['t0_timestamp'])
                elif(len(l2times_results) > 0):
                    last_time_list.append(l2times_results[-1]['t0_timestamp'])

                current_duration = (max(last_time_list) - ref_t0) / 1e9

            # Determine how far we are in run time
            if(max_duration is not None and current_duration > max_duration):
                break

            # Print progress periodically
            line_idx += 1
            if (line_idx % 100000 == 0):
                print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))

            line = fid.readline()

    # Add standard time fields
    common_results = [ti_results,gpu_results,l2times_results]
    for data in common_results:
        add_time_fields(data,ref_t0)


    return (pd.DataFrame(ti_results),
            pd.DataFrame(gpu_results),
            pd.DataFrame(l2times_results))

def parse_phylog_ti_l2_only(phylog, max_duration=None, subtask_filters=None):
    print(f"Parsing phy log %s"%phylog)

    reset_l2_slot()

    ti_results = []
    l2times_results = []
    ref_t0 = None

    fid = open(phylog)
    line = fid.readline()
    current_duration = 0.0
    line_idx = 0
    while(line != ""):
        # Parse task instrumentation messages
        ti_results.extend(parse_ti_line(line,subtask_filters=subtask_filters))

        # Parse L2 times
        l2times_results.extend(parse_l2_times(line))

        if ref_t0 is None:
            # Set reference time if not set
            #Note: all messsages have a t0 timestamp
            if(len(ti_results) > 0):
                ref_t0 = ti_results[0]['t0_timestamp']
            elif(len(l2times_results) > 0):
                ref_t0 = l2times_results[0]['t0_timestamp']

        elif max_duration is not None:
            # Determine how far we are in the run
            last_time_list = []
            if(len(ti_results) > 0):
                last_time_list.append(ti_results[-1]['t0_timestamp'])
            elif(len(l2times_results) > 0):
                last_time_list.append(l2times_results[-1]['t0_timestamp'])

            current_duration = (max(last_time_list) - ref_t0) / 1e9

        # Determine how far we are in run time
        if(max_duration is not None and current_duration > max_duration):
            break

        # Print progress periodically
        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))

        line = fid.readline()

    # Add standard time fields
    common_results = [ti_results,l2times_results]
    for data in common_results:
        add_time_fields(data,ref_t0)

    return (pd.DataFrame(ti_results),
            pd.DataFrame(l2times_results))

def parse_phylog_tick_times(phylog, max_duration=None, max_lines=None):
    print(f"Parsing phy log %s"%phylog)
    results = []
    ref_t0 = None

    fid = open(phylog)
    line = fid.readline()
    line_idx = 0
    current_duration = 0.0
    while(line != ""):
        # Parse task instrumentation messages
        results.extend(parse_tick_times(line))

        if ref_t0 is None and len(results) > 0:
            ref_t0 = results[0]['t0_timestamp']

        #Check our progress
        if(len(results) > 0):
            current_duration = (results[-1]['t0_timestamp'] - ref_t0) / 1e9
            if(max_duration is not None and current_duration > max_duration):
                break

        # Print progress periodically
        line_idx += 1
        if (line_idx % 100000 == 0):
            if(max_duration is not None):
                print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))
            elif(max_lines is not None):
                print("Processed line %i of %i..."%(line_idx,max_lines))
            else:
                print("Processed line %i..."%line_idx)

        if(max_lines is not None and line_idx > max_lines):
            break

        line = fid.readline()

    return (pd.DataFrame(results))

def parse_phylog_order_kernel_debug(phylog, max_duration=None):
    print(f"Parsing phy log for order kernel debug messages %s"%phylog)

    packet_results = []
    rate_results = []
    ref_t0 = None

    fid = open(phylog)
    line = fid.readline()
    current_duration = 0.0
    line_idx = 0
    last_t0_timestamp = None
    while(line != ""):
        # Parse TI messages just to get cpu times
        ti_result = parse_ti_line(line)
        if(len(ti_result) > 0):
            last_t0_timestamp = ti_result[-1]['t0_timestamp']

        # Parse task instrumentation messages
        if(last_t0_timestamp):
            packet_result,rate_result = parse_order_kernel_debug(line,last_t0_timestamp)
            packet_results.extend(packet_result)
            rate_results.extend(rate_result)

        if ref_t0 is None:
            # Set reference time if not set
            #Note: all messsages have a t0 timestamp
            if(len(packet_results) > 0):
                ref_t0 = packet_results[0]['t0_timestamp']

        elif max_duration is not None:
            # Determine how far we are in the run
            last_time_list = []
            if(len(packet_results) > 0):
                last_time_list.append(packet_results[-1]['t0_timestamp'])

            current_duration = (max(last_time_list) - ref_t0) / 1e9

        # Determine how far we are in run time
        if(max_duration is not None and current_duration > max_duration):
            break

        # Print progress periodically
        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))

        line = fid.readline()

    tir_results = [packet_results,rate_results]
    for res in tir_results:
        for aa in res:
            aa['tir'] = (aa['t0_timestamp'] - ref_t0) / 1e9
            aa['tir_slot4'] = (aa['t0_timestamp_slot4'] - ref_t0) / 1e9
            aa['tir_gpu'] = aa['tir'] + 331e-6 + aa['gpu_time']*1e-6

    return (pd.DataFrame(packet_results),pd.DataFrame(rate_results))

# Saving these patterns here in case we add communications task messages back in
#Pattern 2 - Parse MAP_DL "DL Communication Tasks" messages
old_pattern2 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV\.MAP_DL\] \[PHYDRV\] SFN (\d+)\.(\d+) DL Communication Tasks .* Map (\d+) .* \[(.*)\] .* START: (\d+) END: (\d+) .* \[(.*)\] .* START: (\d+) END: (\d+) .* \[(.*)\] .* START: (\d+) END: (\d+) .* \[(.*)\] .* START: (\d+) END: (\d+)")
#Pattern 3 - Parse MAP_DL "UL Communication Tasks" messages
old_pattern3 = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[DRV\.MAP_UL\] \[PHYDRV\] SFN (\d+)\.(\d+) UL Communication Tasks Cell .* Map (\d+) .* \[(.*)\] .* START: (\d+) END: (\d+)")

def parse_dlu_late_symbol_delay_symbol_level(rulog, max_duration=None):
    """
    Function that only parse symbol timing information

    The information is aggregated and summarized, avoiding large dataframes.  Calculates all of the latest packet delays
     (defined at latest packet time - start of window time).  Returns the percentile information as a function of slot.
    """

    print(f"Parsing ru log %s"%rulog)

    # Keep track of delays as function of slot
    PREALLOCATE_SIZE = 600000*14*8 #8 cells, 14 symbols, 600k slots
    slot_df = pd.DataFrame(index=range(PREALLOCATE_SIZE),columns=['slot','symbol','cell','earliest_delay','latest_delay'],dtype=int)

    fid = open(rulog)
    line = fid.readline()
    line_idx = 0
    ref_t0 = None

    current_duration = 0.0
    current_row = 0
    while(line != ""):
        data = parse_ru_symbol_timings(line)
        if(len(data) > 0 and data[0]['task'] == 'DL U Plane'):
            if(ref_t0 is None):
                ref_t0 = data[0]['t0_timestamp']

            for dd in data:
                # Append data to output df
                slot_df.loc[current_row] = [dd['slot'],dd['symbol'],dd['cell'],dd['earliest_delay'],dd['latest_delay']]
                current_row += 1

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9

            if(max_duration is not None and current_duration > max_duration):
                break

        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))
        line = fid.readline()

    slot_df.drop(index=range(current_row,len(slot_df)),inplace=True)

    return slot_df

def parse_dlu_late_symbol_delay_slot_level(rulog, max_duration=None):
    """
    Function that only parse symbol timing information

    The information is aggregated and summarized, avoiding large dataframes.  Calculates all of the latest packet delays
     (defined at latest packet time - start of window time).  Returns the percentile information as a function of slot.
    """

    print(f"Parsing ru log %s"%rulog)

    # Keep track of delays as function of slot
    PREALLOCATE_SIZE = 600000*8 #8 cells, 600k slots
    slot_df = pd.DataFrame(index=range(PREALLOCATE_SIZE),columns=['slot','cell','max_earliest_delay','max_latest_delay'],dtype=int)

    fid = open(rulog)
    line = fid.readline()
    line_idx = 0
    ref_t0 = None

    current_duration = 0.0
    current_row = 0
    while(line != ""):
        data = parse_ru_symbol_timings(line)
        if(len(data) > 0 and data[0]['task'] == 'DL U Plane'):
            if(ref_t0 is None):
                ref_t0 = data[0]['t0_timestamp']

            #Note: each time we parse we have an entry for every symbol
            earliest_delays = [aa['earliest_delay'] for aa in data]
            latest_delays = [aa['latest_delay'] for aa in data]
            max_earliest_delay = max(earliest_delays)
            max_latest_delay = max(latest_delays)
            slot = data[0]['slot']
            cell = data[0]['cell']

            # Append data to output df
            slot_df.loc[current_row] = [slot,cell,max_earliest_delay,max_latest_delay]
            current_row += 1

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9

            if(max_duration is not None and current_duration > max_duration):
                break

        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))
        line = fid.readline()

    slot_df.drop(index=range(current_row,len(slot_df)),inplace=True)

    return slot_df

def parse_dlu_late_packet_symbol_level(rulog, max_duration=None):
    """
    Function that only parse late packet messages

    Results are output in raw dataframe format at symbol level
    """

    print(f"Parsing ru log %s"%rulog)

    # Keep track of delays as function of slot
    PREALLOCATE_SIZE = 600000*14*8 #8 cells, 14 symbols, 600k slots
    slot_df = pd.DataFrame(index=range(PREALLOCATE_SIZE),columns=['slot','symbol','cell','late_packet_count'],dtype=int)

    fid = open(rulog)
    line = fid.readline()
    line_idx = 0
    ref_t0 = None

    current_duration = 0.0
    current_row = 0
    while(line != ""):
        data = parse_late_packets_symbol_level(line)
        if(len(data) > 0 and data[0]['task'] == 'DL U Plane'):
            if(ref_t0 is None):
                ref_t0 = data[0]['t0_timestamp']

            for dd in data:
                slot = dd['slot']
                symbol = dd['symbol']
                cell = dd['cell']
                late_packet_count = dd['late_packet_count']

                # Append data to output df
                slot_df.loc[current_row] = [slot,symbol,cell,late_packet_count]
                current_row += 1

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9

            if(max_duration is not None and current_duration > max_duration):
                break

        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))
        line = fid.readline()

    slot_df.drop(index=range(current_row,len(slot_df)),inplace=True)

    return slot_df

def parse_dlu_late_packet_percentages(rulog, task='DL U Plane', max_duration=None, warmup_slots=0):
    """
    Function that only parses late packet messages

    The information is aggregated and summarized, so this function is useful for processing large files without
     having to return a huge dataframe with all of the information
    """

    print(f"Parsing ru log %s"%rulog)

    # Keep track of total counts and late counts
    total_counts_slot = {} #Indexed by slot
    late_counts_slot = {}

    total_counts_slot_symbol = {}
    late_counts_slot_symbol = {}

    ref_t0 = None

    fid = open(rulog)
    line = fid.readline()
    current_duration = 0.0
    line_idx = 0
    while(line != ""):
        # Parse slot-level data
        data = parse_late_packets_slot_level(line)
        valid_data = (len(data) > 0 and data[0]['task'] == task)

        # Check if duration limit has been hit
        if(valid_data):
            if(ref_t0 is None):
                ref_t0 = data[0]['t0_timestamp']

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9
            if(current_duration > max_duration):
                break

            current_diff = (data[0]['t0_timestamp'] - ref_t0)

            current_slot = int(round(current_diff / (SLOT_TIME*1000)))

        # Increment counts if we are past warmup slots and this is DLU
        if(valid_data and current_slot > warmup_slots):
            for dd in data:
                slot = dd['slot']
                cell = dd['cell']
                late_packet_count = dd['late_packet_count']

                # Add slot as key if not already there
                if(slot not in total_counts_slot.keys()):
                    total_counts_slot[slot] = 0
                    late_counts_slot[slot] = 0

                # Add to counters
                total_counts_slot[slot] += 1
                if(late_packet_count > 0):
                    late_counts_slot[slot] += 1

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9
            if(current_duration > max_duration):
                break

        # Parse symbol-level data
        data = parse_late_packets_symbol_level(line)
        valid_data = (len(data) > 0 and data[0]['task'] == task)

        # Check if duration limit has been hit
        if(valid_data):
            if(ref_t0 is None):
                ref_t0 = data[0]['t0_timestamp']

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9
            if(current_duration > max_duration):
                break

            current_slot = int(round(current_duration / (SLOT_TIME*1e-6)))

        # Increment counts if we are past warmup slots and this is DLU
        if(valid_data and current_slot > warmup_slots):
            for dd in data:
                slot = dd['slot']
                symbol = dd['symbol']
                cell = dd['cell']
                late_packet_count = dd['late_packet_count']

                # Add symbol if not already in map
                key = (slot,symbol)
                if(key not in total_counts_slot_symbol.keys()):
                    total_counts_slot_symbol[key] = 0
                    late_counts_slot_symbol[key] = 0

                # Add to counters
                total_counts_slot_symbol[key] += 1
                if(late_packet_count > 0):
                    late_counts_slot_symbol[key] += 1

        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))
        line = fid.readline()

    #Summarize slot level data and create dataframe
    slot_list = list(total_counts_slot.keys())
    slot_list.sort()
    late_counts_list = [late_counts_slot[aa] for aa in slot_list]
    total_counts_list = [total_counts_slot[aa] for aa in slot_list]
    late_slot_percentages = [100.0*aa/bb for aa,bb in zip(late_counts_list,total_counts_list)]
    slot_df = pd.DataFrame({'slot': slot_list,
                            'late_count': late_counts_list,
                            'total_count': total_counts_list,
                            'late_slot_percentage': late_slot_percentages})

    #Summary symbol level data and create dataframe
    keys = list(total_counts_slot_symbol.keys())
    keys.sort()
    slot_list = [aa[0] for aa in keys]
    symbol_list = [aa[1] for aa in keys]
    late_counts_list = [late_counts_slot_symbol[(aa,bb)] for aa,bb in zip(slot_list,symbol_list)]
    total_counts_list = [total_counts_slot_symbol[(aa,bb)] for aa,bb in zip(slot_list,symbol_list)]
    late_symbol_percentages = [100.0*aa/bb for aa,bb in zip(late_counts_list,total_counts_list)]
    symbol_df = pd.DataFrame({'slot': slot_list,
                              'symbol': symbol_list,
                              'late_count': late_counts_list,
                              'total_count': total_counts_list,
                              'late_symbol_percentage': late_symbol_percentages})

    return slot_df,symbol_df





def parse_dlu_late_packet_percentages2(rulog, task='DL U Plane', max_duration=None, warmup_slots=0):
    """
    Function that only parses late packet messages

    The information is aggregated and summarized, so this function is useful for processing large files without
     having to return a huge dataframe with all of the information
    """

    print(f"Parsing ru log %s"%rulog)

    # Keep track of total counts and late counts
    total_counts_slot = {} #Indexed by (cell,slot)
    late_counts_slot = {}

    total_counts_slot_symbol = {}
    late_counts_slot_symbol = {}

    ref_t0 = None

    fid = open(rulog)
    line = fid.readline()
    current_duration = 0.0
    line_idx = 0
    while(line != ""):
        # Parse slot-level data
        data = parse_late_packets_slot_level(line)
        valid_data = (len(data) > 0 and data[0]['task'] == task)

        # Check if duration limit has been hit
        if(valid_data):
            if(ref_t0 is None):
                ref_t0 = data[0]['t0_timestamp']

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9
            if(current_duration > max_duration):
                break

            current_diff = (data[0]['t0_timestamp'] - ref_t0)

            current_slot = int(round(current_diff / (SLOT_TIME*1000)))

        # Increment counts if we are past warmup slots and this is DLU
        if(valid_data and current_slot > warmup_slots):
            for dd in data:
                slot = dd['slot']
                cell = dd['cell']
                late_packet_count = dd['late_packet_count']

                # Add slot as key if not already there
                key = (cell,slot)
                if(key not in total_counts_slot.keys()):
                    total_counts_slot[key] = 0
                    late_counts_slot[key] = 0

                # Add to counters
                total_counts_slot[key] += 1
                if(late_packet_count > 0):
                    late_counts_slot[key] += 1

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9
            if(current_duration > max_duration):
                break

        # Parse symbol-level data
        data = parse_late_packets_symbol_level(line)
        valid_data = (len(data) > 0 and data[0]['task'] == task)

        # Check if duration limit has been hit
        if(valid_data):
            if(ref_t0 is None):
                ref_t0 = data[0]['t0_timestamp']

            current_duration = (data[0]['t0_timestamp'] - ref_t0)/1e9
            if(current_duration > max_duration):
                break

            current_slot = int(round(current_duration / (SLOT_TIME*1e-6)))

        # Increment counts if we are past warmup slots and this is DLU
        if(valid_data and current_slot > warmup_slots):
            for dd in data:
                slot = dd['slot']
                symbol = dd['symbol']
                cell = dd['cell']
                late_packet_count = dd['late_packet_count']

                # Add symbol if not already in map
                key = (cell,slot,symbol)
                if(key not in total_counts_slot_symbol.keys()):
                    total_counts_slot_symbol[key] = 0
                    late_counts_slot_symbol[key] = 0

                # Add to counters
                total_counts_slot_symbol[key] += 1
                if(late_packet_count > 0):
                    late_counts_slot_symbol[key] += 1

        line_idx += 1
        if (line_idx % 100000 == 0):
            print("Processed line %i (%.1f of %.1fsec)..."%(line_idx,current_duration,max_duration))
        line = fid.readline()

    #Summarize slot level data and create dataframe
    key_list = list(total_counts_slot.keys())
    key_list.sort()
    cell_list = [aa[0] for aa in key_list]
    slot_list = [aa[1] for aa in key_list]
    late_counts_list = [late_counts_slot[aa] for aa in key_list]
    total_counts_list = [total_counts_slot[aa] for aa in key_list]
    late_slot_percentages = [100.0*aa/bb for aa,bb in zip(late_counts_list,total_counts_list)]
    slot_df = pd.DataFrame({'cell': cell_list,
                            'slot': slot_list,
                            'late_count': late_counts_list,
                            'total_count': total_counts_list,
                            'late_slot_percentage': late_slot_percentages})

    #Summary symbol level data and create dataframe
    key_list = list(total_counts_slot_symbol.keys())
    key_list.sort()
    cell_list = [aa[0] for aa in key_list]
    slot_list = [aa[1] for aa in key_list]
    symbol_list = [aa[2] for aa in key_list]
    late_counts_list = [late_counts_slot_symbol[(aa,bb,cc)] for aa,bb,cc in zip(cell_list,slot_list,symbol_list)]
    total_counts_list = [total_counts_slot_symbol[(aa,bb,cc)] for aa,bb,cc in zip(cell_list,slot_list,symbol_list)]
    late_symbol_percentages = [100.0*aa/bb for aa,bb in zip(late_counts_list,total_counts_list)]
    symbol_df = pd.DataFrame({'cell': cell_list,
                              'slot': slot_list,
                              'symbol': symbol_list,
                              'late_count': late_counts_list,
                              'total_count': total_counts_list,
                              'late_symbol_percentage': late_symbol_percentages})

    return slot_df,symbol_df

def get_num_tasks(df_ti, task_prefix):
    task_names = list(set(df_ti[df_ti.task.str.startswith(task_prefix)].task))
    num_tasks = len(task_names)
    return num_tasks

#Determines the number of DL Tasks allocated for C-Plane.  If 0, this means that this is a legacy log from before c-plane task splitting
def get_num_dlc_tasks(df_ti):

    return get_num_tasks(df_ti, "DL Task C-Plane")

#Determines the number of UL Tasks allocated for C-Plane.  If 0, this means that this is a legacy log from before c-plane task splitting
def get_num_ulc_tasks(df_ti):

    return get_num_tasks(df_ti, "UL Task CPlane")

def get_num_gpu_comms_prepare_tasks(df_ti):

    return get_num_tasks(df_ti, "DL Task GPU Comms Prepare")

#Returns the first t0 found in the ti messages
def get_ref_t0(phylog,max_duration=0.1,num_proc=1,include_ti_mti_check=False):
    print("Determining reference t0...")
    p1 = Parsenator(phylog,
                    [parse_ti_line,parse_mti_line],
                    [['t0_timestamp'],['t0_timestamp']],
                    0,max_duration,num_processes=num_proc)
    parsed_result = p1.parse()
    df_ti = parsed_result[0]
    df_mti = parsed_result[1]

    #Note: might be better to switch this function to easy_parse (limiting number of line rather than dependent on messages existing in log)
    # easy_parse(input_log,parser_list,max_lines=None)

    if(len(df_ti) > 0):
        ref_t0 = df_ti['t0_timestamp'].min()
    elif(len(df_mti) > 0):
        ref_t0 = df_mti['t0_timestamp'].min()
    else:
        print("WARNING :: no df_ti messages found in log, cannot determine reference t0 in get_ref_t0")
        ref_t0 = None

    if(include_ti_mti_check):
        return ref_t0,len(df_ti)>0,len(df_mti)>0
    else:
        return ref_t0

def get_first_t0(phylog,max_lines=100000):
    parsed_result = easy_parse(phylog,[parse_ti_line,parse_mti_line],max_lines=max_lines)
    df_ti = parsed_result[0]
    df_mti = parsed_result[1]

    if(len(df_ti) > 0):
        return df_ti['t0_timestamp'].min()
    elif(len(df_mti) > 0):
        return df_mti['t0_timestamp'].min()
    else:
        print("WARNING :: no df_ti messages found in log, cannot determine first t0 in get_first_t0")
        return None
    
def get_last_t0(phylog,max_lines=100000):
    os.system("tail -n %i %s > /tmp/phy_cut.log"%(max_lines,phylog))
    parsed_result = easy_parse("/tmp/phy_cut.log",[parse_ti_line,parse_mti_line],max_lines=max_lines)
    df_ti = parsed_result[0]
    df_mti = parsed_result[1]

    if(len(df_ti) > 0):
        return df_ti['t0_timestamp'].max()
    elif(len(df_mti) > 0):
        return df_mti['t0_timestamp'].max()
    else:
        print("WARNING :: no df_ti messages found in log, cannot determine last t0 in get_last_t0")
        return None
