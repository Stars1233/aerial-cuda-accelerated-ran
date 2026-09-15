#!/usr/bin/python3 -u

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

import os
import sys
import re
import math
import copy
import time
import subprocess
import argparse
import os.path
import enum

log_path = os.getenv('LOG_PATH')
if log_path is None:
    cubb_sdk = os.getenv('cuBB_SDK')
    if cubb_sdk is not None:
        log_path = cubb_sdk + "/logs"

if log_path is None:
    print("Please set LOG_PATH or cuBB_SDK first")
    exit(0)

# ---------------------------------------------------------------------------
# Argument parsing with backward compatibility.
#
# Supported invocations:
#   check_result_cumcp.py <duration>
#   check_result_cumcp.py <duration> [mac.log] [ru.log]                (legacy)
#   check_result_cumcp.py <duration> --mac-log FILE --ru-log FILE
#   check_result_cumcp.py <duration> --cumcp-log FILE [--mac-log FILE]
#   check_result_cumcp.py <duration> --cumcp-sa --cumcp-log FILE       (SA mode)
#
# cumcp_sa_mode ("standalone CUMCP"):
#   False -> MAC log carries legacy [MAC.FAPI] throughput. When --cumcp-log is
#            also given, MAC log ALSO carries [CUMAC.HANDLER] throughput, and
#            BOTH are checked.
#   True  -> MAC log carries ONLY [CUMAC.HANDLER] throughput (no FAPI). Only
#            CUMAC throughput is checked on the MAC side.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Check throughput result from MAC, RU emulator and/or CUMCP logs.",
    usage="%(prog)s <duration> [mac.log] [ru.log] "
          "[--mac-log FILE] [--ru-log FILE] [--cumcp-log FILE] [--cumcp-sa]",
)
parser.add_argument("duration", type=int, help="Test duration in seconds")
parser.add_argument("mac_log_pos", nargs='?', default=None,
                    help="Positional MAC log (legacy)")
parser.add_argument("ru_log_pos", nargs='?', default=None,
                    help="Positional RU log (legacy)")
parser.add_argument("--mac-log", dest="mac_log", default=None,
                    help="Path to MAC log file")
parser.add_argument("--ru-log", dest="ru_log", default=None,
                    help="Path to RU emulator log file")
parser.add_argument("--cumcp-log", dest="cumcp_log", default=None,
                    help="Path to CUMCP log file. Enables cumcp checking.")
parser.add_argument("--cumcp-sa", dest="cumcp_sa", action="store_true", default=False,
                    help="Standalone CUMCP mode: MAC log contains only CUMAC throughput (no FAPI).")

args = parser.parse_args()

duration = args.duration

mac_log_file = args.mac_log if args.mac_log is not None else args.mac_log_pos
ru_log_file = args.ru_log if args.ru_log is not None else args.ru_log_pos
cumcp_log_file = args.cumcp_log
cumcp_sa_mode = args.cumcp_sa

# MAC log is always required (it carries the test init/expected data).
if mac_log_file is None:
    mac_log_file = log_path + "/screenlog_mac.log"

# What kind of throughput lines the MAC log carries:
#   check_fapi_mac -> [MAC.FAPI] Cell X | DL ... UL ... lines (legacy testMAC)
#   check_cumac_mac -> [CUMAC.HANDLER] Cell X | CUMAC ... lines
#
# Three supported cases:
#   (A) cumcp_log provided AND NOT cumcp_sa_mode -> both FAPI and CUMAC in MAC log
#   (B) cumcp_log provided AND     cumcp_sa_mode -> only CUMAC in MAC log
#   (C) cumcp_log NOT provided                   -> only FAPI in MAC log (legacy)
check_fapi_mac = (not cumcp_sa_mode)
check_cumac_mac = (cumcp_log_file is not None)

result_log_file = log_path + "/check_result.log"

# Different tests take longer to startup.
# F08_F_6C_35 - 3 minutes
# F08_F_9C_33 - 5 minutes
# For now - hardcode to 5 minutes (longest of all) - need to make this
# a parameter so we can still optimize for early failures on other tests
# (i.e. error out at 90 seconds instead of 300 for the others)
wait_for_startup_mac = 500
wait_for_throughput = 300

# Allowed error
CONFIG_ALLOWED_ERROR = 0.03

# Allowed continuously fail times and total fail times
CONFIG_CONTINUOUS_FAIL_TIME = 5
CONFIG_TOTAL_FAIL_TIME = 10

negative_test = 0
skip_pusch_slots_check = False  # Works for PUSCH and disables tput checking too
expected_zero = True  # True until we see a non-zero expectation for any cell
# Add 60 seconds to duration for timeout when the expected slot/data is exactly zero.
zero_timeout = duration + 60

# Print logs to both console and file
log_print_file = open(result_log_file, "w+")


def log_print(log_string):
    print(log_string)
    log_print_file.write(log_string)
    log_print_file.write("\n")
    log_print_file.flush()


if (not os.getenv("TRY") is None) and os.getenv("TRY") == "1":
    log_print("Try test, don't exit when get check result error")
    CONFIG_CONTINUOUS_FAIL_TIME = 100000
    CONFIG_TOTAL_FAIL_TIME = 100000

if (os.getenv("L2SA_TEST") == "1"):
    l2sa_test = 1
else:
    l2sa_test = 0

# Print parameters
log_print("===============================================")
log_print("duration=%d allowed_error=%.2f l2sa_test=%d cumcp_sa_mode=%d" %
          (duration, CONFIG_ALLOWED_ERROR, l2sa_test, int(cumcp_sa_mode)))
log_print("mac_log=" + str(mac_log_file))
log_print("ru_log=" + str(ru_log_file))
log_print("cumcp_log=" + str(cumcp_log_file))
log_print("check_fapi_mac=%d check_cumac_mac=%d" % (int(check_fapi_mac), int(check_cumac_mac)))
log_print("result_log=" + result_log_file)

# All channel index, sync with channel_type_t in testMAC/common_defines.hpp
CHANNEL_ID_PUSCH = 0
CHANNEL_ID_PDSCH = 1
CHANNEL_ID_PDCCH_UL = 2
CHANNEL_ID_PDCCH_DL = 3
CHANNEL_ID_PBCH = 4
CHANNEL_ID_PUCCH = 5
CHANNEL_ID_PRACH = 6
CHANNEL_ID_CSI_RS = 7
CHANNEL_ID_SRS = 8
CHANNEL_ID_MAX = 9

# Sync to channel_type_t in testMAC/common_defines.hpp
channel_names = [
    "PUSCH",  # 0
    "PDSCH",  # 1,
    "PDCCH_UL",  # 2,
    "PDCCH_DL",  # 3,
    "PBCH",  # 4,
    "PUCCH",  # 5,
    "PRACH",  # 6,
    "CSI_RS",  # 7,
    "SRS",  # 8,
    "CHANNEL_MAX",  # 9
]


# Error bit layout for [MAC.FAPI] / legacy testMAC throughput parsing.
class MacErr(enum.IntEnum):
    UL_Slots = 0  # PUSCH slots per second error
    DL_Slots = 1  # PDSCH slots per second error
    Prmb = 2  # PRACH preamble number error
    HARQ = 3  # HARQ number error
    SR = 4  # SR number error
    CSI1 = 5  # CSI Part 1 number error
    CSI2 = 6  # CSI Part 2 number error
    SRS = 7  # SRS number error
    ERR_IND = 8  # ERR.indication FAPI message received
    UL_Mbps = 9  # PUSCH data rate error
    DL_Mbps = 10  # PDSCH data rate error
    Invalid = 11  # MAC FAPI validation mismatches
    TimeStamp = 12  # Times stamp is not updating in MAC console output, traffic stopped or process frozen
    BitMAX = 13  # MAX bit number, not an error


class RuErr(enum.IntEnum):
    UL_Slots = 0  # PUSCH slots per second error
    DL_Slots = 1  # PDSCH slots per second error
    PDCCH_UL = 2  # PDCCH_UL slots per second error
    PDCCH_DL = 3  # PDCCH_DL slots per second error
    PBCH = 4  # PBCH slots per second error
    PUCCH = 5  # PUCCH slots per second error
    PRACH = 6  # PRACH slots per second error
    CSI_RS = 7  # CSI_RS slots per second error
    SRS = 8  # SRS slots per second error
    UL_Mbps = 9  # PUSCH data rate error
    DL_Mbps = 10  # PDSCH data rate error
    Bit11 = 11  # Not used
    TimeStamp = 12  # Times stamp is not updating in RU console output, traffic stopped or process frozen.
    BitMAX = 13  # MAX bit number, not an error


# Error bit layout for [CUMAC.HANDLER] line parsing (test_mac/test_cumac CUMAC throughput).
class CuMacErr(enum.IntEnum):
    ERR_IND = 0  # ERR.indication FAPI message received
    Slots = 1  # CUMAC slots per second error
    Invalid = 2  # MAC FAPI validation mismatches
    TimeStamp = 3  # Times stamp is not updating in MAC console output, traffic stopped or process frozen
    UE_SEL = 4  # UE_SEL error
    PRB_ALLOC = 5  # PRB_ALLOC error
    LAYER_SEL = 6  # LAYER_SEL error
    MCS_SEL = 7  # MCS_SEL error
    PFM_SORT = 8  # PFM_SORT error
    MU_UE_GRP = 9  # MU-MIMO UE grouping (muUeGrp) error
    BitMAX = 10  # MAX bit number, not an error


class CumcpErr(enum.IntEnum):
    ERR_IND = 0  # ERR.indication FAPI message received
    Slots = 1  # CUMAC slots per second error
    TimeStamp = 2  # Times stamp is not updating in MAC console output, traffic stopped or process frozen
    UE_SEL = 3  # UE_SEL error
    PRB_ALLOC = 4  # PRB_ALLOC error
    LAYER_SEL = 5  # LAYER_SEL error
    MCS_SEL = 6  # MCS_SEL error
    PFM_SORT = 7  # PFM_SORT error
    MU_UE_GRP = 8  # MU-MIMO UE grouping (muUeGrp) error
    BitMAX = 9  # MAX bit number, not an error


def _err_to_string(prefix, enum_cls, value):
    s = prefix + ":[" + hex(value)
    for i in range(0, enum_cls.BitMAX.value):
        if (value & (1 << i) != 0):
            s += " " + enum_cls(i).name
    s += "]"
    return s


def fapi_err_to_string(value):
    return _err_to_string("MAC", MacErr, value)


def cumac_err_to_string(value):
    return _err_to_string("CUMAC", CuMacErr, value)


def ru_err_to_string(value):
    return _err_to_string("RU", RuErr, value)


def cumcp_err_to_string(value):
    return _err_to_string("CUMCP", CumcpErr, value)


# Throughput data class for one cell. Holds the union of fields for all modes
# so that the same object can be used as "expected", "current" or "last" for
# FAPI, CUMAC (MAC-side), RU emulator and CUMCP checking.
class Thrput:

    def __init__(self, cell_id):
        self.cell_id = cell_id
        # [MAC.FAPI] / RU emulator fields
        self.dl_thrput = 0
        self.ul_thrput = 0
        self.mac_ul_thrput = 0  # Only used in expected result object
        self.mac_ul_drop = 0  # Only used in expected result object
        self.prmb = 0
        self.sr = 0
        self.harq = 0
        self.csi1 = 0
        self.csi2 = 0
        self.error = 0
        self.invalid = 0
        self.slots = [0 for ch in range(CHANNEL_ID_MAX)]
        # [CUMAC.HANDLER] / CUMCP fields
        self.cumac_slots = 0
        self.cumac_invalid = 0  # CUMAC handler "INV" counter (distinct from FAPI self.invalid)
        self.ue_sel = 0
        self.prb_alloc = 0
        self.layer_sel = 0
        self.mcs_sel = 0
        self.pfm_sort = 0
        self.mu_ue_grp = 0
        self.timestamp = ""

    def is_all_zero(self):
        if (self.dl_thrput != 0):
            return False
        if (self.ul_thrput != 0):
            return False
        if (self.prmb != 0):
            return False
        if (self.sr != 0):
            return False
        if (self.harq != 0):
            return False
        if (self.csi1 != 0):
            return False
        if (self.csi2 != 0):
            return False
        if (self.error != 0):
            return False
        for ch in range(0, CHANNEL_ID_MAX):
            if (self.slots[ch] != 0):
                return False
        if (self.cumac_slots != 0):
            return False
        if (self.cumac_invalid != 0):
            return False
        return True

    # Multiply (1 - err) and round down
    def set_low_limit(self, err):
        rate = 1 - err
        self.ul_thrput *= rate
        self.dl_thrput *= rate
        self.mac_ul_thrput *= rate
        self.ul_thrput = float(math.floor(self.ul_thrput * 100)) / 100
        self.dl_thrput = float(math.floor(self.dl_thrput * 100)) / 100
        self.mac_ul_thrput = float(math.floor(self.mac_ul_thrput * 100)) / 100
        self.prmb = math.floor(self.prmb * rate)
        self.sr = math.floor(self.sr * rate)
        self.harq = math.floor(self.harq * rate)
        self.csi1 = math.floor(self.csi1 * rate)
        self.csi2 = math.floor(self.csi2 * rate)
        self.error = math.floor(self.error * rate)
        self.invalid = math.floor(self.invalid * rate)
        for ch in range(0, CHANNEL_ID_MAX):
            self.slots[ch] = math.floor(self.slots[ch] * rate)
        self.cumac_slots = math.floor(self.cumac_slots * rate)
        self.cumac_invalid = math.floor(self.cumac_invalid * rate)
        self.ue_sel = math.floor(self.ue_sel * rate)
        self.prb_alloc = math.floor(self.prb_alloc * rate)
        self.layer_sel = math.floor(self.layer_sel * rate)
        self.mcs_sel = math.floor(self.mcs_sel * rate)
        self.pfm_sort = math.floor(self.pfm_sort * rate)
        self.mu_ue_grp = math.floor(self.mu_ue_grp * rate)
        return 0

    # Multiply (1 + err) and round up
    def set_high_limit(self, err):
        rate = 1 + err
        self.ul_thrput *= rate
        self.dl_thrput *= rate
        self.mac_ul_thrput *= rate
        self.ul_thrput = float(math.ceil(self.ul_thrput * 100)) / 100
        self.dl_thrput = float(math.ceil(self.dl_thrput * 100)) / 100
        self.mac_ul_thrput = float(math.ceil(self.mac_ul_thrput * 100)) / 100
        self.prmb = math.ceil(self.prmb * rate)
        self.sr = math.ceil(self.sr * rate)
        self.harq = math.ceil(self.harq * rate)
        self.csi1 = math.ceil(self.csi1 * rate)
        self.csi2 = math.ceil(self.csi2 * rate)
        self.error = math.ceil(self.error * rate)
        self.invalid = math.ceil(self.invalid * rate)
        for ch in range(0, CHANNEL_ID_MAX):
            self.slots[ch] = math.ceil(self.slots[ch] * rate)
        self.cumac_slots = math.ceil(self.cumac_slots * rate)
        self.cumac_invalid = math.ceil(self.cumac_invalid * rate)
        self.ue_sel = math.ceil(self.ue_sel * rate)
        self.prb_alloc = math.ceil(self.prb_alloc * rate)
        self.layer_sel = math.ceil(self.layer_sel * rate)
        self.mcs_sel = math.ceil(self.mcs_sel * rate)
        self.pfm_sort = math.ceil(self.pfm_sort * rate)
        self.mu_ue_grp = math.ceil(self.mu_ue_grp * rate)
        return 0

    # Parse expected slot count and throughput data from test_mac initial log. Example:
    # ExpectedSlots: Cell=0 PUSCH=400 PDSCH=1600 PDCCH_UL=0 PDCCH_DL=0 PBCH=0 PUCCH=0 PRACH=0 CSI_RS=0 SRS=0
    # ExpectedData: Cell=0 DL=1586.276800 UL=249.104000 Prmb=0 HARQ=0 SR=0 CSI1=0 CSI2=0 ERR=0
    def parse_expected_fapi(self, slot_line, data_line):
        for ch in range(0, CHANNEL_ID_MAX):
            self.slots[ch] = parse_int(slot_line, channel_names[ch], "=")
        self.ul_thrput = parse_float(data_line, "UL", "=")
        self.dl_thrput = parse_float(data_line, "DL", "=")
        self.mac_ul_drop = parse_float(data_line, "MAC_UL_DROP", "=", 0.0)
        self.mac_ul_thrput = self.ul_thrput - self.mac_ul_drop
        self.prmb = parse_int(data_line, "Prmb", "=")
        self.sr = parse_int(data_line, "SR", "=")
        self.harq = parse_int(data_line, "HARQ", "=")
        self.csi1 = parse_int(data_line, "CSI1", "=")
        self.csi2 = parse_int(data_line, "CSI2", "=")
        self.error = parse_int(data_line, "ERR", "=")
        self.invalid = parse_int(data_line, "INV", "=")

        if (self.ul_thrput == 0 and
            self.dl_thrput == 0 and
            self.mac_ul_drop == 0 and
            self.mac_ul_thrput == 0 and
            self.prmb == 0 and
            self.sr == 0 and
            self.harq == 0 and
            self.csi1 == 0 and
            self.csi2 == 0 and
            self.error == 0 and
            self.invalid == 0 and
            all(v == 0 for v in self.slots)):
            # expected_zero is defaulted to True; keep it so here.
            pass
        else:
            global expected_zero
            expected_zero = False
        return 0

    # Parse expected cumac target throughput data. Example:
    # CUMAC_TargetThrput: Cell=0 CUMAC_SLOT=1000 UE_SEL=1000 PRB_ALLOC=1000 LAYER_SEL=1000
    #                     MCS_SEL=1000 PFM_SORT=1000 MU_UE_GRP=1000 ERR=0 INV=0
    def parse_expected_cumac(self, data_line):
        self.cumac_slots = parse_int(data_line, "CUMAC_SLOT", "=")
        self.error = parse_int(data_line, "ERR", "=")
        self.cumac_invalid = parse_int(data_line, "INV", "=")
        self.ue_sel = parse_int(data_line, "UE_SEL", "=", default=0, regular=True)
        self.prb_alloc = parse_int(data_line, "PRB_ALLOC", "=", default=0, regular=True)
        self.layer_sel = parse_int(data_line, "LAYER_SEL", "=", default=0, regular=True)
        self.mcs_sel = parse_int(data_line, "MCS_SEL", "=", default=0, regular=True)
        self.pfm_sort = parse_int(data_line, "PFM_SORT", "=", default=0, regular=True)
        self.mu_ue_grp = parse_int(data_line, "MU_UE_GRP", "=", default=0, regular=True)
        return 0

    # Parse one cell throughput data from one line test_mac [MAC.FAPI] log
    def parse_fapi(self, line):
        if (len(line) == 0):
            return 1
        # Parse DL/UL from "| DL 1586.28 Mbps 1600 Slots | UL  249.10 Mbps  400 Slots |"
        self.slots[CHANNEL_ID_PDSCH] = parse_int(line, "DL[ ]+[\\S]+[ ]+Mbps", " ", 0, True)
        self.slots[CHANNEL_ID_PUSCH] = parse_int(line, "UL[ ]+[\\S]+[ ]+Mbps", " ", 0, True)
        self.dl_thrput = parse_float(line, " DL ", " ")
        self.ul_thrput = parse_float(line, " UL ", " ")
        self.prmb = parse_int(line, "Prmb", " ")
        self.harq = parse_int(line, "HARQ", " ")
        self.sr = parse_int(line, "SR", " ")
        self.csi1 = parse_int(line, "CSI1", " ")
        self.csi2 = parse_int(line, "CSI2", " ")
        self.error = parse_int(line, "ERR", " ")
        self.invalid = parse_int(line, "INV", " ")
        self.timestamp = shell_cmd("echo '%s' | awk '{print $1}'" % (line))
        return 0

    # Check whether one cell FAPI throughput matches the expected values
    def check_fapi(self, low_limit, high_limit, last):
        global negative_test
        ret = 0
        if (self.timestamp == last.timestamp):
            ret |= 1 << MacErr.TimeStamp
        if (negative_test):
            if (self.error < low_limit.error or self.error > high_limit.error):
                ret |= 1 << MacErr.ERR_IND
            if (self.invalid < low_limit.invalid or self.invalid > high_limit.invalid):
                ret |= 1 << MacErr.Invalid
        else:
            if (self.slots[CHANNEL_ID_PDSCH] < low_limit.slots[CHANNEL_ID_PDSCH] or self.slots[CHANNEL_ID_PDSCH] > high_limit.slots[CHANNEL_ID_PDSCH]):
                ret |= 1 << MacErr.DL_Slots
            if (skip_pusch_slots_check is False and (self.slots[CHANNEL_ID_PUSCH] < low_limit.slots[CHANNEL_ID_PUSCH] or self.slots[CHANNEL_ID_PUSCH] > high_limit.slots[CHANNEL_ID_PUSCH])):
                ret |= 1 << MacErr.UL_Slots
            if (self.dl_thrput < low_limit.dl_thrput or self.dl_thrput > high_limit.dl_thrput):
                ret |= 1 << MacErr.DL_Mbps
            if (skip_pusch_slots_check is False and (self.ul_thrput < low_limit.mac_ul_thrput or self.ul_thrput > high_limit.mac_ul_thrput)):
                ret |= 1 << MacErr.UL_Mbps
            if (l2sa_test == 1):  # Skip other values in L2SA test
                return ret
            if (self.invalid < low_limit.invalid or self.invalid > high_limit.invalid):
                ret |= 1 << MacErr.Invalid
            if (self.prmb < low_limit.prmb or self.prmb > high_limit.prmb):
                ret |= 1 << MacErr.Prmb
            if (self.sr < low_limit.sr or self.sr > high_limit.sr):
                ret |= 1 << MacErr.SR
            if (self.harq < low_limit.harq or self.harq > high_limit.harq):
                ret |= 1 << MacErr.HARQ
            # TODO: redo for Channel PUCCH based CSI1 and CSI2
            if (skip_pusch_slots_check is False and (self.csi1 < low_limit.csi1 or self.csi1 > high_limit.csi1)):
                ret |= 1 << MacErr.CSI1
            # TODO: redo for Channel PUCCH based CSI1 and CSI2
            if (skip_pusch_slots_check is False and (self.csi2 < low_limit.csi2 or self.csi2 > high_limit.csi2)):
                ret |= 1 << MacErr.CSI2
        return ret

    # Parse one cell throughput data from one line ru_emulator throughput log
    def parse_ru_emulator(self, line):
        if (len(line) == 0):
            return 1
        # Parse DL/UL from "| DL 1586.28 Mbps 1600 Slots | UL  249.10 Mbps  400 Slots |"
        self.slots[CHANNEL_ID_PDSCH] = parse_int(line, "DL[ ]+[\\S]+[ ]+Mbps", " ", 0, True)
        self.slots[CHANNEL_ID_PUSCH] = parse_int(line, "UL[ ]+[\\S]+[ ]+Mbps", " ", 0, True)
        self.dl_thrput = parse_float(line, " DL ", " ")
        self.ul_thrput = parse_float(line, " UL ", " ")
        for ch in range(2, CHANNEL_ID_MAX):
            self.slots[ch] = parse_int(line, channel_names[ch], " ")
        self.timestamp = shell_cmd("echo '%s' | awk '{print $1}'" % (line))
        return 0

    # Check whether one cell throughput data match the expected values (ru_emulator)
    def check_ru_emulator(self, low_limit, high_limit, last):
        global negative_test
        ret = 0
        if not negative_test:
            if (self.dl_thrput < low_limit.dl_thrput or self.dl_thrput > high_limit.dl_thrput):
                ret |= 1 << RuErr.DL_Mbps
            if (self.ul_thrput < low_limit.ul_thrput or self.ul_thrput > high_limit.ul_thrput):
                ret |= 1 << RuErr.UL_Mbps
            for ch in range(0, CHANNEL_ID_MAX):
                if (self.slots[ch] < low_limit.slots[ch] or self.slots[ch] > high_limit.slots[ch]):
                    ret |= 1 << RuErr(ch).value
        if (self.timestamp == last.timestamp):
            ret |= 1 << RuErr.TimeStamp
        return ret

    # Parse one cell throughput data from one line [CUMAC.HANDLER] log in MAC file.
    # Format: "Cell  0 | CUMAC 1999 | UE_SEL 100 | PRB_ALLOC 100 | LAYER_SEL 100 |
    #          MCS_SEL 100 | PFM_SORT 100 | MU_UE_GRP 100 | ERR 0 | INV 0 | Slots 2000"
    def parse_cumac(self, line):
        if (len(line) == 0):
            return 1
        self.cumac_slots = parse_int(line, "CUMAC ", " ")
        self.error = parse_int(line, "ERR", " ")
        self.cumac_invalid = parse_int(line, "INV", " ")
        self.ue_sel = parse_int(line, "UE_SEL", " ", default=0, regular=True)
        self.prb_alloc = parse_int(line, "PRB_ALLOC", " ", default=0, regular=True)
        self.layer_sel = parse_int(line, "LAYER_SEL", " ", default=0, regular=True)
        self.mcs_sel = parse_int(line, "MCS_SEL", " ", default=0, regular=True)
        self.pfm_sort = parse_int(line, "PFM_SORT", " ", default=0, regular=True)
        self.mu_ue_grp = parse_int(line, "MU_UE_GRP", " ", default=0, regular=True)
        self.timestamp = shell_cmd("echo '%s' | awk '{print $1}'" % (line))
        return 0

    # Check whether one cell CUMAC throughput matches the expected values
    def check_cumac(self, low_limit, high_limit, last):
        global negative_test
        ret = 0
        if (self.timestamp == last.timestamp):
            ret |= 1 << CuMacErr.TimeStamp
        if (negative_test and (self.error < low_limit.error or self.error > high_limit.error)):
            ret |= 1 << CuMacErr.ERR_IND
        else:
            if (self.cumac_slots < low_limit.cumac_slots or self.cumac_slots > high_limit.cumac_slots):
                ret |= 1 << CuMacErr.Slots
            if (self.cumac_invalid < low_limit.cumac_invalid or self.cumac_invalid > high_limit.cumac_invalid):
                ret |= 1 << CuMacErr.Invalid
            if (self.ue_sel < low_limit.ue_sel or self.ue_sel > high_limit.ue_sel):
                ret |= 1 << CuMacErr.UE_SEL
            if (self.prb_alloc < low_limit.prb_alloc or self.prb_alloc > high_limit.prb_alloc):
                ret |= 1 << CuMacErr.PRB_ALLOC
            if (self.layer_sel < low_limit.layer_sel or self.layer_sel > high_limit.layer_sel):
                ret |= 1 << CuMacErr.LAYER_SEL
            if (self.mcs_sel < low_limit.mcs_sel or self.mcs_sel > high_limit.mcs_sel):
                ret |= 1 << CuMacErr.MCS_SEL
            if (self.pfm_sort < low_limit.pfm_sort or self.pfm_sort > high_limit.pfm_sort):
                ret |= 1 << CuMacErr.PFM_SORT
            if (self.mu_ue_grp < low_limit.mu_ue_grp or self.mu_ue_grp > high_limit.mu_ue_grp):
                ret |= 1 << CuMacErr.MU_UE_GRP
        return ret

    # Parse one cell throughput data from one line cumcp throughput log.
    # Format: "Cell  0 | CUMAC 1999 | ... | ERR 0 | Slots 2000"
    def parse_cumcp(self, line):
        if (len(line) == 0):
            return 1
        self.cumac_slots = parse_int(line, "CUMAC", " ")
        self.error = parse_int(line, "ERR", " ")
        self.ue_sel = parse_int(line, "UE_SEL", " ", default=0, regular=True)
        self.prb_alloc = parse_int(line, "PRB_ALLOC", " ", default=0, regular=True)
        self.layer_sel = parse_int(line, "LAYER_SEL", " ", default=0, regular=True)
        self.mcs_sel = parse_int(line, "MCS_SEL", " ", default=0, regular=True)
        self.pfm_sort = parse_int(line, "PFM_SORT", " ", default=0, regular=True)
        self.mu_ue_grp = parse_int(line, "MU_UE_GRP", " ", default=0, regular=True)
        self.timestamp = shell_cmd("echo '%s' | awk '{print $1}'" % (line))
        return 0

    # Check whether one cell throughput data match the expected values (cumcp)
    def check_cumcp(self, low_limit, high_limit, last):
        global negative_test
        ret = 0
        if (self.cumac_slots < low_limit.cumac_slots or self.cumac_slots > high_limit.cumac_slots):
            ret |= 1 << CumcpErr.Slots
        if (self.error < low_limit.error or self.error > high_limit.error):
            ret |= 1 << CumcpErr.ERR_IND
        # TODO: implement and check individual task slot counts in cumac_cp.
        # cumac_cp NVLOGC throughput line is CUMAC/ERR/Slots only (see print_cumac_cp_thrput).
        # If that format gains per-task fields, enable the checks below:
        # if (self.ue_sel < low_limit.ue_sel or self.ue_sel > high_limit.ue_sel):
        #     ret |= 1 << CumcpErr.UE_SEL
        # if (self.prb_alloc < low_limit.prb_alloc or self.prb_alloc > high_limit.prb_alloc):
        #     ret |= 1 << CumcpErr.PRB_ALLOC
        # if (self.layer_sel < low_limit.layer_sel or self.layer_sel > high_limit.layer_sel):
        #     ret |= 1 << CumcpErr.LAYER_SEL
        # if (self.mcs_sel < low_limit.mcs_sel or self.mcs_sel > high_limit.mcs_sel):
        #     ret |= 1 << CumcpErr.MCS_SEL
        # if (self.pfm_sort < low_limit.pfm_sort or self.pfm_sort > high_limit.pfm_sort):
        #     ret |= 1 << CumcpErr.PFM_SORT
        # if (self.mu_ue_grp < low_limit.mu_ue_grp or self.mu_ue_grp > high_limit.mu_ue_grp):
        #     ret |= 1 << CumcpErr.MU_UE_GRP
        if (self.timestamp == last.timestamp):
            ret |= 1 << CumcpErr.TimeStamp
        return ret

    # Format all non-zero values to a string for log
    def to_string(self, round=True):
        s = ""
        if (len(self.timestamp) != 0):
            s += self.timestamp
        if (self.dl_thrput > 0 or self.slots[CHANNEL_ID_PDSCH] > 0):
            s += " DL="
            s += ("%.2f" % (self.dl_thrput)) if round else str(self.dl_thrput)
            s += "/%d" % (self.slots[CHANNEL_ID_PDSCH])
        if (self.ul_thrput > 0 or self.slots[CHANNEL_ID_PUSCH] > 0):
            s += " UL="
            s += ("%.2f" % (self.ul_thrput)) if round else str(self.ul_thrput)
            s += "/%d" % (self.slots[CHANNEL_ID_PUSCH])
        for ch in range(2, CHANNEL_ID_MAX):
            if (self.slots[ch] > 0):
                s += " " + channel_names[ch] + "=" + str(self.slots[ch])
        if (self.cumac_slots > 0):
            s += " CUMAC=" + str(self.cumac_slots)
        if (self.ue_sel > 0):
            s += " UE_SEL=" + str(self.ue_sel)
        if (self.prb_alloc > 0):
            s += " PRB_ALLOC=" + str(self.prb_alloc)
        if (self.layer_sel > 0):
            s += " LAYER_SEL=" + str(self.layer_sel)
        if (self.mcs_sel > 0):
            s += " MCS_SEL=" + str(self.mcs_sel)
        if (self.pfm_sort > 0):
            s += " PFM_SORT=" + str(self.pfm_sort)
        if (self.mu_ue_grp > 0):
            s += " MU_UE_GRP=" + str(self.mu_ue_grp)
        if (self.prmb > 0):
            s += " Prmb=" + str(self.prmb)
        if (self.sr > 0):
            s += " SR=" + str(self.sr)
        if (self.harq > 0):
            s += " HARQ=" + str(self.harq)
        if (self.csi1 > 0):
            s += " CSI1=" + str(self.csi1)
        if (self.csi2 > 0):
            s += " CSI2=" + str(self.csi2)
        if (negative_test and self.error > 0):
            s += " ERR=" + str(self.error)
        if (self.invalid > 0):
            s += " INV=" + str(self.invalid)
        if (self.cumac_invalid > 0):
            s += " CUMAC_INV=" + str(self.cumac_invalid)
        s = "[" + s.strip() + "]"
        return s

    def expected_string(self):
        s = self.to_string(False)
        if (self.mac_ul_drop > 0):
            s += " [MAC_UL_DROP=" + str(self.mac_ul_drop)
            s += " MAC_UL=" + str(self.mac_ul_thrput)
            s += "]"
        return s


# Function to call shell command
def shell_cmd(cmd, print_cmd=False, print_err=True):
    if print_cmd:
        log_print('[shell] %s' % cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    status = 1 if err else 0
    if (status != 0 and print_err):
        log_print('[Shell ERROR] ' + cmd)
        log_print(str(err, encoding="utf-8"))
    return str(output.strip(), encoding="utf-8")


# Parse a string value after a prefix. Example "prefix=value"
def parse_string(input_string, prefix, delimiter, default="", regular=False):
    try:
        prefix_value = re.search(prefix + "[ " + delimiter + "]*" + "[\\S]+", input_string.strip()).group(0)
        value = prefix_value.split(delimiter)[-1].strip()
    except:
        if (not(prefix in input_string) or regular):
            return default
        else:
            log_print("[ERROR]: Failed to parse [" + prefix + delimiter + "] from [" + input_string + "]")
            return default
    else:
        pass
    return value


def parse_int(input_string, prefix, delimiter, default=0, regular=False):
    str_val = parse_string(input_string, prefix, delimiter, str(default), regular)
    return int(str_val)


def parse_float(input_string, prefix, delimiter, default=0.0, regular=False):
    str_val = parse_string(input_string, prefix, delimiter, str(default), regular)
    return float(str_val)


# Wait for process running until observing expected log
def wait_for_log(log_file, expect_string, timeout=60):
    counter = 0
    grep_result = ""
    while (True):
        grep_result = shell_cmd(f'grep -s -o "{expect_string}" {log_file}')
        if (len(grep_result) != 0):
            break
        counter += 1
        len_result = shell_cmd(f"wc -l {log_file}")
        if (counter > timeout):
            log_print(f"Wait for log: [{expect_string}] ... ({len_result}) timeout")
            break
        else:
            log_print(f"Wait for log [{expect_string}] ... ({len_result}) counter={counter}")
            time.sleep(1)
    return 0


# Parse expected throughput from log. Populates the single Thrput per cell
# with FAPI and/or CUMAC expected data as driven by check_fapi_mac /
# check_cumac_mac.
def parse_expected_thrput(log_file, cell_num):
    thrput_list = []
    for cell_id in range(0, cell_num):
        thrput = Thrput(cell_id)
        if check_fapi_mac:
            slot_line = shell_cmd("cat %s | grep 'ExpectedSlots: Cell=%d'" % (log_file, cell_id))
            data_line = shell_cmd("cat %s | grep 'ExpectedData: Cell=%d'" % (log_file, cell_id))
            thrput.parse_expected_fapi(slot_line, data_line)
        if check_cumac_mac:
            data_line = shell_cmd("cat %s | grep 'CUMAC_TargetThrput: Cell=%d'" % (log_file, cell_id))
            thrput.parse_expected_cumac(data_line)
            if not thrput.is_all_zero():
                global expected_zero
                expected_zero = False
        thrput_list.append(thrput)
        log_print("Expected thrput: Cell " + str(cell_id) + ": " + thrput.expected_string())
    return thrput_list


MAC_LOG = None

# Regex patterns for MAC log line classification.
#   Case (A) / (B) : when cumcp_log is provided, FAPI lines always carry the
#                    explicit [MAC.FAPI] tag.
#   Case (C)       : legacy MAC log has bare "Cell N | DL ..." lines.
_FAPI_TAGGED_RE = re.compile(r'.*\[MAC\.FAPI\]\s+Cell\s+(\d+)')
_FAPI_LEGACY_RE = re.compile(r'.*Cell\s+(\d+)')
_CUMAC_RE = re.compile(r'.*\[CUMAC\.HANDLER\]\s+Cell\s+(\d+)')


def parse_mac_result(log_file, cell_num, want_fapi, want_cumac):
    """Parse both FAPI and CUMAC throughput lines from the MAC log in a
    single forward pass.

    Returns a tuple (fapi_list, cumac_list). Lists are empty when the
    corresponding want_* flag is False.
    """
    global MAC_LOG

    if MAC_LOG is None:
        try:
            MAC_LOG = open(log_file)
        except IOError as e:
            log_print(f"Error opening MAC log file {log_file}: {e}")
            return ([], [])

    fapi_result = []
    cumac_result = []
    errCounter = 0
    maxRetries = 10
    # Current testMAC always tags throughput lines with [MAC.FAPI], so use
    # the tagged regex regardless of whether --cumcp-log was provided.
    fapi_tagged = True

    target_fapi = cell_num if want_fapi else 0
    target_cumac = cell_num if want_cumac else 0

    while len(fapi_result) < target_fapi or len(cumac_result) < target_cumac:
        try:
            line = next(MAC_LOG)

            # Try CUMAC first - it always carries the [CUMAC.HANDLER] tag
            # and would otherwise be matched by the loose legacy FAPI regex.
            if want_cumac and len(cumac_result) < target_cumac:
                m = _CUMAC_RE.match(line)
                if m:
                    cell_id = m.group(1)
                    thrput = Thrput(cell_id)
                    thrput.parse_cumac(line.strip())
                    cumac_result.append(thrput)
                    continue

            if want_fapi and len(fapi_result) < target_fapi:
                if fapi_tagged:
                    m = _FAPI_TAGGED_RE.match(line)
                else:
                    # Legacy: bare "Cell N"; skip any line with a handler tag
                    # just in case.
                    if '[CUMAC.HANDLER]' in line or '[MAC.FAPI]' in line:
                        m = None
                    else:
                        m = _FAPI_LEGACY_RE.match(line)
                if m:
                    cell_id = m.group(1)
                    thrput = Thrput(cell_id)
                    thrput.parse_fapi(line.strip())
                    fapi_result.append(thrput)

        except StopIteration:
            errCounter += 1
            if errCounter > maxRetries:
                print(f"retried {maxRetries} times - giving up")
                break
            else:
                time.sleep(1)

    return (fapi_result, cumac_result)


RU_LOG = None


def parse_ru_result(log_file, cell_num, maxRetries=10):
    global RU_LOG

    if RU_LOG is None:
        try:
            RU_LOG = open(log_file)
        except IOError as e:
            log_print(f"Error opening RU log file {log_file}: {e}")
            return []
    result = []
    errCounter = 0
    while len(result) < cell_num:
        try:
            line = next(RU_LOG)
            m = re.match(r'.*\[RU\]\s+Cell\s+(\d+)', line)
            if m:
                cell_id = m.group(1)
                thrput = Thrput(cell_id)
                thrput.parse_ru_emulator(line.strip())
                result.append(thrput)
        except StopIteration:
            errCounter += 1
            if errCounter > maxRetries:
                print(f"retried {maxRetries} times - giving up")
                break
            else:
                time.sleep(1)

    return result


CUMCP_LOG = None


def parse_cumcp_result(log_file, cell_num, maxRetries=10):
    global CUMCP_LOG

    if CUMCP_LOG is None:
        try:
            CUMCP_LOG = open(log_file)
        except IOError as e:
            log_print(f"Error opening CUMCP log file {log_file}: {e}")
            return []
    result = []
    errCounter = 0
    while len(result) < cell_num:
        try:
            line = next(CUMCP_LOG)
            m = re.match(r'.*\[CUMCP.HANDLER\]\s+Cell\s+(\d+)', line)
            if m:
                cell_id = m.group(1)
                thrput = Thrput(cell_id)
                thrput.parse_cumcp(line.strip())
                result.append(thrput)
        except StopIteration:
            errCounter += 1
            if errCounter > maxRetries:
                print(f"retried {maxRetries} times - giving up")
                break
            else:
                time.sleep(1)

    return result


def check_fapi_thrput(result_list, low_limit_list, high_limit_list, cell_num, last_result, rets):
    for cell_id in range(0, cell_num):
        rets[cell_id] += result_list[cell_id].check_fapi(low_limit_list[cell_id], high_limit_list[cell_id], last_result[cell_id])


def check_cumac_thrput(result_list, low_limit_list, high_limit_list, cell_num, last_result, rets):
    for cell_id in range(0, cell_num):
        rets[cell_id] += result_list[cell_id].check_cumac(low_limit_list[cell_id], high_limit_list[cell_id], last_result[cell_id])


def check_ru_thrput(result_list, low_limit_list, high_limit_list, cell_num, last_result, rets):
    for cell_id in range(0, cell_num):
        rets[cell_id] += result_list[cell_id].check_ru_emulator(low_limit_list[cell_id], high_limit_list[cell_id], last_result[cell_id])


def check_cumcp_thrput(result_list, low_limit_list, high_limit_list, cell_num, last_result, rets):
    for cell_id in range(0, cell_num):
        rets[cell_id] += result_list[cell_id].check_cumcp(low_limit_list[cell_id], high_limit_list[cell_id], last_result[cell_id])


def wait_for_ru_thrput_start(timeout=60):
    global negative_test
    # See GT-6528 - sleep for a bit to let some warning logs scroll by before we parse for the RU logs we're looking for
    time.sleep(5)
    counter = 0
    while (True):
        counter += 1
        if (counter > timeout):
            log_print(f"Wait for ru_emulator throughput start ... ({result_list[0].timestamp}) - ({len_result}) timeout")
            break

        result_list = parse_ru_result(ru_log_file, cell_num, maxRetries=1)
        len_result = shell_cmd(f"cat {ru_log_file} | wc -l")
        if result_list:
            for cell_id in range(0, cell_num):
                if (negative_test or (not result_list[cell_id].is_all_zero())):
                    return 0

            log_print(f"Wait for ru_emulator throughput start ... ({result_list[0].timestamp}) - ({len_result}) counter={counter}")
        else:
            log_print(f"Wait for ru_emulator throughput start ... - ({len_result}) counter={counter}")

    return 0


# Skip RU starting 0 throughput since MAC comes later
def skip_ru_thrput_start(max_seconds=60):
    global negative_test
    counter = 0
    while (True):
        result_list = parse_ru_result(ru_log_file, cell_num)
        for cell_id in range(0, cell_num):
            if (negative_test or (not result_list[cell_id].is_all_zero())):
                return 0
        counter += 1
        len_result = shell_cmd(f"wc -l {ru_log_file}")
        if (counter > max_seconds):
            log_print(f"Skip ru_emulator starting 0 throughput ... ({len_result}) exceeds max_seconds")
            break
        else:
            log_print(f"Skip ru_emulator starting 0 throughput ... ({len_result}) counter={counter}")
    return 0


################################################################
# Execution start
################################################################

# Wait for MAC test to initialize. Keyword depends on which driver owns the
# MAC log. In cumcp_sa_mode the driver is test_cumac; otherwise it's test_mac.
mac_init_wait_string = "TestCUMAC started" if cumcp_sa_mode else "testmac_init"
wait_for_log(mac_log_file, mac_init_wait_string, wait_for_startup_mac)

# Parse cell_num/show_thrput/negative_test/channel_mask from MAC log.
if cumcp_sa_mode:
    line = shell_cmd("grep 'cumac_handler constructed' " + mac_log_file)
    show_thrput = 0
    cell_num = parse_int(line, "cell_num", "=")
    negative_test = 0
    channel_mask = 0  # Not used in cumcp_sa_mode
else:
    line = shell_cmd("grep testmac_init " + mac_log_file)
    show_thrput = parse_int(line, "show_thrput", "=")
    cell_num = parse_int(line, "cell_num", "=")
    negative_test = parse_int(line, "negative_test", "=")
    channel_mask_line = shell_cmd("grep 'channel_mask=' " + mac_log_file)
    channel_mask = int(parse_string(channel_mask_line, "channel_mask", "="), 16)

if negative_test > 0:
    skip_pusch_slots_check = True

launch_pattern = shell_cmd("grep -o launch_pattern.*.yaml " + mac_log_file)

log_print("TestCase: %s cell_num=%s channel_mask=0x%02X negative_test=%d cumcp_sa_mode=%d" %
          (launch_pattern, cell_num, channel_mask, negative_test, int(cumcp_sa_mode)))

if (cell_num == 0 or (not cumcp_sa_mode and channel_mask == 0)):
    log_print("Error parameters")
    log_print_file.close()
    sys.exit(1)

# Parse expected throughput list (populates FAPI and/or CUMAC fields)
expected_list = parse_expected_thrput(mac_log_file, cell_num)
low_limit_list = copy.deepcopy(expected_list)
high_limit_list = copy.deepcopy(expected_list)

# Calculate low and high limits of expected throughput data
for cell_id in range(0, cell_num):
    low_limit_list[cell_id].set_low_limit(CONFIG_ALLOWED_ERROR)
    high_limit_list[cell_id].set_high_limit(CONFIG_ALLOWED_ERROR)
    log_print("Pass criterion low:  Cell " + str(cell_id) + ": " + low_limit_list[cell_id].expected_string())
    log_print("Pass criterion high: Cell " + str(cell_id) + ": " + high_limit_list[cell_id].expected_string())

# Decide which log sources are active for this run.
ru_exist = (ru_log_file is not None) and os.path.isfile(ru_log_file)
cumcp_exist = (cumcp_log_file is not None) and os.path.isfile(cumcp_log_file)

# FAPI throughput shows up in the MAC console only when show_thrput==0, AND
# when the MAC log is not cumac-only (i.e. not in cumcp_sa_mode).
fapi_exist = check_fapi_mac and (show_thrput == 0)
# CUMAC throughput in the MAC log is present whenever cumcp_log is provided
# (cases A and B).
cumac_exist = check_cumac_mac

# Whether we need to parse anything from the MAC log during the main loop
mac_exist = fapi_exist or cumac_exist

# Wait for testmac and/or external emulator throughput to start
if fapi_exist or cumac_exist:
    wait_for_log(mac_log_file, "Cell  0 |", wait_for_throughput)
elif ru_exist or cumcp_exist:
    if expected_zero:
        log_print(f"The expected data/slot throughput is zero for everything. SLEEPING for {zero_timeout} seconds and then exiting cleanly.")
        count = 0
        sleep_increment = 10
        while count < zero_timeout:
            count += sleep_increment
            time.sleep(sleep_increment)
            log_print(f"Slept {count} seconds. Exiting after {zero_timeout} seconds.")
        log_print(f"Slept {count} seconds. Exiting.")
        log_print("Test PASS")
        sys.exit(0)
    else:
        if ru_exist:
            wait_for_ru_thrput_start(wait_for_throughput)

if ru_exist:
    skip_ru_thrput_start(500)

# Skip the first 1 second unstable logs
time.sleep(1)

log_print("Throughput check start ... fapi_exist=" + str(fapi_exist) +
          " cumac_exist=" + str(cumac_exist) +
          " ru_exist=" + str(ru_exist) +
          " cumcp_exist=" + str(cumcp_exist))
log_print("===============================================")

# Run log checking for at most duration time
time_counter = 0
total_fail = 0
continuous_fail = 0
max_continuous_fail = 0
test_pass = True

last_fapi_result = [Thrput(cell_id) for cell_id in range(cell_num)]
last_cumac_result = [Thrput(cell_id) for cell_id in range(cell_num)]
last_ru_result = [Thrput(cell_id) for cell_id in range(cell_num)]
last_cumcp_result = [Thrput(cell_id) for cell_id in range(cell_num)]

fapi_errs = [0 for cell_id in range(cell_num)]
cumac_errs = [0 for cell_id in range(cell_num)]
ru_errs = [0 for cell_id in range(cell_num)]
cumcp_errs = [0 for cell_id in range(cell_num)]

# Pre-init result lists so the first-iteration logging path is safe when a
# given source is inactive.
fapi_result = [Thrput(cell_id) for cell_id in range(cell_num)]
cumac_result = [Thrput(cell_id) for cell_id in range(cell_num)]
ru_result = [Thrput(cell_id) for cell_id in range(cell_num)]
cumcp_result = [Thrput(cell_id) for cell_id in range(cell_num)]

ts_start = time.time()
test_time = 0

while (test_time < duration):
    if mac_exist:
        # Check for logs like "Finished running 600000 slots test"
        expect_string = "Finished running"
        grep_result = shell_cmd(f'grep -s -o "{expect_string}" {mac_log_file}')
        if (len(grep_result) != 0):
            log_print("Found 'Finished running' in testmac logs - test finished")
            break

    time.sleep(1)
    ts_now = time.time()
    test_time = ts_now - ts_start
    timestamp = time.gmtime(test_time)
    timestr = time.strftime("%H:%M:%S", timestamp)
    fapi_errs = [0 for cell_id in range(cell_num)]
    cumac_errs = [0 for cell_id in range(cell_num)]
    ru_errs = [0 for cell_id in range(cell_num)]
    cumcp_errs = [0 for cell_id in range(cell_num)]

    # Parse and check MAC log (FAPI and/or CUMAC)
    if mac_exist:
        new_fapi, new_cumac = parse_mac_result(mac_log_file, cell_num, fapi_exist, cumac_exist)
        if fapi_exist and len(new_fapi) == cell_num:
            fapi_result = new_fapi
            check_fapi_thrput(fapi_result, low_limit_list, high_limit_list, cell_num, last_fapi_result, fapi_errs)
            last_fapi_result = fapi_result
        if cumac_exist and len(new_cumac) == cell_num:
            cumac_result = new_cumac
            check_cumac_thrput(cumac_result, low_limit_list, high_limit_list, cell_num, last_cumac_result, cumac_errs)
            last_cumac_result = cumac_result

    # Parse and check ru_emulator throughput
    if ru_exist:
        ru_result = parse_ru_result(ru_log_file, cell_num)
        check_ru_thrput(ru_result, low_limit_list, high_limit_list, cell_num, last_ru_result, ru_errs)
        last_ru_result = ru_result

    # Parse and check cumcp throughput
    if cumcp_exist:
        cumcp_result = parse_cumcp_result(cumcp_log_file, cell_num)
        check_cumcp_thrput(cumcp_result, low_limit_list, high_limit_list, cell_num, last_cumcp_result, cumcp_errs)
        last_cumcp_result = cumcp_result

    # Print throughput in console. MAC/CUMAC come from mac_log_file (MAC is
    # the [MAC.FAPI] throughput, CUMAC is the [CUMAC.HANDLER] throughput).
    for cell_id in range(0, cell_num):
        mac_str = " MAC:" + fapi_result[cell_id].to_string() if fapi_exist else ""
        ru_str = " RU:" + ru_result[cell_id].to_string() if ru_exist else ""
        cumac_str = " CUMAC:" + cumac_result[cell_id].to_string() if cumac_exist else ""
        cumcp_str = " CUMCP:" + cumcp_result[cell_id].to_string() if cumcp_exist else ""
        err_str = ""
        err_str += " mac_err=" + hex(fapi_errs[cell_id]) if fapi_exist else ""
        err_str += " ru_err=" + hex(ru_errs[cell_id]) if ru_exist else ""
        err_str += " cumac_err=" + hex(cumac_errs[cell_id]) if cumac_exist else ""
        err_str += " cumcp_err=" + hex(cumcp_errs[cell_id]) if cumcp_exist else ""
        log_print(timestr + " Cell " + str(cell_id) + ":" + mac_str + ru_str + cumac_str + cumcp_str + err_str)

    # Accumulate fail counters
    if (sum(fapi_errs) == 0 and sum(cumac_errs) == 0 and sum(ru_errs) == 0 and sum(cumcp_errs) == 0):
        continuous_fail = 0
    else:
        continuous_fail += 1
        total_fail += 1
        max_continuous_fail = continuous_fail if continuous_fail > max_continuous_fail else max_continuous_fail

    # If continuously failed for CONFIG_CONTINUOUS_FAIL_TIME time, treat as fail
    if (continuous_fail >= CONFIG_CONTINUOUS_FAIL_TIME or total_fail >= CONFIG_TOTAL_FAIL_TIME):
        test_pass = False
        break

if RU_LOG is not None:
    RU_LOG.close()

if CUMCP_LOG is not None:
    CUMCP_LOG.close()

if MAC_LOG is not None:
    MAC_LOG.close()

log_print("Test time: %d seconds, max continuous fail: %d, total fail: %d" %
          (test_time, max_continuous_fail, total_fail))

for cell_id in range(0, cell_num):
    err_str = "The last fail: Cell " + str(cell_id) + ":"
    if (fapi_errs[cell_id] != 0):
        err_str += " " + fapi_err_to_string(fapi_errs[cell_id])
    if (cumac_errs[cell_id] != 0):
        err_str += " " + cumac_err_to_string(cumac_errs[cell_id])
    if (ru_errs[cell_id] != 0):
        err_str += " " + ru_err_to_string(ru_errs[cell_id])
    if (cumcp_errs[cell_id] != 0):
        err_str += " " + cumcp_err_to_string(cumcp_errs[cell_id])
    if (fapi_errs[cell_id] != 0 or cumac_errs[cell_id] != 0 or ru_errs[cell_id] != 0 or cumcp_errs[cell_id] != 0):
        log_print(err_str)

if (test_pass):
    log_print("Test PASS")
    return_value = 0
else:
    log_print("Test FAILED")
    return_value = 1

log_print_file.close()
sys.exit(return_value)
