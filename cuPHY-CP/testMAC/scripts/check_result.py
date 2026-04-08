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
import traceback
import os.path
import enum

log_path = os.getenv('LOG_PATH')
if log_path is None:
    log_path = os.getenv('cuBB_SDK') + "/logs"

if log_path is None:
    print("Please set LOG_PATH or cuBB_SDK first")
    exit(0)

if (len(sys.argv) >= 2):
    duration = int(sys.argv[1])
else:
    print("Usage: ./check_result.py <duration> [mac.log] [ru.log]")
    exit(0)

if (len(sys.argv) >= 3):
    mac_log_file = str(sys.argv[2])
else:
    mac_log_file = log_path + "/screenlog_mac.log"

if (len(sys.argv) >= 4):
    ru_log_file = str(sys.argv[3])
else:
    ru_log_file = log_path + "/screenlog_ru.log"

result_log_file = log_path + "/check_result.log"

# Different tests take longer to startup.
# F08_F_6C_35 - 3 minutes
# F08_F_9C_33 - 5 minutes
# For now - hardcode to 5 minutes (longeest of all) - need to make this
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
skip_pusch_slots_check = False # Works for PUSCH and disables tput checking too
expected_zero = True  # Set to true by default, and if any cell is non-zero it will set this global to false
ru_zero_timeout = duration + 60  # Add 60 seconds to duration for timeout when the expected slot/data is exactly zero.

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
log_print("duration=%d allowed_error=%.2f l2sa_test=%d" % (duration, CONFIG_ALLOWED_ERROR, l2sa_test))
log_print("mac_log=" + mac_log_file)
log_print("ru_log=" + ru_log_file)
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


def mac_err_to_string(value):
    s = "MAC:[" + hex(value)
    for i in range(0, MacErr.BitMAX.value):
        if (value & (1 << i) != 0):
            s += " " + MacErr(i).name
    s += "]"
    return s


def ru_err_to_string(value):
    s = "RU:[" + hex(value)
    for i in range(0, RuErr.BitMAX.value):
        if (value & (1 << i) != 0):
            s += " " + RuErr(i).name
    s += "]"
    return s


# Throughput data class for one cell
class Thrput:

    def __init__(self, cell_id):
        self.cell_id = cell_id
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
        return 0

    # Parse expected slot count and throughput data from test_mac initial log. Example:
    # ExpectedSlots: Cell=0 PUSCH=400 PDSCH=1600 PDCCH_UL=0 PDCCH_DL=0 PBCH=0 PUCCH=0 PRACH=0 CSI_RS=0 SRS=0
    # ExpectedData: Cell=0 DL=1586.276800 UL=249.104000 Prmb=0 HARQ=0 SR=0 CSI1=0 CSI2=0 ERR=0
    def parse_expected(self, slot_line, data_line):
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

        # if negative_test:
        #     self.slots[CHANNEL_ID_PUSCH] = 0
        #     self.ul_thrput = 0

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
            # Do nothing in this case - expected_zero is defaulted to True. We set to False for any cell that has non-zero expectations.
            pass
        else:
            # We have expected_zero set to True by default, then if any one of the cells parsed by parse_expected() have non-zero
            # expected throughput or data, then we set the global to False.
            # Otherwise, if everything is expected to be zero, then we keep expected_zero as True so we do not wait forever for throughput
            # Example - TC 144, which just has ZP-CSI_RS, so the RU will not expect any traffic during this test.
            global expected_zero
            expected_zero = False

        return 0

    # Parse one cell throughput data from one line test_mac throughput log
    def parse_testmac(self, line):
        if (len(line) == 0):
            return 1
        # Parse DL/UL from "| DL 1586.28 Mbps 1600 Slots | UL  249.10 Mbps  400 Slots |"
        self.slots[CHANNEL_ID_PDSCH] = parse_int(line, "DL[ ]+[\S]+[ ]+Mbps", " ", 0, True)
        self.slots[CHANNEL_ID_PUSCH] = parse_int(line, "UL[ ]+[\S]+[ ]+Mbps", " ", 0, True)
        self.dl_thrput = parse_float(line, " DL ", " ")
        self.ul_thrput = parse_float(line, " UL ", " ")
        # Parse other data
        self.prmb = parse_int(line, "Prmb", " ")
        self.harq = parse_int(line, "HARQ", " ")
        self.sr = parse_int(line, "SR", " ")
        self.csi1 = parse_int(line, "CSI1", " ")
        self.csi2 = parse_int(line, "CSI2", " ")
        self.error = parse_int(line, "ERR", " ")
        self.invalid = parse_int(line, "INV", " ")
        self.timestamp = shell_cmd("echo '%s' | awk '{print $1}'" % (line))
        return 0

    # Check whether one cell throughput data match the expected values
    def check_testmac(self, low_limit, high_limit, last):
        global negative_test
        ret = 0
        # If time stamp not change, it means the log was stopped
        if (self.timestamp == last.timestamp):
            ret |= 1 << MacErr.TimeStamp
        if (negative_test):
            if (self.error < low_limit.error or self.error > high_limit.error):
                ret |= 1 << MacErr.ERR_IND
            if (self.invalid < low_limit.invalid or self.invalid > high_limit.invalid):
                ret |= 1 << MacErr.Invalid
        else:
            # Check PDSCH, PUSCH slot count in test_mac
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

    # Parse one cell throughput data from one line test_mac throughput log
    def parse_ru_emulator(self, line):
        if (len(line) == 0):
            return 1
        # Parse DL/UL from "| DL 1586.28 Mbps 1600 Slots | UL  249.10 Mbps  400 Slots |"
        self.slots[CHANNEL_ID_PDSCH] = parse_int(line, "DL[ ]+[\S]+[ ]+Mbps", " ", 0, True)
        self.slots[CHANNEL_ID_PUSCH] = parse_int(line, "UL[ ]+[\S]+[ ]+Mbps", " ", 0, True)
        self.dl_thrput = parse_float(line, " DL ", " ")
        self.ul_thrput = parse_float(line, " UL ", " ")
        # Parse other change slot count
        for ch in range(2, CHANNEL_ID_MAX):
            self.slots[ch] = parse_int(line, channel_names[ch], " ")
        self.timestamp = shell_cmd("echo '%s' | awk '{print $1}'" % (line))
        return 0

    # Check whether one cell throughput data match the expected values
    def check_ru_emulator(self, low_limit, high_limit, last):
        global negative_test
        ret = 0
        if not negative_test:
            if (self.dl_thrput < low_limit.dl_thrput or self.dl_thrput > high_limit.dl_thrput):
                ret |= 1 << RuErr.DL_Mbps
            if (self.ul_thrput < low_limit.ul_thrput or self.ul_thrput > high_limit.ul_thrput):
                ret |= 1 << RuErr.UL_Mbps
            # Check all channel slot count in ru_emulator
            for ch in range(0, CHANNEL_ID_MAX):
                if (self.slots[ch] < low_limit.slots[ch] or self.slots[ch] > high_limit.slots[ch]):
                    ret |= 1 << RuErr(ch).value
        # If time stamp not change, it means the log was stopped
        if (self.timestamp == last.timestamp):
            ret |= 1 << RuErr.TimeStamp
        return ret

    # Format all non-zero values to a string for log
    def to_string(self, round=True):
        s = ""
        if (len(self.timestamp) != 0):
            s += self.timestamp
        # s += "Cell " + str(self.cell_id) + ":"
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
        if ( negative_test and self.error > 0): s += " ERR=" + str(self.error)
        if (self.invalid > 0):
            s += " INV=" + str(self.invalid)
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
    # Print the command for debug if enabled
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
        prefix_value = re.search(prefix + "[ " + delimiter + "]*" + "[\S]+", input_string.strip()).group(0)
        value = prefix_value.split(delimiter)[-1].strip()
    except:
        if (not(prefix in input_string) or regular):
            return default
        else:
            log_print("[ERROR]: Failed to parse [" + prefix + delimiter + "] from [" + input_string + "]")
            # traceback.print_exc()
            return default
    else:
        pass
    return value


# Parse a int value after a prefix. Example "prefix=value"
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


# Parse expected throughput from log. Example:
def parse_expected_thrput(log_lines, cell_num):
    thrput_list = []
    for cell_id in range(0, cell_num):
        slot_line = shell_cmd("cat %s | grep 'ExpectedSlots: Cell=%d'" % (log_lines, cell_id))
        data_line = shell_cmd("cat %s | grep 'ExpectedData: Cell=%d'" % (log_lines, cell_id))
        thrput = Thrput(cell_id)
        thrput.parse_expected(slot_line, data_line)
        thrput_list.append(thrput)
        log_print("Expected thrput: Cell " + str(cell_id) + ": " + thrput.expected_string())
    return thrput_list


MAC_LOG=None
def parse_testmac_result(log_file, cell_num):
    global MAC_LOG

    if MAC_LOG is None:
        MAC_LOG = open(log_file)
    result = []
    errCounter = 0
    maxRetries = 10
    while len(result) < cell_num:
        try:
            line = next(MAC_LOG)
            m = re.match(r'.*Cell\s+(\d+)', line)
            if m:
                cell_id = m.group(1)
                thrput = Thrput(cell_id)
                thrput.parse_testmac(line.strip())
                result.append(thrput)
        except StopIteration:
            # Debug information - too much for the logs
            #print ("reached end of", log_file)
            errCounter += 1
            if errCounter > maxRetries:
                print (f"retried {maxRetries} times - giving up")
                break
            else:
                time.sleep(1)

    return result


RU_LOG=None
def parse_ru_result(log_file, cell_num, maxRetries=10):
    global RU_LOG

    if RU_LOG is None:
        RU_LOG = open(log_file)
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
            # Debug information - too much for the logs
            #print ("reached end of", log_file)
            errCounter += 1
            if errCounter > maxRetries:
                print (f"retried {maxRetries} times - giving up")
                break
            else:
                time.sleep(1)

    return result


def check_testmac_thrput(result_list, low_limit_list, high_limit_list, cell_num, last_result, rets):
    for cell_id in range(0, cell_num):
        rets[cell_id] += result_list[cell_id].check_testmac(low_limit_list[cell_id], high_limit_list[cell_id], last_result[cell_id])


def check_ru_thrput(result_list, low_limit_list, high_limit_list, cell_num, last_result, rets):
    for cell_id in range(0, cell_num):
        rets[cell_id] += result_list[cell_id].check_ru_emulator(low_limit_list[cell_id], high_limit_list[cell_id], last_result[cell_id])


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
            # If the result_list is empty, then it hasn't gotten to the point where it sees throughput. We do not have a timestamp
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


# Keep checking until see "testmac_init" log
wait_for_log(mac_log_file, "testmac_init", wait_for_startup_mac)

# Parse cell_num and show_thrput
line = shell_cmd("grep testmac_init " + mac_log_file)
show_thrput = parse_int(line, "show_thrput", "=")
cell_num = parse_int(line, "cell_num", "=")
negative_test = parse_int(line, "negative_test", "=")

if negative_test > 0:
    skip_pusch_slots_check = True

# Parse enabled channels
line = shell_cmd("grep 'channel_mask=' " + mac_log_file)
channel_mask = int(parse_string(line, "channel_mask", "="), 16)  # Hex string to int

# Parse launch pattern file name
launch_pattern = shell_cmd("grep -o launch_pattern.*.yaml " + mac_log_file)

log_print("TestCase: %s cell_num=%s channel_mask=0x%02X negative_test=%d" % (launch_pattern, cell_num, channel_mask, negative_test))

if (cell_num == 0 or channel_mask == 0):
    # Close the log file
    log_print("Error parameters")
    log_print_file.close()
    sys.exit(1)

# Parse expected throughput list
expected_list = parse_expected_thrput(mac_log_file, cell_num)
low_limit_list = copy.deepcopy(expected_list)
high_limit_list = copy.deepcopy(expected_list)

# Calculate low limitation and high limitation of expected throughput data
for cell_id in range(0, cell_num):
    low_limit_list[cell_id].set_low_limit(CONFIG_ALLOWED_ERROR)
    high_limit_list[cell_id].set_high_limit(CONFIG_ALLOWED_ERROR)
    log_print("Pass criterion low:  Cell " + str(cell_id) + ": " + low_limit_list[cell_id].expected_string())
    log_print("Pass criterion high: Cell " + str(cell_id) + ": " + high_limit_list[cell_id].expected_string())

# Whether ru_emulator is running on this machine
if os.path.isfile(ru_log_file):
    ru_exist = True
else:
    ru_exist = False

# Whether test_mac is running on this machine
if (show_thrput == 0):
    mac_exist = True
else:
    mac_exist = False

# Wait for testmac and/or ru_emulator throughput start
if mac_exist:
    wait_for_log(mac_log_file, "Cell  0 |", wait_for_throughput)
elif ru_exist:
    if expected_zero:
        log_print(f"The expected data/slot throughput is zero for everything. SLEEPING for {ru_zero_timeout} seconds and then exiting cleanly.")
        count = 0
        sleep_increment = 10  # Print a log every 10 seconds to show it isn't hanging.
        while count < ru_zero_timeout:
            count += sleep_increment
            time.sleep(sleep_increment)
            log_print(f"Slept {count} seconds. Exiting after {ru_zero_timeout} seconds.")
        log_print(f"Slept {count} seconds. Exiting.")
        log_print("Test PASS")
        sys.exit(0)
    else:
        wait_for_ru_thrput_start(wait_for_throughput)

if ru_exist:
    skip_ru_thrput_start(500)

# Skip the first 1 second unstable logs
time.sleep(1)

log_print("Throughput check start ... mac_exist=" + str(mac_exist) + " ru_exist=" + str(ru_exist))
log_print("===============================================")

# Run log checking for at most duration time
time_counter = 0
total_fail = 0
continuous_fail = 0
max_continuous_fail = 0
test_pass = True

last_mac_result = [Thrput(cell_id) for cell_id in range(cell_num)]
last_ru_result = [Thrput(cell_id) for cell_id in range(cell_num)]

mac_errs = [0 for cell_id in range(cell_num)]
ru_errs = [0 for cell_id in range(cell_num)]

ts_start = time.time()
test_time = 0

while (test_time < duration):
    if mac_exist:
        # Check for logs like "Finished running 600000 slots test"
        # Same way wait_for_log() does it
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
    mac_errs = [0 for cell_id in range(cell_num)]
    ru_errs = [0 for cell_id in range(cell_num)]

    # Parse and check test_mac throughput
    if mac_exist:
        mac_result = parse_testmac_result(mac_log_file, cell_num)
        check_testmac_thrput(mac_result, low_limit_list, high_limit_list, cell_num, last_mac_result, mac_errs)
        last_mac_result = mac_result

    # Parse and check ru_emulator throughput
    if ru_exist:
        ru_result = parse_ru_result(ru_log_file, cell_num)
        check_ru_thrput(ru_result, low_limit_list, high_limit_list, cell_num, last_ru_result, ru_errs)
        last_ru_result = ru_result

    # Print throughput in console
    for cell_id in range(0, cell_num):
        err_str = ""
        mac_thrput = " MAC:" + mac_result[cell_id].to_string() if mac_exist else ""
        ru_thrput = " RU:" + ru_result[cell_id].to_string() if ru_exist else ""
        err_str += " mac_err=" + hex(mac_errs[cell_id]) if mac_exist else ""
        err_str += " ru_err=" + hex(ru_errs[cell_id]) if ru_exist else ""
        log_print(timestr + " Cell " + str(cell_id) + ":" + mac_thrput + ru_thrput + err_str)

    # Add to result counters
    if (sum(mac_errs) == 0 and sum(ru_errs) == 0):
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

if MAC_LOG is not None:
    MAC_LOG.close()

log_print("Test time: %d seconds, max continuous fail: %d, total fail: %d" % (test_time, max_continuous_fail, total_fail))

for cell_id in range(0, cell_num):
    err_str = "The last fail: Cell " + str(cell_id) + ":"
    if (mac_errs[cell_id] != 0):
        err_str += " " + mac_err_to_string(mac_errs[cell_id])
    if (ru_errs[cell_id] != 0):
        err_str += " " + ru_err_to_string(ru_errs[cell_id])
    if (mac_errs[cell_id] != 0 or ru_errs[cell_id] != 0):
        log_print(err_str)

# Exit with status code
if (test_pass):
    log_print("Test PASS")
    return_value = 0
else:
    log_print("Test FAILED")
    return_value = 1

# Close the log file
log_print_file.close()
sys.exit(return_value)
