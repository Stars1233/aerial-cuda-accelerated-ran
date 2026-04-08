#!/usr/bin/python3 -u

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
    print("Usage: ./check_result.py <duration> [mac.log] [cumcp.log]")
    exit(0)

if (len(sys.argv) >= 3):
    mac_log_file = str(sys.argv[2])
else:
    mac_log_file = log_path + "/screenlog_mac.log"

if (len(sys.argv) >= 4):
    cumcp_log_file = str(sys.argv[3])
else:
    cumcp_log_file = log_path + "/screenlog_cum.log"

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
cumcp_zero_timeout = duration + 60  # Add 60 seconds to duration for timeout when the expected slot/data is exactly zero.

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
log_print("cumcp_log=" + cumcp_log_file)
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
    ERR_IND = 0  # ERR.indication FAPI message received
    Slots = 1  # CUMAC slots per second error
    Invalid = 2  # MAC FAPI validation mismatches
    TimeStamp = 3  # Times stamp is not updating in MAC console output, traffic stopped or process frozen
    UE_SEL = 4  # UE_SEL error
    PRB_ALLOC = 5  # PRB_ALLOC error
    LAYER_SEL = 6  # LAYER_SEL error
    MCS_SEL = 7  # MCS_SEL error
    PFM_SORT = 8  # PFM_SORT error
    BitMAX = 9  # MAX bit number, not an error

class CumcpErr(enum.IntEnum):
    ERR_IND = 0  # ERR.indication FAPI message received
    Slots = 1  # CUMAC slots per second error
    TimeStamp = 2  # Times stamp is not updating in MAC console output, traffic stopped or process frozen
    UE_SEL = 3  # UE_SEL error
    PRB_ALLOC = 4  # PRB_ALLOC error
    LAYER_SEL = 5  # LAYER_SEL error
    MCS_SEL = 6  # MCS_SEL error
    PFM_SORT = 7  # PFM_SORT error
    BitMAX = 8  # MAX bit number, not an error


def mac_err_to_string(value):
    s = "MAC:[" + hex(value)
    for i in range(0, MacErr.BitMAX.value):
        if (value & (1 << i) != 0):
            s += " " + MacErr(i).name
    s += "]"
    return s


def cumcp_err_to_string(value):
    s = "CUMCP:[" + hex(value)
    for i in range(0, CumcpErr.BitMAX.value):
        if (value & (1 << i) != 0):
            s += " " + CumcpErr(i).name
    s += "]"
    return s


# Throughput data class for one cell
class Thrput:

    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.slots = 0
        self.error = 0
        self.invalid = 0
        self.ue_sel = 0
        self.prb_alloc = 0
        self.layer_sel = 0
        self.mcs_sel = 0
        self.pfm_sort = 0
        self.timestamp = ""

    def is_all_zero(self):
        if (self.slots != 0):
            return False
        return True

    # Multiply (1 - err) and round down
    def set_low_limit(self, err):
        rate = 1 - err
        self.slots = math.floor(self.slots * rate)
        self.error = math.floor(self.error * rate)
        self.invalid = math.floor(self.invalid * rate)
        self.ue_sel = math.floor(self.ue_sel * rate)
        self.prb_alloc = math.floor(self.prb_alloc * rate)
        self.layer_sel = math.floor(self.layer_sel * rate)
        self.mcs_sel = math.floor(self.mcs_sel * rate)
        self.pfm_sort = math.floor(self.pfm_sort * rate)
        return 0

    # Multiply (1 + err) and round up
    def set_high_limit(self, err):
        rate = 1 + err
        self.slots = math.ceil(self.slots * rate)
        self.error = math.ceil(self.error * rate)
        self.invalid = math.ceil(self.invalid * rate)
        self.ue_sel = math.ceil(self.ue_sel * rate)
        self.prb_alloc = math.ceil(self.prb_alloc * rate)
        self.layer_sel = math.ceil(self.layer_sel * rate)
        self.mcs_sel = math.ceil(self.mcs_sel * rate)
        self.pfm_sort = math.ceil(self.pfm_sort * rate)
        return 0

    # Parse expected slot count and throughput data from test_mac initial log. Example:
    # ExpectedSlots: Cell=0 PUSCH=400 PDSCH=1600 PDCCH_UL=0 PDCCH_DL=0 PBCH=0 PUCCH=0 PRACH=0 CSI_RS=0 SRS=0
    # ExpectedData: Cell=0 DL=1586.276800 UL=249.104000 Prmb=0 HARQ=0 SR=0 CSI1=0 CSI2=0 ERR=0
    # CUMAC_TargetThrput: Cell=0 CUMAC_SLOT=1000 UE_SEL=1000 PRB_ALLOC=1000 LAYER_SEL=1000 MCS_SEL=1000 PFM_SORT=1000 ERR=0 INV=0
    def parse_expected(self, data_line):
        self.slots = parse_int(data_line, "CUMAC_SLOT", "=")
        self.error = parse_int(data_line, "ERR", "=")
        self.invalid = parse_int(data_line, "INV", "=")
        self.ue_sel = parse_int(data_line, "UE_SEL", "=", default=0, regular=True)
        self.prb_alloc = parse_int(data_line, "PRB_ALLOC", "=", default=0, regular=True)
        self.layer_sel = parse_int(data_line, "LAYER_SEL", "=", default=0, regular=True)
        self.mcs_sel = parse_int(data_line, "MCS_SEL", "=", default=0, regular=True)
        self.pfm_sort = parse_int(data_line, "PFM_SORT", "=", default=0, regular=True)
        return 0

    # Parse one cell throughput data from one line test_mac throughput log
    def parse_testmac(self, line):
        if (len(line) == 0):
            return 1
        # Parse from format: "Cell  0 | CUMAC 1999 | UE_SEL 100 | PRB_ALLOC 100 | LAYER_SEL 100 | MCS_SEL 100 | PFM_SORT 100 | ERR    0 | INV 1999 | Slots 2000"
        # Parse DL/UL from "| DL 1586.28 Mbps 1600 Slots | UL  249.10 Mbps  400 Slots |"
        self.slots = parse_int(line, "CUMAC ", " ")
        self.error = parse_int(line, "ERR", " ")
        self.invalid = parse_int(line, "INV", " ")
        self.ue_sel = parse_int(line, "UE_SEL", " ", default=0, regular=True)
        self.prb_alloc = parse_int(line, "PRB_ALLOC", " ", default=0, regular=True)
        self.layer_sel = parse_int(line, "LAYER_SEL", " ", default=0, regular=True)
        self.mcs_sel = parse_int(line, "MCS_SEL", " ", default=0, regular=True)
        self.pfm_sort = parse_int(line, "PFM_SORT", " ", default=0, regular=True)
        self.timestamp = shell_cmd("echo '%s' | awk '{print $1}'" % (line))
        return 0

    # Check whether one cell throughput data match the expected values
    def check_testmac(self, low_limit, high_limit, last):
        global negative_test
        ret = 0
        # If time stamp not change, it means the log was stopped
        if (self.timestamp == last.timestamp):
            ret |= 1 << MacErr.TimeStamp
        if (negative_test and (self.error < low_limit.error or self.error > high_limit.error)):
            ret |= 1 << MacErr.ERR_IND
        else:
            # Check PDSCH, PUSCH slot count in test_mac
            if (self.slots < low_limit.slots or self.slots > high_limit.slots):
                ret |= 1 << MacErr.Slots
            if (self.invalid < low_limit.invalid or self.invalid > high_limit.invalid):
                ret |= 1 << MacErr.Invalid
            # Check individual task slot counts
            if (self.ue_sel < low_limit.ue_sel or self.ue_sel > high_limit.ue_sel):
                ret |= 1 << MacErr.UE_SEL
            if (self.prb_alloc < low_limit.prb_alloc or self.prb_alloc > high_limit.prb_alloc):
                ret |= 1 << MacErr.PRB_ALLOC
            if (self.layer_sel < low_limit.layer_sel or self.layer_sel > high_limit.layer_sel):
                ret |= 1 << MacErr.LAYER_SEL
            if (self.mcs_sel < low_limit.mcs_sel or self.mcs_sel > high_limit.mcs_sel):
                ret |= 1 << MacErr.MCS_SEL
            if (self.pfm_sort < low_limit.pfm_sort or self.pfm_sort > high_limit.pfm_sort):
                ret |= 1 << MacErr.PFM_SORT

        return ret

    # Parse one cell throughput data from one line cumcp throughput log
    def parse_cumcp(self, line):
        if (len(line) == 0):
            return 1
        # Parse from format: "Cell  0 | CUMAC 1999 | UE_SEL 100 | PRB_ALLOC 100 | LAYER_SEL 100 | MCS_SEL 100 | PFM_SORT 100 | ERR    0 | Slots 2000"
        self.slots = parse_int(line, "CUMAC", " ")
        self.error = parse_int(line, "ERR", " ")
        self.ue_sel = parse_int(line, "UE_SEL", " ", default=0, regular=True)
        self.prb_alloc = parse_int(line, "PRB_ALLOC", " ", default=0, regular=True)
        self.layer_sel = parse_int(line, "LAYER_SEL", " ", default=0, regular=True)
        self.mcs_sel = parse_int(line, "MCS_SEL", " ", default=0, regular=True)
        self.pfm_sort = parse_int(line, "PFM_SORT", " ", default=0, regular=True)
        self.timestamp = shell_cmd("echo '%s' | awk '{print $1}'" % (line))
        return 0

    # Check whether one cell throughput data match the expected values
    def check_cumcp(self, low_limit, high_limit, last):
        global negative_test
        ret = 0
        if (self.slots < low_limit.slots or self.slots > high_limit.slots):
            ret |= 1 << CumcpErr.Slots
        if (self.error < low_limit.error or self.error > high_limit.error):
            ret |= 1 << CumcpErr.ERR_IND
        # TODO: Impelemnt and check individual task slot counts in cumac_cp
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
        # If time stamp not change, it means the log was stopped
        if (self.timestamp == last.timestamp):
            ret |= 1 << CumcpErr.TimeStamp
        return ret

    # Format all non-zero values to a string for log
    def to_string(self, round=True):
        s = ""
        if (len(self.timestamp) != 0):
            s += self.timestamp

        if (self.slots > 0):
            s += " CUMAC=" + str(self.slots)
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
        if (self.error > 0):
            s += " ERR=" + str(self.error)
        if (self.invalid > 0):
            s += " INV=" + str(self.invalid)
        s = "[" + s.strip() + "]"
        return s

    def expected_string(self):
        s = self.to_string(False)
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
        data_line = shell_cmd("cat %s | grep 'CUMAC_TargetThrput: Cell=%d'" % (log_lines, cell_id))
        thrput = Thrput(cell_id)
        thrput.parse_expected(data_line)
        thrput_list.append(thrput)
        log_print("Expected thrput: Cell " + str(cell_id) + ": " + thrput.expected_string())
    return thrput_list


MAC_LOG=None
def parse_testmac_result(log_file, cell_num):
    global MAC_LOG

    if MAC_LOG is None:
        try:
            MAC_LOG = open(log_file)
        except IOError as e:
            log_print(f"Error opening MAC log file {log_file}: {e}")
            return []
    result = []
    errCounter = 0
    maxRetries = 10
    while len(result) < cell_num:
        try:
            line = next(MAC_LOG)
            # Match only throughput lines with pipe format: "Cell  0 | CUMAC 1999 | ERR    0 | INV 1999 | Slots 2000"
            m = re.match(r'.*\[CUMAC.HANDLER\]\s+Cell\s+(\d+)', line)
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


CUMCP_LOG=None
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
            # Match only throughput lines with pipe format: "Cell  0 | CUMAC 1999 | ERR    0 | Slots 2000"
            m = re.match(r'.*\[CUMCP.HANDLER\]\s+Cell\s+(\d+)', line)
            if m:
                cell_id = m.group(1)
                thrput = Thrput(cell_id)
                thrput.parse_cumcp(line.strip())
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


def check_cumcp_thrput(result_list, low_limit_list, high_limit_list, cell_num, last_result, rets):
    for cell_id in range(0, cell_num):
        rets[cell_id] += result_list[cell_id].check_cumcp(low_limit_list[cell_id], high_limit_list[cell_id], last_result[cell_id])


def wait_for_cumcp_thrput_start(timeout=60):
    global negative_test
    # See GT-6528 - sleep for a bit to let some warning logs scroll by before we parse for the CUMCP logs we're looking for
    time.sleep(5)
    counter = 0
    while (True):
        counter += 1
        if (counter > timeout):
            log_print(f"Wait for cumcp throughput start ... ({result_list[0].timestamp}) - ({len_result}) timeout")
            break

        result_list = parse_cumcp_result(cumcp_log_file, cell_num, maxRetries=1)
        len_result = shell_cmd(f"cat {cumcp_log_file} | wc -l")
        if result_list and len(result_list) >= cell_num:
            for cell_id in range(0, cell_num):
                if (negative_test or (not result_list[cell_id].is_all_zero())):
                    return 0

            log_print(f"Wait for cumcp throughput start ... ({result_list[0].timestamp}) - ({len_result}) counter={counter}")
        else:
            # If the result_list is empty, then it hasn't gotten to the point where it sees throughput. We do not have a timestamp
            log_print(f"Wait for cumcp throughput start ... - ({len_result}) counter={counter}")

    return 0


# Skip CUMCP starting 0 throughput since MAC comes later
def skip_cumcp_thrput_start(max_seconds=60):
    global negative_test
    counter = 0
    while (True):
        result_list = parse_cumcp_result(cumcp_log_file, cell_num)
        if len(result_list) < cell_num:
            # Not enough results yet, continue waiting
            pass
        else:
            for cell_id in range(0, cell_num):
                if (negative_test or (not result_list[cell_id].is_all_zero())):
                    return 0
        counter += 1
        len_result = shell_cmd(f"wc -l {cumcp_log_file}")
        if (counter > max_seconds):
            log_print(f"Skip cumcp starting 0 throughput ... ({len_result}) exceeds max_seconds")
            break
        else:
            log_print(f"Skip cumcp starting 0 throughput ... ({len_result}) counter={counter}")
    return 0


################################################################
# Execution start
################################################################


# Keep checking until see "testmac_init" log
wait_for_log(mac_log_file, "TestCUMAC started", wait_for_startup_mac)

# Parse cell_num and show_thrput
line = shell_cmd("grep 'cumac_handler constructed' " + mac_log_file)
show_thrput = 0
cell_num = parse_int(line, "cell_num", "=")
negative_test = 0

# Parse launch pattern file name
launch_pattern = shell_cmd("grep -o launch_pattern.*.yaml " + mac_log_file)

log_print("TestCase: %s cell_num=%s" % (launch_pattern, cell_num))

if (cell_num == 0):
    # Close the log file
    log_print("Error parameters")
    log_print_file.close()
    sys.exit(1)

# Parse expected throughput list
expected_list = parse_expected_thrput(mac_log_file, cell_num)
low_limit_list = copy.deepcopy(expected_list)
high_limit_list = copy.deepcopy(expected_list)

# Check if any cell has non-zero expected throughput
for cell_id in range(0, cell_num):
    if not expected_list[cell_id].is_all_zero():
        expected_zero = False
        break

# Calculate low limitation and high limitation of expected throughput data
for cell_id in range(0, cell_num):
    low_limit_list[cell_id].set_low_limit(CONFIG_ALLOWED_ERROR)
    high_limit_list[cell_id].set_high_limit(CONFIG_ALLOWED_ERROR)
    log_print("Pass criterion low:  Cell " + str(cell_id) + ": " + low_limit_list[cell_id].expected_string())
    log_print("Pass criterion high: Cell " + str(cell_id) + ": " + high_limit_list[cell_id].expected_string())

# Whether cumcp is running on this machine
if os.path.isfile(cumcp_log_file):
    cumcp_exist = True
else:
    cumcp_exist = False

# Whether test_mac is running on this machine
if (show_thrput == 0):
    mac_exist = True
else:
    mac_exist = False

# Wait for testmac and/or cumcp throughput start
if mac_exist:
    wait_for_log(mac_log_file, "Cell  0 |", wait_for_throughput)
elif cumcp_exist:
    if expected_zero:
        log_print(f"The expected data/slot throughput is zero for everything. SLEEPING for {cumcp_zero_timeout} seconds and then exiting cleanly.")
        count = 0
        sleep_increment = 10  # Print a log every 10 seconds to show it isn't hanging.
        while count < cumcp_zero_timeout:
            count += sleep_increment
            time.sleep(sleep_increment)
            log_print(f"Slept {count} seconds. Exiting after {cumcp_zero_timeout} seconds.")
        log_print(f"Slept {count} seconds. Exiting.")
        log_print("Test PASS")
        sys.exit(0)
    else:
        wait_for_cumcp_thrput_start(wait_for_throughput)

if cumcp_exist:
    skip_cumcp_thrput_start(500)

# Skip the first 1 second unstable logs
time.sleep(1)

log_print("Throughput check start ... mac_exist=" + str(mac_exist) + " cumcp_exist=" + str(cumcp_exist))
log_print("===============================================")

# Run log checking for at most duration time
time_counter = 0
total_fail = 0
continuous_fail = 0
max_continuous_fail = 0
test_pass = True

last_mac_result = [Thrput(cell_id) for cell_id in range(cell_num)]
last_cumcp_result = [Thrput(cell_id) for cell_id in range(cell_num)]

mac_errs = [0 for cell_id in range(cell_num)]
cumcp_errs = [0 for cell_id in range(cell_num)]

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
    cumcp_errs = [0 for cell_id in range(cell_num)]

    # Parse and check test_mac throughput
    if mac_exist:
        mac_result = parse_testmac_result(mac_log_file, cell_num)
        check_testmac_thrput(mac_result, low_limit_list, high_limit_list, cell_num, last_mac_result, mac_errs)
        last_mac_result = mac_result

    # Parse and check cumcp throughput
    if cumcp_exist:
        cumcp_result = parse_cumcp_result(cumcp_log_file, cell_num)
        check_cumcp_thrput(cumcp_result, low_limit_list, high_limit_list, cell_num, last_cumcp_result, cumcp_errs)
        last_cumcp_result = cumcp_result

    # Print throughput in console
    for cell_id in range(0, cell_num):
        err_str = ""
        mac_thrput = " MAC:" + mac_result[cell_id].to_string() if mac_exist else ""
        cumcp_thrput = " CUMCP:" + cumcp_result[cell_id].to_string() if cumcp_exist else ""
        err_str += " mac_err=" + hex(mac_errs[cell_id]) if mac_exist else ""
        err_str += " cumcp_err=" + hex(cumcp_errs[cell_id]) if cumcp_exist else ""
        log_print(timestr + " Cell " + str(cell_id) + ":" + mac_thrput + cumcp_thrput + err_str)

    # Add to result counters
    if (sum(mac_errs) == 0 and sum(cumcp_errs) == 0):
        continuous_fail = 0
    else:
        continuous_fail += 1
        total_fail += 1
        max_continuous_fail = continuous_fail if continuous_fail > max_continuous_fail else max_continuous_fail

    # If continuously failed for CONFIG_CONTINUOUS_FAIL_TIME time, treat as fail
    if (continuous_fail >= CONFIG_CONTINUOUS_FAIL_TIME or total_fail >= CONFIG_TOTAL_FAIL_TIME):
        test_pass = False
        break

if CUMCP_LOG is not None:
    CUMCP_LOG.close()

if MAC_LOG is not None:
    MAC_LOG.close()

log_print("Test time: %d seconds, max continuous fail: %d, total fail: %d" % (test_time, max_continuous_fail, total_fail))

for cell_id in range(0, cell_num):
    err_str = "The last fail: Cell " + str(cell_id) + ":"
    if (mac_errs[cell_id] != 0):
        err_str += " " + mac_err_to_string(mac_errs[cell_id])
    if (cumcp_errs[cell_id] != 0):
        err_str += " " + cumcp_err_to_string(cumcp_errs[cell_id])
    if (mac_errs[cell_id] != 0 or cumcp_errs[cell_id] != 0):
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
