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

import os
import sys
import subprocess
import os.path

log_path = os.getenv('LOG_PATH')
if log_path is None:
    log_path = os.getenv('cuBB_SDK') + "/logs"

if log_path is None:
    print("Please set LOG_PATH or cuBB_SDK first")
    exit(0)

ru_log_file = log_path + "/screenlog_ru.log"
result_log_file = log_path + "/check_aggregate_result.log"

# Print logs to both console and file
log_print_file = open(result_log_file, "w+")

def log_print(log_string):
    print(log_string)
    log_print_file.write(log_string)
    log_print_file.write("\n")
    log_print_file.flush()

# Print parameters
log_print("===============================================")
log_print("ru_log=" + ru_log_file)
log_print("result_log=" + result_log_file)

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
        log_print(err)
    return str(output.strip(), encoding="utf-8")

################################################################
# Execution start
################################################################

# Whether ru_emulator is running on this machine
if os.path.isfile(ru_log_file):
    ru_exist = True
else:
    ru_exist = False

test_pass = True
if ru_exist:
    if os.getenv('ON_TIME_SLOT_PERCENTAGE_THRESHOLDS') is None:
        log_print('ENV variable ON_TIME_SLOT_PERCENTAGE_THRESHOLDS not set - not checking on-time percentages')
    else:
        # The ENV parameter is passed like [thresh:thresh:thresh:thresh:thresh] - example [99.9:99.9:99.9:99.9:99.9]
        # So use [1:-1] to get rid of the brackets, and split on :
        thresholds = os.getenv('ON_TIME_SLOT_PERCENTAGE_THRESHOLDS')[1:-1].split(':')
        print(f"DEBUG - THRESHOLDS = {thresholds}")
        dl_c_threshold = thresholds[0]
        dl_u_threshold = thresholds[1]
        ul_c_threshold = thresholds[2]
        worst_slot_per_cell_ul_c_threshold = thresholds[3]  # Worst slot for each cell reported separately
        worst_slot_avg_cell_ul_c_threshold = thresholds[4]  # Worst slot but averaged accross cells


        # Check the ru-emulator log for on-time slot percentage
        #
        # Examples for messages in log file
        # 02:21:21.937337 [RU] | DL C MIN ON TIME SLOTS %       |      99.97% |
        # 02:21:21.937345 [RU] | DL U MIN ON TIME SLOTS %       |      99.97% |
        # 02:21:21.937353 [RU] | UL C MIN ON TIME SLOTS %       |      99.98% |

        # REGEX to use in ru_emulator file, on-time percentage threshold to meet
        on_time_checks = [
            {"regex": "DL C MIN ON TIME SLOTS %", "threshold": float(dl_c_threshold)},
            {"regex": "DL U MIN ON TIME SLOTS %", "threshold": float(dl_u_threshold)},
            {"regex": "UL C MIN ON TIME SLOTS %", "threshold": float(ul_c_threshold)},
        ]
        for check in on_time_checks:
            result = shell_cmd(f"grep \"{check['regex']}\" {ru_log_file}").strip().split("|")[-2].strip()
            result = float(result[:-1])  # Remove the % and convert to float
            if result < check['threshold']:
                test_pass = False
                log_print(f"TEST FAILED - {check['regex']} {result} is less than expected {check['threshold']}")
            else:
                log_print(f"{check['regex']} {result} matches expectations of {check['threshold']}")

        # REGEX to use in ru_emulator file, WORST ONTIME U-Plane - this is per-cell
        # Example (12 C test case):
        # 18:13:17.928199 [RU] | WORST ONTIME U-Plane Slot %    | S76  98.68% | S06  99.63% | S76  98.66% | S06  99.58% | S76  98.76% | S06  99.59% | S76  98.73% | S06  99.62% | S76  98.77% | S06  99.74% | S76  98.82% | S06  99.71% |
        worst_slot_per_cell_ontime_uplane_checks = [
            {"regex": "WORST ONTIME U-Plane Slot %", "threshold": float(worst_slot_per_cell_ul_c_threshold)},
        ]
        for check in worst_slot_per_cell_ontime_uplane_checks:
            result = shell_cmd(f"grep \"{check['regex']}\" {ru_log_file}").strip().split("|")
            result = result[2:-1]  # the list is now all of the entries for each of the cells in the format " SLOT PERCENTAGE "
            cell_num = 0
            for cell_result in result:
                cell_result = cell_result.strip()      # Remove extra whitespace
                cell_result = cell_result.split()[1]   # Get the numeric percentage with %
                cell_result = float(cell_result[:-1])  # Remove the % character and convert to float
                if cell_result < check['threshold']:
                    test_pass = False
                    log_print(f"TEST FAILED - cell {cell_num} - {check['regex']} {cell_result} is less than expected {check['threshold']}")
                else:
                    log_print(f"cell {cell_num} - {check['regex']} {cell_result} matches expectations of {check['threshold']}")

                cell_num += 1

        # This is the worst on time percentage per slot where each slot is averaged across all cells
        # Example
        # 20:27:31.342057 [RU] | WORST AVG ONTIME U Slot %      | S46  96.63% |
        worst_slot_avg_cell_ontime_uplane_checks = [
            {"regex": "WORST AVG ONTIME U Slot %", "threshold": float(worst_slot_avg_cell_ul_c_threshold)},
        ]
        for check in worst_slot_avg_cell_ontime_uplane_checks:
            result = shell_cmd(f"grep \"{check['regex']}\" {ru_log_file}").strip().split("|")
            result = result[2:-1]        # The list is now just the one entry in the format " SLOT PERCENTAGE "
            result = result[0].strip()   # Remove extra whitespace
            result = result.split()[1]   # Just get the numeric value with % character
            result = float(result[:-1])  # Remove % character and convert to float
            if result < check['threshold']:
                test_pass = False
                log_print(f"TEST FAILED - {check['regex']} {result} is less than expected {check['threshold']}")
            else:
                log_print(f"{check['regex']} {result} matches expectations of {check['threshold']}")

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
