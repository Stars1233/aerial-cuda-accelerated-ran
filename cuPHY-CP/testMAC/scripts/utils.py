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
import re
import time
import subprocess
import traceback
import os.path


# Print logs to both console and file
# log_print_file = open(result_log_file, "w+")
def log_print(log_string):
    print(log_string)
    # log_print_file.write(log_string)
    # log_print_file.write("\n")

# Function to call shell command
def shell_cmd(cmd, print_cmd=False, print_err=True):
    # Print the command for debug if enabled
    if print_cmd: log_print('[shell] %s' % cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    status = 1 if err else 0
    if (status != 0 and print_err):
        log_print('[Shell ERROR] ' + cmd)
        log_print(err)
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
        grep_result = shell_cmd("grep -s -o \"" + expect_string + "\" " + log_file)
        if (len(grep_result) != 0):
            break
        counter += 1
        if (counter > timeout):
            log_print("Wait for log: [" + expect_string + "] ... timeout")
            break
        else:
            log_print("Wait for log [" + expect_string + "] ... counter=" + str(counter))
            time.sleep(1)
    return 0


# Parse a string value after a prefix. Example "prefix=value"
def sed_set(file_path, prefix, subfix, value, debug=False):
    cmd = "sed -i \"s/" + prefix + ".*" + subfix + "/" + prefix + value + subfix + "/\" " + file_path;
    if debug: log_print('[sed] %s' % cmd)
    shell_cmd(cmd)
    return 0

