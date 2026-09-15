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

import fileinput
import os
import re
import argparse


def update_para_yaml(filepath, paramname, newvalue):
    """Replace the scalar value of a top-level YAML key, preserving any inline comment.

    cuMAC example parameters now live in cuMAC/examples/parameters.yaml and are
    loaded at runtime (see cuMAC/examples/parameters.cpp). Lines look like:

        numActiveUePerCellConst: 500  # 100, 500, 1200. should be <= 2048

    This rewrites only the value token, keeping the trailing comment intact.
    """
    # ^<key>: <value><rest-of-line (e.g. inline comment)>
    pattern = re.compile(rf'^(\s*{re.escape(paramname)}\s*:\s*)(\S+)(.*)$')
    matched = False
    with fileinput.FileInput(filepath, inplace=True) as file:
        for line in file:
            m = pattern.match(line)
            if m:
                matched = True
                line = f"{m.group(1)}{newvalue}{m.group(3)}\n"
            print(line, end='')
    if not matched:
        # fileinput already streamed the (unmodified) file to stdout/back;
        # surface the miss so callers see it in the log.
        print(f"WARNING: key '{paramname}' not found in {filepath}")


def update_para_header(filepath, paramname, newvalue):
    """Legacy fallback: replace a `#define NAME VALUE` line in a C header."""
    param_line = "#define " + paramname
    with fileinput.FileInput(filepath, inplace=True) as file:
        for line in file:
            if param_line in line:
                line = f" {param_line}          {newvalue}\n"
            print(line, end='')


def update_para(filepath, paramname, newvalue):
    """Dispatch to the YAML or legacy-header updater based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".yaml", ".yml"):
        update_para_yaml(filepath, paramname, newvalue)
    else:
        update_para_header(filepath, paramname, newvalue)


def parse_args():
    parser = argparse.ArgumentParser(description="description")
    parser.add_argument('--file', '-f', nargs='?', help='file to be updated', required=True, type=str)
    parser.add_argument('--param', '-p', nargs='?', help='param name', required=True, type=str)
    parser.add_argument('--value', '-v', nargs='?', help='new value', required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    update_para(args.file, args.param, args.value)


main()
