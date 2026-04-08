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
import argparse


def update_para(filepath, paramname, newvalue):
    param_line = "#define " + paramname
    # Open the file in read mode
    with fileinput.FileInput(filepath, inplace=True) as file:
        # Iterate over the lines in the file
        for line in file:
            # Check if the line contains the parameter name
            if param_line in line:
                # Replace the line with the new value
                line = f" {param_line}          {newvalue}\n"
            # Print the line (with the modified value) to the file
            print(line, end='')

    # print(f"Value of {paramname} changed to {newvalue}")


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
