#!/usr/bin/env python

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

# Run like this:
# make -j 20 2>&1 | ../util/ptxas_lcheck.py

import fileinput
import re
import subprocess

re_func_str  = r'ptxas info    : Function properties for (.*)'
re_local_str = r'    (\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads'

func_name = ''

kernels_with_lmem = []

for line in fileinput.input():
    line = line.rstrip()
    if not func_name:
        m = re.search(re_func_str, line)
        if m:
            func_name = m.group(1)
            #print('function: %s' % func_name)
        else:
            func_name = ''
    else:
        m = re.search(re_local_str, line)
        if m:
            if (int(m.group(1)) != 0) or (int(m.group(2)) != 0) or (int(m.group(3)) != 0):
                #print(line)
                cmd = 'c++filt %s' % func_name
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                func_name_demangled = proc.stdout.readline().decode('ascii')
                kernels_with_lmem.append((func_name_demangled.rstrip(), line))
        else:
            func_name = ''
    print(line)

kernels_with_lmem_sorted = sorted(kernels_with_lmem, key=lambda t:t[0])

print('Kernels with local memory usage:')
for k in kernels_with_lmem_sorted:
    print(k[0])
    print(k[1])
        
