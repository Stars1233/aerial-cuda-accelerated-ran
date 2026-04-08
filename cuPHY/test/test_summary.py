#!/bin/python3

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

import pandas
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vector',  nargs='?', help='Test Vector to log')
parser.add_argument('-t', '--test',    nargs='?', help='Test type (e.g. Sanitizer) Run')
parser.add_argument('-r', '--results', nargs='?', help='Results (e.g. number of errors)')
parser.add_argument('-l', '--log',     default='testSummary.csv', help='File to record logs')

args=parser.parse_args()

if(args.test != None and args.results != None and args.vector != None):
    try:
        testSummary = pandas.read_csv(args.log,index_col=0)
    except:
        testSummary = pandas.DataFrame()

    # Just use the filename if a path was provided
    testvector = args.vector.split('/')[-1]
    testSummary.loc[testvector,args.test] = int(args.results)

    testSummary.to_csv(args.log)
