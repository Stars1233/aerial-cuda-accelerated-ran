#!/bin/bash

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

if [ "$cuBB_SDK" = "" ]; then
    echo "Please export cuBB_SDK"
    exit 0
fi

lp_files=$1
if [ "$lp_files" = "" ]; then
    echo "Usage: export cuBB_SDK and run add_tv.sh with launch pattern file name. Example:"
    echo "    ./add_tv.sh launch_pattern_F08*"
    echo "    ./add_tv.sh launch_pattern_POC2_1C.yaml"
    exit 0
fi

cd $cuBB_SDK/testVectors/multi-cell

cp ../tv_to_be_updated_by_5gmodel_cicd_pipeline.txt tmp.txt
grep -h -o "path:.*" $lp_files >> tmp.txt

# remove "path:" and space
sed -i "s/path: //g" tmp.txt
sed -i "s/ //g" tmp.txt
sort tmp.txt | uniq > ../tv_to_be_updated_by_5gmodel_cicd_pipeline.txt
rm tmp.txt

