#!/bin/bash

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


# Script to print UUID for 5GModel in aerial_sdk

SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

LC_ALL=C

cd $cuBB_SDK

# Build grep -v filter to exclude certain files and directories from UUID creation
# This way we can ignore files that do not affect the outcome of TV generation.
# e.g. egrep -v "\.md$|5GModel/documents"
non_uuid="\.md$"
non_uuid="${non_uuid}|5GModel/aerial_mcore/aerial_pkg"
non_uuid="${non_uuid}|5GModel/nr_matlab/CompilerSDKOutput"
non_uuid="${non_uuid}|5GModel/nr_matlab/scripts"
non_uuid="${non_uuid}|5GModel/documents"

find 5GModel -type f -exec md5sum {} \; | egrep -v "$non_uuid" | sort -fid -k 2 | md5sum - | awk '{ print $1 }'
