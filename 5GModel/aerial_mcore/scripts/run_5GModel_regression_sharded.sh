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

# Exit on first error
set -e

# Configure channels
channels=("ssb" "pdcch" "pdsch" "csirs" "dlmix" "prach" "pucch" "pusch" "srs" "ulmix" "bfw" "cuPHY-CP_TVs" "perf_TVs" "cfgTemplate" "launchPatternFile" "CfgTV_nvbug")

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
echo $SCRIPT starting...
cd $PROJECT_ROOT

# Setup script
source scripts/setup.sh

# Install developer mode package
scripts/install_dev_pkg.sh

# Run example scripts
pushd examples
exitcode=0
pids=()
stats=()

for (( idx=0; idx<${#channels[@]}; idx++)); do
   channel="${channels[$idx]}"
   echo Starting channel $channel
   python3 -u ./example_5GModel_regression.py $channel > result_$channel.txt 2>&1 &
   pids+=("$!")
done

for (( idx=0; idx<${#channels[@]}; idx++)); do
   pid="${pids[$idx]}"
   channel="${channels[$idx]}"
   echo Waiting for channel $channel on pid $pid
   if wait $pid; then
      stats+=("PASS")
      echo "Channel $p success"
   else
      echo "Channel $p failure"
      stats+=("FAIL")
      exitcode=1
   fi
done

for (( idx=0; idx<${#channels[@]}; idx++)); do
   echo
   echo "----------------------------------------------------------------------"
   echo "Log for channel $channel"
   echo
   channel="${channels[$idx]}"
   cat result_$channel.txt
done

echo
echo "========================================================================="
echo "Summary Results"
echo
for (( idx=0; idx<${#channels[@]}; idx++)); do
   channel="${channels[$idx]}"
   status="${stats[$idx]}"
   echo "Channel $channel status $status"
done
popd


# Finished
echo $SCRIPT finished
exit $exitcode
