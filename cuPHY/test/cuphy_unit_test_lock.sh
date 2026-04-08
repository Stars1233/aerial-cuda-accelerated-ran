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

LOCKFILE="/var/lock/cuphy_unit_test.lock"

set -m

ACQ=0
REL=0

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
   -a) # Acquire lock
     ACQ=1
     shift
     ;;
   -r) # Release lock
     REL=1
     shift
     ;;
   -*|-h|-?) # Usage
     echo "Usage:"
     echo " -a    Create $LOCKFILE"
     echo " -r    Remove $LOCKFILE"
     shift
     exit
     ;;
   *)
     POSITIONAL_ARGS+=("$1")
     shift
     ;;
   esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [ $ACQ -eq 1 ]
then
   echo "Creating lockfile $LOCKFILE"
   touch $LOCKFILE
elif [ $REL -eq 1 ]
then
   echo "Removing $LOCKFILE"
   rm $LOCKFILE
fi



