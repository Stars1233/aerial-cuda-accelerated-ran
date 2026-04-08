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

NVIPC_HOME=$(dirname $(readlink -f "$0"))

if [ `whoami` = "root" ];then
    USE_SUDO=""
else
    USE_SUDO="sudo -E"
fi

# Run under build folder
echo "$USE_SUDO ./nvIPC/tests/example/test_ipc $@"
$USE_SUDO ./nvIPC/tests/example/test_ipc $@
