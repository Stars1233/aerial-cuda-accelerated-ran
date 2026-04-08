#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

printenv
gdb -q --batch \
    -ex 'handle SIGHUP nostop pass' \
    -ex 'handle SIGQUIT nostop pass' \
    -ex 'handle SIGPIPE nostop pass' \
    -ex 'handle SIGALRM nostop pass' \
    -ex 'handle SIGTERM nostop pass' \
    -ex 'handle SIGUSR1 nostop pass' \
    -ex 'handle SIGUSR2 nostop pass' \
    -ex 'handle SIGINT nostop pass' \
    -ex 'handle SIGCHLD nostop pass' \
    -ex 'set print thread-events off' \
    -ex 'python import datetime' \
    -ex 'python print("Started",datetime.datetime.now())' \
    -ex 'run' \
    -ex 'python print("Stopped",datetime.datetime.now())' \
    -ex 'thread apply all bt' \
    --tty=/dev/stdout \
    --args $*
