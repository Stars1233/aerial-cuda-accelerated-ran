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

# USAGE: PYAERIAL_IMAGE=<image> $cuBB_SDK/pyaerial/container/run.sh

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
host_cuBB_SDK=$(builtin cd $SCRIPT_DIR/../..;pwd)
echo $SCRIPT starting...
source $host_cuBB_SDK/cuPHY-CP/container/setup.sh


if [ -z "$1" ]; then
   echo Start container instance at bash prompt
   CMDS="/bin/bash"
else
   CMDS="$@"
   echo Run command then exit container
   echo Command: $CMDS
fi

if [[ ! $(grep -i "export PS1=\"\[host:" ~/.bashrc) ]]; then
    CMDS="echo 'export PS1=\"[host: $host_cuBB_SDK] \$PS1\"' >> ~/.bashrc && $CMDS"
fi

AERIAL_PLATFORM=${AERIAL_PLATFORM:-amd64}
TARGETARCH=$(basename $AERIAL_PLATFORM)

if [[ -z $PYAERIAL_IMAGE ]]; then
   PYAERIAL_IMAGE=pyaerial:${USER}-${AERIAL_VERSION_TAG}-${TARGETARCH}
fi

docker run --privileged \
            -it --rm \
            $AERIAL_EXTRA_FLAGS \
            --gpus all \
            --name pyaerial_$USER \
            --add-host pyaerial_$USER:127.0.0.1 \
            --network host --shm-size=4096m \
            --device=/dev/gdrdrv:/dev/gdrdrv \
            -u $USER_ID:$GROUP_ID \
            -w /opt/nvidia/cuBB \
            -v $host_cuBB_SDK:/opt/nvidia/cuBB \
            -v $host_cuBB_SDK:/opt/nvidia/aerial_sdk \
            -v /dev/hugepages:/dev/hugepages \
            -v /lib/modules:/lib/modules \
            -v /var/log/aerial:/var/log/aerial \
            -e host_cuBB_SDK=$host_cuBB_SDK \
            --userns=host --ipc=host \
            $PYAERIAL_IMAGE fixuid -q /bin/bash -c "$CMDS"
