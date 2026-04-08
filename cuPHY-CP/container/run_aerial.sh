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

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
host_cuBB_SDK=$(builtin cd $SCRIPT_DIR/../..;pwd)
echo $SCRIPT starting...
source $SCRIPT_DIR/setup.sh

# Help function
show_help() {
    cat << EOF
Usage: $(basename $0) [OPTIONS] [COMMAND]

Run NVIDIA Aerial container as a development environment for building and testing
the Aerial SDK.

CONTAINER ENVIRONMENT:
    The host Aerial SDK directory is mounted into the container at /opt/nvidia/cuBB.
    The host path is available inside the container via the environment variable:
        \$host_cuBB_SDK

    This allows seamless development with host files accessible inside the container.

OPTIONS:
    -d, --detached       Run container in detached mode
                         Without command: keeps container alive with tail -f /dev/null
                         With command: runs the specified command in background

    -p, --modify-prompt  Modify bash prompt inside container to show host SDK path
                         Adds "[host: /path/to/sdk]" prefix to PS1

    -h, --help           Display this help message and exit

COMMAND:
    Optional command to run inside the container.
    If not specified, starts an interactive bash shell (or tail in detached mode).

EXAMPLES:
    # Start interactive container
    $(basename $0)

    # Start detached container (keeps running)
    $(basename $0) -d

    # Run command in interactive mode
    $(basename $0) ./build_script.sh

    # Run command in detached mode, use attach_aerial.sh to attach to the container
    $(basename $0) -d

    # Start with custom prompt
    $(basename $0) -p

    # Combine flags
    $(basename $0) -d -p

DETACHED MODE USAGE:
    After starting in detached mode, you can:
    - Exec shell:   docker exec -it c_aerial_\$USER /bin/bash
    - Use helper:   ./attach_aerial.sh
    - Stop:         docker stop c_aerial_\$USER

EOF
    exit 0
}

# Parse command-line flags
DETACHED_FLAG=""
MODIFY_PROMPT_FLAG=""
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -d|--detached)
            DETACHED_FLAG="-d"
            echo "Detached mode enabled"
            shift
            ;;
        -p|--modify-prompt)
            MODIFY_PROMPT_FLAG="1"
            echo "Prompt modification enabled"
            shift
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set commands based on remaining arguments
if [ ${#REMAINING_ARGS[@]} -eq 0 ]; then
   if [ -n "$DETACHED_FLAG" ]; then
      echo Start container instance in detached mode with keep-alive
      CMDS="tail -f /dev/null"
   else
      echo Start container instance at bash prompt
      CMDS="/bin/bash"
   fi
else
   CMDS="${REMAINING_ARGS[@]}"
   if [ -n "$DETACHED_FLAG" ]; then
      echo Run command in detached mode
      echo Command: $CMDS
   else
      echo Run command then exit container
      echo Command: $CMDS
   fi
fi

has_docker_gpu() {
  docker info --format '{{json .Runtimes}}' 2>/dev/null \
    | grep -q '"nvidia"' \
    && command -v nvidia-container-cli >/dev/null 2>&1 \
    && nvidia-container-cli info >/dev/null 2>&1
}

GPU_FLAG=""
GDRDRV_DEVICE_FLAG=""
server_id=""
[ -f "/sys/devices/virtual/dmi/id/board_vendor" ] && server_id=$(cat /sys/devices/virtual/dmi/id/board_vendor)"-"
[ -f "/sys/devices/virtual/dmi/id/board_name" ] && server_id+=$(cat /sys/devices/virtual/dmi/id/board_name)
echo "server_id: $server_id"
if [[ $server_id == "NVIDIA-P4242" ]]; then
    AERIAL_CHECK_GDRDRV=0
    echo "Skipping gdrdrv check for DGX Spark"
fi
if has_docker_gpu; then
  GPU_FLAG="--gpus all"
  if [ "${AERIAL_CHECK_GDRDRV:-1}" -eq 1 ] && ! modinfo gdrdrv > /dev/null 2>&1; then
      echo ""
      echo "Please download and install from https://developer.nvidia.com/gdrcopy.  Or set AERIAL_CHECK_GDRDRV=0 and re-run"
      exit 1
  else
      GDRDRV_DEVICE_FLAG="--device=/dev/gdrdrv:/dev/gdrdrv"
  fi
else
  echo This system has no GPU, running without --gpus all parameter
  echo Creating soft link for libcuda.so.1 for RU Emulator dependency
  CMDS="sudo ln -s /usr/local/cuda/compat/libcuda.so.1 /usr/lib/\$(arch)-linux-gnu/libcuda.so.1 && $CMDS"
fi

if [[ "$(arch)" == "aarch64" ]]; then
    if [[ "$(lscpu | grep -s Cortex- )" == *"A78"* ]]; then
        AERIAL_VERSION_TAG=${AERIAL_VERSION_TAG}-bf3
    fi
fi

if [ -n "$MODIFY_PROMPT_FLAG" ]; then
    if [[ ! $(grep -i "export PS1=\"\[host:" ~/.bashrc) ]]; then
        CMDS="echo 'export PS1=\"[host: $host_cuBB_SDK] \$PS1\"' >> ~/.bashrc && $CMDS"
    fi
fi

echo Command: $CMDS

# Check if running in a TTY (but not in detached mode)
if [ -z "$DETACHED_FLAG" ] && [ -t 0 ]; then
    INTERACTIVE_FLAG="-it"
    echo "TTY detected - running in interactive mode"
else
    INTERACTIVE_FLAG=""
    if [ -n "$DETACHED_FLAG" ]; then
        echo "Running in detached mode (no interactive flags)"
    else
        echo "No TTY detected - running in non-interactive mode"
    fi
fi

docker pull --platform=$AERIAL_PLATFORM $AERIAL_REPO$AERIAL_IMAGE_NAME:$AERIAL_VERSION_TAG
if [[ "$?" != "0" ]]; then
    echo "WARNING - The docker pull for $AERIAL_REPO$AERIAL_IMAGE_NAME:$AERIAL_VERSION_TAG with platform $AERIAL_PLATFORM FAILED"
    echo "You may have an image locally that could be used. This may be stale."
    read -p "Do you want to continue? y/n " ret
    if [[ "$ret" == "y" ]]; then
        echo "Continuing..."
    else
        echo "Exiting."
        exit 1
    fi
fi
docker run --platform=$AERIAL_PLATFORM \
    --privileged \
    $DETACHED_FLAG $INTERACTIVE_FLAG --rm \
    $AERIAL_EXTRA_FLAGS \
    $GPU_FLAG \
    --name c_aerial_$USER \
    --hostname c_aerial_$USER \
    --add-host c_aerial_$USER:127.0.0.1 \
    --network host --shm-size=4096m \
    $GDRDRV_DEVICE_FLAG \
    -u $USER_ID:$GROUP_ID \
    -w /opt/nvidia/cuBB \
    -v $host_cuBB_SDK:/opt/nvidia/cuBB \
    -v /mnt/cicd_tvs:/mnt/cicd_tvs \
    -v /var/lock:/var/lock \
    -v /dev/hugepages:/dev/hugepages \
    -v /lib/modules:/lib/modules \
    -v /etc:/host/etc \
    -v /run/systemd/system:/run/systemd/system \
    -v /var/run/dbus/system_bus_socket:/var/run/dbus/system_bus_socket \
    -v /var/log:/host/var/log:ro \
    -e host_cuBB_SDK=$host_cuBB_SDK \
    --userns=host --ipc=host -v /var/log/aerial:/var/log/aerial \
    $AERIAL_REPO$AERIAL_IMAGE_NAME:$AERIAL_VERSION_TAG fixuid /bin/bash -c "$CMDS"
RETVAL=$?

if [ -n "$DETACHED_FLAG" ]; then
    echo ""
    echo "Container started in detached mode: c_aerial_$USER"
    echo "To exec shell:  docker exec -it c_aerial_$USER /bin/bash or use attach_aerial.sh"
    echo "To stop:        docker stop c_aerial_$USER"
fi
exit $RETVAL
