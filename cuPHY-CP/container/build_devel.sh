#!/bin/bash -e

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

# Exit on first error
set -e

# Switch to SCRIPT_DIR directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
echo $SCRIPT starting...
cd $SCRIPT_DIR
source ./setup.sh

if [[ "$AERIAL_VERSION_TAG" == "$(unset AERIAL_VERSION_TAG && source ./setup.sh && echo $AERIAL_VERSION_TAG)" ]]
then
    echo "Error: Do not run this script without setting the AERIAL_VERSION_TAG variable"
    exit 1
fi

TARGETARCH=$(basename $AERIAL_PLATFORM)
case "$TARGETARCH" in
    "amd64")
        CPU_TARGET=x86_64
        ;;
    "arm64")
        CPU_TARGET=aarch64
        ;;
    *)
        echo "Unsupported target architecture"
        exit 1
        ;;
esac

hpccm --recipe aerial_build_devel_recipe.py --cpu-target $CPU_TARGET --format docker --userarg AERIAL_REPO=$AERIAL_REPO AERIAL_VERSION_TAG=$AERIAL_VERSION_TAG > Dockerfile_tmp
if [[ -n "$AERIAL_BUILDER" ]]
then
    docker buildx build --builder $AERIAL_BUILDER --pull --load --platform $AERIAL_PLATFORM -t $AERIAL_REPO$AERIAL_IMAGE_NAME:${AERIAL_VERSION_TAG} -f Dockerfile_tmp .
else
    DOCKER_BUILDKIT=1 docker build --network host --no-cache --platform $AERIAL_PLATFORM -t $AERIAL_REPO$AERIAL_IMAGE_NAME:${AERIAL_VERSION_TAG} -f Dockerfile_tmp .
fi
rm Dockerfile_tmp
