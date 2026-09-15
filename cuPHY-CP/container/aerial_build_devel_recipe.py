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

"""Aerial Build Development image hpccm recipe using Ubuntu base OS
Usage:
$ hpccm --recipe aerial_build_devel_recipe.py --format docker
"""

import os
from typing import Optional

def required_env(name: str) -> str:
    value: Optional[str] = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set")
    return value

# Check if AERIAL_REPO user argument exists, if not, raise an error.
AERIAL_REPO = USERARG.get('AERIAL_REPO')
if AERIAL_REPO is None:
    raise RuntimeError("User argument AERIAL_REPO must be set")

AERIAL_VERSION_TAG = USERARG.get('AERIAL_VERSION_TAG')
if AERIAL_VERSION_TAG is None:
    raise RuntimeError("User argument AERIAL_VERSION_TAG must be set")


if cpu_target == 'x86_64':
    TARGETARCH='amd64'
elif cpu_target == 'aarch64':
    TARGETARCH='arm64'
else:
    raise RuntimeError("Unsupported platform")

CONTAINER_UBUNTU_DISTRO = required_env("CONTAINER_UBUNTU_DISTRO")

# Use Aerial base image
Stage0 += baseimage(image=f'{AERIAL_REPO}aerial_base:{AERIAL_VERSION_TAG}', _arch=cpu_target, _distro=CONTAINER_UBUNTU_DISTRO)

ospackages=[
        'autoconf',
        'automake',
        'autotools-dev',
        'bc',
        'bison',
        'debhelper',
        'check',
        'chrpath',
        'clang-format',
        'clang-tidy',
        'cmake-format',
        'cppcheck',
        'ethtool',
        'flex',
        'gdb',
        'git-lfs',
        'girepository-2.0',
        'help2man',
        'htop',
        'iproute2',
        'jq',
        'libbsd-dev',
        'libcairo2',
        'libcairo2-dev',
        'libcurl4-openssl-dev',
        'libglib2.0-dev',
        'libjson-c-dev',
        'libltdl-dev',
        'libmnl-dev',
        'libnghttp2-dev',
        'libnl-route-3-dev',
        'libnl-3-dev',
        'libnuma-dev',
        'libpcap-dev',
        'libsubunit0',
        'libsubunit-dev',
        'liburiparser-dev',
        'lsof',
        'libssl-dev',
        'm4',
        'net-tools',
        'ninja-build',
        'pciutils',
        'pkg-config',
        'pybind11-dev',
        'python3-apt',
        'python3-cairo',
        'python3-pyelftools',
        'python3-testresources',
        'python3.12-venv',
        'psmisc',
        'quilt',
        'rt-tests',
        'screen',
        'software-properties-common',
        'swig',
        'tcpdump',
        'tmux',
        'numactl',
        'zip',
        'binutils-dev',     # Needed for backward-cpp to pretty-print and elaborated stacktrace
        'libdwarf-dev',     # Needed for backward-cpp to pretty-print and elaborated stacktrace
        ]

Stage0 += user(user='root')
Stage0 += packages(ospackages=ospackages)

TENSORRT_VERSION = required_env("TENSORRT_VERSION")
TENSORRT_MAJOR = ".".join(TENSORRT_VERSION.split(".")[:3])
TENSORRT_CUDA_VERSION = required_env("TENSORRT_CUDA_VERSION")
if cpu_target == 'x86_64':
    TENSORRT_ARCH = "x86_64-gnu"
else:
    TENSORRT_ARCH = "aarch64-gnu"

TENSORRT_FILENAME = f"TensorRT-{TENSORRT_VERSION}.Linux.{TENSORRT_ARCH}.cuda-{TENSORRT_CUDA_VERSION}.tar.gz"
TENSORRT_URL = f"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/{TENSORRT_MAJOR}/tars/{TENSORRT_FILENAME}"

# Download and install TensorRT
Stage0 += shell(commands=[
    f'wget -q {TENSORRT_URL} -O /tmp/{TENSORRT_FILENAME}',
    f'tar -xzf /tmp/{TENSORRT_FILENAME} -C /tmp/',
    f'cp -Pr /tmp/TensorRT-{TENSORRT_VERSION}/lib/* /usr/local/lib/',
    f'cp -P /tmp/TensorRT-{TENSORRT_VERSION}/bin/* /usr/local/bin/',
    f'cp -r /tmp/TensorRT-{TENSORRT_VERSION}/include/* /usr/local/include/',
    'ldconfig',
    f'rm -rf /tmp/TensorRT-{TENSORRT_VERSION} /tmp/{TENSORRT_FILENAME}',
])

Stage0 += environment(variables={
    "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH:/usr/local/lib",
})

# Screen setup
Stage0 += shell(commands=[
    'echo "logfile screenlog_%t.log" >> /etc/screenrc',
    'echo "logfile flush 1" >> /etc/screenrc',
    'echo "defshell -bash" >> /etc/screenrc',
    ])

Stage0 += copy(src='requirements.txt', dest='/tmp/')
Stage0 += shell(commands=[
    'uv pip install --system --break-system-packages -r /tmp/requirements.txt',
    'rm /tmp/requirements.txt',
])

# Install Nsight Systems
if cpu_target == 'x86_64':
    cli_package_url = 'https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_3/NsightSystems-linux-cli-public-2026.3.1.157-3804839.deb'

if cpu_target == 'aarch64':
    cli_package_url = 'https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_3/nsight-systems-cli-2026.3.1_2026.3.1.157-1_arm64.deb'

Stage0 += shell(commands=[
    f'wget {cli_package_url}',
    f'dpkg -i {os.path.basename(cli_package_url)}',
    f'rm {os.path.basename(cli_package_url)}',
])

if cpu_target == 'aarch64':
    yq_binary='wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_arm64 -O /usr/bin/yq'
else:
    yq_binary='wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq'
Stage0 += shell(commands=[
    yq_binary,
    'chmod +x /usr/bin/yq',
    ])

# Needed by data_lake
Stage0 += shell(commands=[
    'wget -O /usr/include/zmq.hpp https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp',
    'wget -O /usr/include/zmq_addon.hpp https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq_addon.hpp',
    ])

Stage0 += user(user='aerial')

Stage0 += workdir(directory='$cuBB_SDK')
