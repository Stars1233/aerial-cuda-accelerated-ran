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

# Check if AERIAL_REPO user argument exists
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

# Use Aerial base image
Stage0 += baseimage(image=f'{AERIAL_REPO}aerial_base:{AERIAL_VERSION_TAG}', _arch=cpu_target, _distro='ubuntu22')

ospackages=[
        'autoconf',
        'automake',
        'autotools-dev',
        'bc',
        'bison',
        'debhelper',
        'check',
        'chrpath',
        'dpatch',
        'ethtool',
        'flex',
        'gdb',
        'git-lfs',
        'help2man',
        'htop',
        'iproute2',
        'jq',
        'libbsd-dev',
        'libcairo2',
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
        'mlocate',
        'net-tools',
        'ninja-build',
        'pciutils',
        'pkg-config',
        'pybind11-dev',
        'python3-apt',
        'python3-cairo',
        'python3-pyelftools',
        'python3-testresources',
        'python3.10-venv',
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

# Install TensorRT for CUDA 13.0 (compatible with CUDA 13.1 runtime)
# TensorRT 10.14.1.48 is the latest version supporting CUDA 13.0
# Note: As of Jan 2026, TensorRT doesn't officially support CUDA 13.1 yet
# CUDA 13.0 libraries work with CUDA 13.1 due to forward compatibility
TENSORRT_VERSION = "10.14.1.48"
TENSORRT_MAJOR = "10.14.1"
CUDA_VERSION = "13.0"
if cpu_target == 'x86_64':
    TENSORRT_ARCH = "x86_64-gnu"
else:
    TENSORRT_ARCH = "aarch64-gnu"

TENSORRT_FILENAME = f"TensorRT-{TENSORRT_VERSION}.Linux.{TENSORRT_ARCH}.cuda-{CUDA_VERSION}.tar.gz"
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

Stage0 += pip(pip="pip3", requirements=f'requirements.txt')

# Install nsight-systems
if cpu_target == 'x86_64':
    cli_package_url = 'https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_1/NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb'

if cpu_target == 'aarch64':
    cli_package_url = 'https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_1/nsight-systems-cli-2026.1.1_2026.1.1.204-1_arm64.deb'

Stage0 += shell(commands=[
    f'wget {cli_package_url}',
    f'dpkg -i {os.path.basename(cli_package_url)}',
    f'rm {os.path.basename(cli_package_url)}',
])

# Workaround - Needed so host-launched graphs appear under the right green context
# Updated for CUDA 13.1
if cpu_target == 'x86_64':
    Stage0 += shell(commands=["cp /usr/local/cuda/lib64/libcupti.so /opt/nvidia/nsight-systems-cli/2026.1.1/target-linux-x64/libcupti.so.13.1"])

if cpu_target == 'aarch64':
    Stage0 += shell(commands=["cp /usr/local/cuda/lib64/libcupti.so /opt/nvidia/nsight-systems-cli/2026.1.1/target-linux-sbsa-armv8/libcupti-sbsa.so.13.1"])

if cpu_target == 'aarch64':
    yq_binary='wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_arm64 -O /usr/bin/yq'
else:
    yq_binary='wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq'
Stage0 += shell(commands=[
    yq_binary,
    'chmod +x /usr/bin/yq',
    ])

Stage0 += user(user='aerial')

Stage0 += workdir(directory='$cuBB_SDK')

