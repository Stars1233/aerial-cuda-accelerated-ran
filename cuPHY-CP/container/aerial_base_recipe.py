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

"""Aerial Base image hpccm recipe using Ubuntu base OS
Usage:
$ hpccm --recipe aerial_base_recipe.py --format docker
"""

import os
from typing import Optional

def required_env(name: str) -> str:
    value: Optional[str] = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set")
    return value

if cpu_target == 'x86_64':
    TARGETARCH='amd64'
elif cpu_target == 'aarch64':
    TARGETARCH='arm64'
else:
    raise RuntimeError("Unsupported platform")

AERIAL_GPU_TYPE = USERARG.get('AERIAL_GPU_TYPE')
DPDK_TAG = USERARG.get('DPDK_TAG', '')
CONTAINER_CUDA_IMAGE = required_env("CONTAINER_CUDA_IMAGE")
CONTAINER_UBUNTU_DISTRO = required_env("CONTAINER_UBUNTU_DISTRO")
CONTAINER_GDRCOPY_VERSION = required_env("CONTAINER_GDRCOPY_VERSION")
DOCA_VERSION = required_env("DOCA_VERSION")
DOCA_BUILD = required_env("DOCA_BUILD")
CONTAINER_DOCA_UBUNTU_VERSION = required_env("CONTAINER_DOCA_UBUNTU_VERSION")
MATHDX_VERSION = required_env("MATHDX_VERSION")
MATHDX_CUDA_VERSION = required_env("MATHDX_CUDA_VERSION")
DOCA_HOST_ID = f"{DOCA_VERSION}-{DOCA_BUILD}-{CONTAINER_DOCA_UBUNTU_VERSION}"
MATHDX_ARCHIVE = f"nvidia-mathdx-{MATHDX_VERSION}-{MATHDX_CUDA_VERSION}.tar.gz"

stages.extend([hpccm.Stage()])
Bootstrap = stages[0]
Externals = stages[1]
Base      = stages[2]


Bootstrap += baseimage(image=CONTAINER_CUDA_IMAGE, _arch=cpu_target, _distro=CONTAINER_UBUNTU_DISTRO, _as='bootstrap')

if cpu_target == 'aarch64':
    Bootstrap += environment(variables={
        'CUDA_ROOT': '/usr/local/cuda',
        'CUDA_INC_DIR': '/usr/local/cuda/include'
        })

# Apt configuration to retry on network errors
Bootstrap += shell(commands=[
    'echo \'Acquire::Retries "3";\' > /etc/apt/apt.conf.d/80-retries'
    ])

# Pin CUDA nvcc to a specific version from the base image's CUDA repo so it
# doesn't drift on apt-get upgrade. 13.3.33-1 is available in the
# developer.download.nvidia.com repo the base image already configures.
_CUDA_NVCC_VERSION = '13.3.33-1'

# CUDA's rolling APT repository can publish new package versions after a
# versioned CUDA base image is released. Hold every installed CUDA component
# before the general OS upgrade, rather than only cuda-nvcc, so a rebuild keeps
# the complete toolkit and compatibility-library set from the base image.
#
# These are dpkg-query patterns, quoted below so the shell does not expand them.
# Keep the set limited to CUDA/NVIDIA toolkit components: do not accidentally
# hold unrelated Ubuntu packages such as libcurl.
_CUDA_APT_PACKAGE_PATTERNS = (
    'cccl-*',
    'cuda',
    'cuda-*',
    'gds-tools-*',
    'libcublas-*',
    'libcudla-*',
    'libcufft-*',
    'libcufile-*',
    'libcuobjclient-*',
    'libcurand-*',
    'libcusolver-*',
    'libcusparse-*',
    'libnpp-*',
    'libnvfatbin-*',
    'libnvjitlink-*',
    'libnvjpeg-*',
    'libnvptxcompiler-*',
    'libnvvm-*',
    # Nsight Systems is installed explicitly in aerial_build_devel_recipe.py.
    # Hold only the CUDA base image's Nsight Compute packages here.
    'nsight-compute*',
    'nvidia-gds*',
)
_CUDA_APT_PACKAGE_QUERY = ' '.join(
    f"'{pattern}'" for pattern in _CUDA_APT_PACKAGE_PATTERNS
)
# Ubuntu 24.04 ships an 'ubuntu' user at uid/gid 1000; move it out of the
# way before we create the 'aerial' user at the same uid/gid.
Bootstrap += shell(commands=[
    'groupmod -g 1001 ubuntu',
    'usermod -u 1001 ubuntu',
    ])
Bootstrap += shell(commands=[
    'apt-get update -y',
    f'apt-get install -y --allow-downgrades cuda-nvcc-13-3={_CUDA_NVCC_VERSION}',
    (
        # dpkg retains records for removed packages; only installed packages
        # can be passed to apt-mark hold.
        "{ dpkg-query -W -f='${binary:Package} ${db:Status-Status}\\n' "
        f'{_CUDA_APT_PACKAGE_QUERY} 2>/dev/null || true; }} '
        "| awk '$2 == \"installed\" { print $1 }' "
        '| xargs --no-run-if-empty apt-mark hold'
    ),
    'apt-get upgrade -y',
    'rm -rf /var/lib/apt/lists/*',
    ])

Bootstrap += packages(ospackages=[
    'curl',
    'doxygen',
    'gcc-13',
    'g++-13',
    'git',
    'graphviz',
    'hdf5-tools',
    'python3-dev',
    'wget',
    ])

# The upstream fixuid v0.6.0 binary embeds unsupported Go 1.20.7 standard
# library code with known vulnerabilities.  Rebuild the unchanged, pinned
# source with a supported Go toolchain instead of installing that prebuilt
# binary.  The builder is pinned by its multi-architecture manifest digest and
# mounted for this RUN only.  The source tree is removed, so neither the Go
# toolchain nor the source is included in the resulting image layer.
FIXUID_COMMIT = 'c639d9b3d183248f8de031679654bf2146f88083'
FIXUID_GO_VERSION = '1.27.1'
FIXUID_GO_BUILDER = 'golang@sha256:648f440f42a0958804efb24df176f806f9d353b41f1c0627f666428e40310f6b'

Bootstrap += generic_build(
    repository='https://github.com/boxboat/fixuid.git',
    commit=FIXUID_COMMIT,
    directory='fixuid',
    build=[
        f'test "$(git rev-parse HEAD)" = "{FIXUID_COMMIT}"',
        f'CGO_ENABLED=0 GOOS=linux GOARCH={TARGETARCH} GOTOOLCHAIN=local '
        f'/usr/local/go/bin/go build -mod=readonly -trimpath -buildvcs=true '
        f'-o /usr/local/bin/fixuid .',
        f"/usr/local/go/bin/go version /usr/local/bin/fixuid | grep -Fq 'go{FIXUID_GO_VERSION}'",
        ],
    _run_arguments=f'--mount=from={FIXUID_GO_BUILDER},source=/usr/local/go,target=/usr/local/go',
    )

# Configure wget with retry logic to handle intermittent GitHub CDN issues
Bootstrap += shell(commands=[
    'printf "retry_connrefused = on\\n'
    'retry_on_http_error = 503,504,429\\n'
    'waitretry = 10\\n'
    'read_timeout = 60\\n'
    'timeout = 30\\n'
    'tries = 20\\n'
    'dns_timeout = 30\\n" > /etc/wgetrc'
    ])

# CMake
Bootstrap += cmake(eula=True, version='3.28.3')

TRIPLET = '{}-linux-gnu'.format(cpu_target)

# Export canonical CC/CXX so any tool that resolves the compiler from the
# environment (make, autoconf, etc.) gets the triplet binary explicitly.
Bootstrap += environment(variables={
    'CC':  '/usr/bin/{t}-gcc'.format(t=TRIPLET),
    'CXX': '/usr/bin/{t}-g++'.format(t=TRIPLET),
})

# GDRCopy
Bootstrap += gdrcopy(version=CONTAINER_GDRCOPY_VERSION, ldconfig=True)

#
# Externals stage
#
Externals += baseimage(image='bootstrap', _arch=cpu_target, _distro=CONTAINER_UBUNTU_DISTRO, _as='externals')

# Needed by libyaml, mimalloc, prometheus
Externals += packages(ospackages=[
    'automake',
    'libcurl4-openssl-dev',
    'libdw-dev',
    'libpcre2-dev',
    'libssl-dev',
    'libtool',
    'meson',
    'pkg-config',
    ])


# Using this variable to fix issue with multi copy of runtime below
externals = generic_cmake(
    repository='https://github.com/google/googletest.git',
    # branch='v1.17.0',
    commit='52eb8108c5bdec04579160ae17225d66034bd723',
    recursive=True,
    preconfigure=[
        # Re-sync submodules to the pinned commit's gitlinks: hpccm's recursive
        # clone leaves them at live HEAD, not what the pinned commit records.
        'git submodule update --init --recursive',
        'mkdir -p /usr/local/share/licenses/googletest',
        'cp LICENSE /usr/local/share/licenses/googletest/ 2>/dev/null',
        ],
    )
Externals += externals

#https://github.com/google/benchmark/blob/main/README.md#installation
Externals += generic_cmake(
    cmake_opts=['-DBENCHMARK_ENABLE_GTEST_TESTS=OFF', '-DBENCHMARK_ENABLE_TESTING=OFF'],
    repository='https://github.com/google/benchmark.git',
    # branch='v1.9.4',
    commit='eddb0241389718a23a42db6af5f0164b6e0139af',
    recursive=True,
    preconfigure=[
        # Re-sync submodules to the pinned commit's gitlinks: hpccm's recursive
        # clone leaves them at live HEAD, not what the pinned commit records.
        'git submodule update --init --recursive',
        'mkdir -p /usr/local/share/licenses/benchmark',
        'cp LICENSE /usr/local/share/licenses/benchmark/ 2>/dev/null',
        ],
    )

Externals += generic_build(
    build=[
        'mkdir -p /usr/local/share/licenses/libyaml',
        'cp License /usr/local/share/licenses/libyaml/ 2>/dev/null',
        './bootstrap','./configure','make'
    ],
    install=['make install'],
    repository='https://github.com/yaml/libyaml.git',
    # branch='0.2.5',
    commit='2c891fc7a770e8ba2fec34fc6b545c672beb37e6',
    )

Externals += generic_cmake(
    repository='https://github.com/CESNET/libyang.git',
    # branch='v3.13.5',
    commit='efe43e3790822a3dc64d7d28db935d03fff8b81f',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/libyang',
        'cp LICENSE /usr/local/share/licenses/libyang/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    cmake_opts=['-DWITH_DOCS=OFF', '-DBUILD_TESTING=OFF'],
    repository='https://github.com/CESNET/libyang-cpp.git',
    # branch='v4',
    commit='249da7280864fbda5fccb340b455b7000ebfe67d',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/libyang-cpp',
        'cp LICENSE /usr/local/share/licenses/libyang-cpp/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    cmake_opts=['-DBUILD_SHARED_LIBS=ON','-DENABLE_PUSH=OFF','-DENABLE_COMPRESSION=OFF','-DENABLE_TESTING=OFF'],
    repository='https://github.com/jupp0r/prometheus-cpp.git',
    # branch='v1.3.0',
    commit='e5fada43131d251e9c4786b04263ce98b6767ba5',
    recursive=True,
    preconfigure=[
        # Re-sync submodules to the pinned commit's gitlinks: hpccm's recursive
        # clone leaves them at live HEAD, not what the pinned commit records.
        'git submodule update --init --recursive',
        'mkdir -p /usr/local/share/licenses/prometheus-cpp',
        'cp LICENSE /usr/local/share/licenses/prometheus-cpp/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    cmake_opts=['-DBUILD_SHARED_LIBS=ON'],
    repository='https://gitlab.com/libeigen/eigen.git',
    # branch='3.4.1',
    commit='d71c30c47858effcbd39967097a2d99ee48db464',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/eigen',
        'cp COPYING.MPL2 /usr/local/share/licenses/eigen/ 2>/dev/null',
    ],
    )

Externals += generic_cmake(
    repository='https://github.com/ClickHouse/clickhouse-cpp.git',
    # branch='v2.6.0',
    commit='69195246a3b39542c397ef27df9f46ec4a4bf206',
    cmake_opts=['-DCMAKE_POSITION_INDEPENDENT_CODE=ON'],
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/clickhouse-cpp',
        'cp LICENSE /usr/local/share/licenses/clickhouse-cpp/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    repository='https://github.com/CLIUtils/CLI11.git',
    # branch='v2.5.0',
    commit='4160d259d961cd393fd8d67590a8c7d210207348',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/CLI11',
        'cp LICENSE /usr/local/share/licenses/CLI11/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    repository='https://github.com/gsl-lite/gsl-lite.git',
    # branch='v1.0.1',
    commit='56dab5ce071c4ca17d3e0dbbda9a94bd5a1cbca1',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/gsl-lite',
        'cp LICENSE /usr/local/share/licenses/gsl-lite/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    repository='https://github.com/jbeder/yaml-cpp.git',
    # branch='0.8.0',
    commit='f7320141120f720aecc4c32be25586e7da9eb978',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/yaml-cpp',
        'cp LICENSE /usr/local/share/licenses/yaml-cpp/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    repository='https://github.com/quicknir/wise_enum.git',
    # branch='3.1.0',
    commit='34ac79f7ea2658a148359ce82508cc9301e31dd3',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/wise_enum',
        'cp LICENSE /usr/local/share/licenses/wise_enum/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    cmake_opts=['-DBUILD_TESTING=OFF'],
    repository='https://github.com/TartanLlama/expected.git',
    #branch='v1.3.1',
    commit='1770e3559f2f6ea4a5fb4f577ad22aeb30fbd8e4',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/tl-expected',
        'cp COPYING /usr/local/share/licenses/tl-expected/ 2>/dev/null',
    ],
    )

Externals += generic_cmake(
    cmake_opts=['-DENABLE_TEST=OFF'],
    repository='https://github.com/joboccara/NamedType.git',
    #branch='master'  # no release since v1.1.0; using master tip for constexpr operators, Incrementable/Decrementable skills, and Comparable symmetry fix
    commit='76668abe09807f92a695ee5e868f9719e888e65f',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/NamedType',
        'cp LICENSE /usr/local/share/licenses/NamedType/ 2>/dev/null',
    ],
    )

Externals += generic_cmake(
    cmake_opts=[
        '-DRANGE_V3_TESTS=OFF',
        '-DRANGE_V3_EXAMPLES=OFF',
        '-DRANGE_V3_DOCS=OFF',
    ],
    repository='https://github.com/ericniebler/range-v3.git',
    # branch='0.12.0',
    commit='a81477931a8aa2ad025c6bda0609f38e09e4d7ec',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/range-v3',
        'cp LICENSE.txt /usr/local/share/licenses/range-v3/ 2>/dev/null',
    ],
    )

Externals += generic_cmake(
    repository='https://github.com/bombela/backward-cpp.git',
    # branch='v1.6',
    commit='3bb9240cb15459768adb3e7d963a20e1523a6294',
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/backward-cpp',
        'cp LICENSE.txt /usr/local/share/licenses/backward-cpp/ 2>/dev/null',
        ],
    )

Externals += generic_cmake(
    repository='https://github.com/yhirose/cpp-httplib.git',
    # branch='v0.27.0',
    commit='eacc1ca98e5fef25184c7d417e8417225e05e65d',
    cmake_opts=['-DHTTPLIB_REQUIRE_OPENSSL=ON', '-DHTTPLIB_USE_ZSTD_IF_AVAILABLE=OFF'],
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/cpp-httplib',
        'cp LICENSE /usr/local/share/licenses/cpp-httplib/ 2>/dev/null',
        ],
    )

Externals += copy(src='/patches', dest='/tmp/patches')

Externals += generic_cmake(
    cmake_opts=['-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DBUILD_SHARED_LIBS=ON'],
    repository='https://github.com/fmtlib/fmt.git',
    # branch='10.2.1',
    commit='e69e5f977d458f2650bb346dadf2ad30c5320281',
    recursive=True,
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/fmt',
        'cp LICENSE /usr/local/share/licenses/fmt/ 2>/dev/null',
        ],
    )


Externals += generic_cmake(
    cmake_opts=[
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
    ],
    repository='https://github.com/MengRao/fmtlog.git',
    # branch='',
    commit='acd521b1a64480354136a745c511358da1ec7dc5',
    recursive=False,
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/fmtlog',
        'cp LICENSE /usr/local/share/licenses/fmtlog/ 2>/dev/null',
        'git apply -v /tmp/patches/fmtlog.patch',
        'cp fmtlog.h /usr/local/include/',
        'cp fmtlog-inl.h /usr/local/include/',
        ],
    )

Externals += generic_cmake(
    cmake_opts=['-DBUILD_SHARED_LIBS=ON'],
    repository='https://github.com/microsoft/mimalloc',
    # branch='v2.0.6',
    commit='f2712f4a8f038a7fb4df2790f4c3b7e3ed9e219b',
    recursive=True,
    preconfigure=[
        'mkdir -p /usr/local/share/licenses/mimalloc',
        'cp LICENSE /usr/local/share/licenses/mimalloc/ 2>/dev/null',
        'git apply -v /tmp/patches/mimalloc.patch',
        ],
    )

# https://developer.nvidia.com/cufftdx-downloads
Externals += shell(commands=[
    f'wget -q https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/{MATHDX_CUDA_VERSION}/{MATHDX_ARCHIVE}',
    f'tar xzf {MATHDX_ARCHIVE} -C /usr/local --strip-components=1',
    f'rm {MATHDX_ARCHIVE}',
    ])


#
# Base Stage
#
Base += baseimage(image='bootstrap', _arch=cpu_target, _distro=CONTAINER_UBUNTU_DISTRO, _as='base')

Base += packages(ospackages=[
    'apt-utils',
    'ccache',
    'python3-pip',
    'libdw-dev',
    'libhdf5-dev',
    'libsctp-dev',
    'libcunit1-dev',
    'libzmq3-dev',
    'nlohmann-json3-dev',

    'libabsl-dev',
    'libgrpc++-dev',
    'libgrpc-dev',
    'libprotobuf-dev',
    'protobuf-compiler',
    'protobuf-compiler-grpc',

    # Addons to be able to run
    "sudo",
    "check",
    "ethtool",
    "numactl",
    "kmod",  # lsmod - cuBB_system_checks.py

    # 'Good to have' Addons
    "vim",
    "nano",
    ])

# Pip defaults to 5 retries, but this only works for connection errors, not errors during download
# Increase the timeout from default of 15s to 60s for intermittently bad connections
Base += shell(commands=[
    'printf "[global]\\nretries = 5\\ntimeout = 60\\n" > /etc/pip.conf'
    ])

# Install uv - fast Python package manager
Base += shell(commands=[
    'curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh'
    ])

Base += externals.runtime(_from='externals')

Base += shell(commands=[
    'git clone https://gerrit.o-ran-sc.org/r/scp/oam/modeling /opt/modeling',
    'cd /opt/modeling',
    'git checkout bcbda9bc83169fd6dedec18ce25d08e14f319314',
    ])

# Doca installation
Base += shell(commands=[
    'mkdir -p /tmp/doca-host',
    'cd /tmp/doca-host',
    f'wget -q https://www.mellanox.com/downloads/DOCA/DOCA_v{DOCA_VERSION}/host/doca-host_{DOCA_HOST_ID}_{TARGETARCH}.deb',
    'dpkg -i /tmp/doca-host/doca-host_*.deb',
    'apt-get update -y',
    'rm -rf /tmp/doca-host',
    ])

if not DPDK_TAG:
    Base += packages(ospackages=[
        'dpdk-community-dev',
        ])

""
# Uncomment ^^^ Build from source
repo_dir = f'/usr/share/doca-host-{DOCA_HOST_ID}/repo'

pool_rm = [
    f'rm {repo_dir}/pool/doca-sdk-common_*',
    f'rm {repo_dir}/pool/libdoca-sdk-common-dev_*',
    f'rm {repo_dir}/pool/doca-sdk-verbs_*',
    f'rm {repo_dir}/pool/libdoca-sdk-verbs-dev_*',
    f'rm {repo_dir}/pool/doca-sdk-gpunetio_*',
    f'rm {repo_dir}/pool/libdoca-sdk-gpunetio-dev*',
]
if DPDK_TAG:
    pool_rm += [
        f'rm {repo_dir}/pool/doca-sdk-dpdk-bridge_*',
        f'rm {repo_dir}/pool/libdoca-sdk-dpdk-bridge-dev_*',
        f'rm -f {repo_dir}/pool/*dpdk-community*',
    ]
Base += shell(commands=pool_rm)

# Add aerial .deb files
Base += raw(docker=f"ADD doca-for-aerial-{TARGETARCH}.tgz {repo_dir}/pool/")

# Uncomment vvv Build from source
""
deb_list = [
    'libdoca-sdk-argp-dev',
    'doca-sdk-dpdk-bridge',
    'libdoca-sdk-dpdk-bridge-dev',
    'doca-sdk-eth',
    'libdoca-sdk-eth-dev',
    'doca-sdk-gpunetio',
    'libdoca-sdk-gpunetio-dev',
    'doca-samples',
    ]

deb_packs = ' '.join(deb_list)
Base += shell(commands=[
    # Uncomment vvv when building from source
    'rm -rf /var/lib/apt/lists/*',
    'apt-get clean',
    f'(cd {repo_dir} && dpkg-scanpackages --arch {TARGETARCH} pool /dev/null > Packages && apt-ftparchive release . > Release && rm InRelease)',
    "sed -i 's|\[signed-by=.*\]|\[trusted=yes\]|g' /etc/apt/sources.list.d/doca.list",
    'apt-get update -y',
    # Uncomment ^^^ when building from source
    f'DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {deb_packs}',
    'rm -rf /var/lib/apt/lists/*',
    ])

# Configure the fixuid binary built in the bootstrap stage.
Base += shell(commands=[
    'addgroup --gid 1000 aerial',
    'adduser --uid 1000 --ingroup aerial --home /home/aerial --shell /bin/bash --disabled-password --gecos "" aerial',
    'USER=aerial',
    'GROUP=aerial',
    'test -x /usr/local/bin/fixuid',
    'chown root:root /usr/local/bin/fixuid',
    'chmod 4755 /usr/local/bin/fixuid',
    'mkdir -p /etc/fixuid',
    'printf "user: $USER\\ngroup: $GROUP\\npaths:\\n  - /home/$USER\\n" > /etc/fixuid/config.yml',
    '/bin/echo "aerial ALL = (root) NOPASSWD: ALL" >> /etc/sudoers',
    'runuser -l aerial -c "mkdir -p /home/aerial/.local/bin"',
   ])

# Setup CUDA device env
Base += environment(variables={
    'AERIAL_PROCESS': '1',
    'cuBB_SDK': '/opt/nvidia/cuBB',
    'CUDA_DEVICE_MAX_CONNECTIONS': '8',
    'CUDA_MODULE_LOADING': 'EAGER',
    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
    'LD_LIBRARY_PATH': '',
    'NVIDIA_DISABLE_REQUIRE': '1'
    })

Base += user(user='aerial')
Base += workdir(directory='/')

Base += raw(docker='CMD ["bash", "-c", "echo Aerial SDK container ready, going to sleep; sleep infinity"]')
