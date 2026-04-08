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

"""PyAerial release Docker image hpccm recipe using Aerial release image as the base.

Usage:
$ hpccm
    --recipe pyaerial_recipe.py
    --format docker
    --userarg AERIAL_BASE_IMAGE
"""
AERIAL_BASE_IMAGE = USERARG.get("AERIAL_BASE_IMAGE")
if AERIAL_BASE_IMAGE is None:
    raise RuntimeError("User argument AERIAL_BASE_IMAGE must be set")


if cpu_target == 'x86_64':
    TARGETARCH='amd64'
    PyAerial = stages[0]
elif cpu_target == 'aarch64':
    TARGETARCH='arm64'
    TensorFlow = stages[0]
    PyAerial = stages[1]
else:
    raise RuntimeError("Unsupported platform")

PyAerial += baseimage(image=AERIAL_BASE_IMAGE, _distro='ubuntu22', _arch=cpu_target)

PyAerial += user(user='root')

PyAerial += packages(ospackages=[
    'cudnn9-cuda-13',
    ])

PyAerial += environment(variables={
    "VIRTUAL_ENV": "/opt/venv",
    "PATH": "/opt/venv/bin:$PATH",
    })

PyAerial += shell(commands=[
    'mkdir -p $VIRTUAL_ENV',
    'chown aerial:aerial $VIRTUAL_ENV',
    ])

PyAerial += user(user='aerial')

PyAerial += shell(commands=[
    'python3 -m venv $VIRTUAL_ENV --system-site-packages',
    ])

PyAerial += copy(src='.', dest='$cuBB_SDK', _chown="aerial")

# install PyAerial.
PyAerial += shell(commands=[
    '$cuBB_SDK/pyaerial/scripts/install_dev_pkg.sh'
    ])

PyAerial += shell(commands=[
    'pip install pip --upgrade',
    f'pip install -r $cuBB_SDK/pyaerial/container/requirements.txt -r $cuBB_SDK/pyaerial/container/requirements-{TARGETARCH}.txt',
])

PyAerial += shell(commands=[
    'rm -rf /home/aerial/.cache',
    ])
PyAerial += workdir(directory='/home/aerial')
PyAerial += raw(docker='CMD ["/bin/bash"]')
