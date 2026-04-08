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

import os
import sys
import setuptools

# Check if BUILD_ID environment variable exists - from Jenkins
BUILD_ID = os.environ.get('BUILD_ID')
if BUILD_ID is None:
    raise RuntimeError("Environment variable BUILD_ID must be set")

MCORE_BUILD_TYPE = os.environ.get('MCORE_BUILD_TYPE')
if MCORE_BUILD_TYPE is None:
    MCORE_BUILD_TYPE=''

# Create release version according to https://www.python.org/dev/peps/pep-0440/
# The full version, including alpha/beta/rc tags
version=f"0.20261.{BUILD_ID}.{MCORE_BUILD_TYPE}{BUILD_ID}"

setuptools.setup(
    name="aerial_mcore",
    version=version,
    author="NVIDIA",
    description="Aerial 5GModel Python bindings",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={"": ["*.ctf","VERSION_aerial.py"]},
    python_requires=">=3.8",
    zip_safe=False
)
