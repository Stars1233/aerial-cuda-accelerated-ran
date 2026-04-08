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

"""Provides pyAerial version and release numbering."""
import os

# Check if BUILD_ID environment variable exists - from Jenkins
BUILD_ID = os.environ.get("BUILD_ID")
if BUILD_ID is None:
    raise RuntimeError("Environment variable BUILD_ID must be set")

BUILD_TYPE = os.environ.get("BUILD_TYPE")
if BUILD_TYPE is None:
    BUILD_TYPE = "dev"

if BUILD_TYPE == "rel":
    BUILD_TYPE = ""  # don't use 'rel' in the build string

# The short X.Y version
VERSION = "2026.1"  # pylint: disable=invalid-name

# Create release version according to https://www.python.org/dev/peps/pep-0440/
# The full version, including alpha/beta/rc tags
RELEASE = f"{VERSION}.{BUILD_TYPE}{BUILD_ID}"
