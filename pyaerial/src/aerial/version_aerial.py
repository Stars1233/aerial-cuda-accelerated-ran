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
BUILD_ID = os.environ.get("BUILD_ID") or "1"

# The short X.Y version
VERSION = "2026.2"  # pylint: disable=invalid-name

# Create release version according to https://www.python.org/dev/peps/pep-0440/
# Local version segment (+build.N) tags each CI build.
RELEASE = f"{VERSION}+build.{BUILD_ID}"
