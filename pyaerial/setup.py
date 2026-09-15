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

"""Setup file for PyAerial package."""
import sysconfig
from pathlib import Path

import setuptools

EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
PYCUPHY_DIR = Path(__file__).parent / "src" / "aerial" / "pycuphy"

if not EXT_SUFFIX:
    raise RuntimeError("EXT_SUFFIX is unavailable; cannot package _pycuphy extension")
if not PYCUPHY_DIR.is_dir():
    raise RuntimeError(f"pycuphy package directory not found: {PYCUPHY_DIR}")


def _pycuphy_native_artifacts() -> list[str]:
    core = PYCUPHY_DIR / f"_pycuphy{EXT_SUFFIX}"
    if not core.is_file():
        raise FileNotFoundError(f"Missing native extension: {core}")
    return [core.name]


class BinaryDistribution(setuptools.Distribution):
    """Treat the pre-built _pycuphy extension as a platform library."""

    def has_ext_modules(self) -> bool:
        return True


setuptools.setup(
    distclass=BinaryDistribution,
    package_data={
        "aerial": ["version_aerial.py"],
        "aerial.pycuphy": _pycuphy_native_artifacts(),
    },
    zip_safe=False,
)
