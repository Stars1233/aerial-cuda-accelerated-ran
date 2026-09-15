#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import List
from setuptools import setup, find_packages

def parse_requirements(path: str) -> List[str]:
    """Parse a requirements.txt file into a list of dependency strings."""
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#') and not line.startswith('-')]

setup(
    name='aerial_postproc',
    version='1.0.0',
    description='Aerial Post Processing - Performance analysis and log processing tools',
    packages=find_packages(),
    python_requires='>=3.8,<3.11',
    install_requires=parse_requirements(
        Path(__file__).parent / 'requirements.txt'
    ),
    entry_points={
        'console_scripts': [
            # Add CLI entry points here if needed
        ],
    },
)
