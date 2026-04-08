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

from setuptools import setup, find_packages

setup(
    name='aerial_postproc',
    version='1.0.0',
    description='Aerial Post Processing - Performance analysis and log processing tools',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.5.2',
        'numpy>=1.24.0',
        'bokeh>=2.4.3',
        'holoviews>=1.15.2',
        'h5py>=3.7.0',
        'pyarrow>=12.0.1',
        'tables>=3.7.0',
        'pyparsing>=3.0.9',
        'pyyaml>=6.0',
        'colorama>=0.4.6',
        'cxxfilt',
        'ntplib',
    ],
    entry_points={
        'console_scripts': [
            # Add CLI entry points here if needed
        ],
    },
)
