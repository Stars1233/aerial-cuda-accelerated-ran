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

import os
import pandas as pd
import numpy as np


def is_higher_is_better(metric_name):
    """Return True for metrics where higher values indicate better performance."""
    return (metric_name.endswith('_min') or
            metric_name.endswith('_ontime_percentage') or
            metric_name.endswith('_headroom'))


def is_percentage_metric(metric_name):
    """Return True for percentage metrics (affects display unit and precision)."""
    return metric_name.endswith('_percentage')


def compute_worst_case(values, metric_name):
    """Return the worst-case value from a series: max for lower-is-better, min for higher-is-better."""
    valid = values.dropna()
    if len(valid) == 0:
        return np.nan
    if is_higher_is_better(metric_name):
        return valid.min()
    else:
        return valid.max()


def build_requirements_map(req_df, all_slots):
    """
    Build requirements map with slot-specific overrides.

    Processes rows in order. Later rows with specific slots override
    earlier "all" rows for those slots.

    Returns: {metric: {slot: (required_value, headroom)}}
    """
    requirements_map = {}

    for idx, row in req_df.iterrows():
        metric = row['metric_name']
        required_value = float(row['required_value'])
        headroom = float(row['headroom'])
        slots_str = str(row['slots']).strip()

        if metric not in requirements_map:
            requirements_map[metric] = {}

        if slots_str == '' or slots_str == 'nan':
            target_slots = all_slots
        else:
            target_slots = [int(s.strip()) for s in slots_str.split(',')]

        for slot in target_slots:
            requirements_map[metric][slot] = (required_value, headroom)

    return requirements_map


def build_effective_slot_groups(req_df, all_slots):
    """
    Compute effective slot lists per row, accounting for overrides.

    Uses build_requirements_map to resolve overrides, then for each row
    returns the slots whose final (required_value, headroom) matches
    that row's values.

    Returns: list parallel to req_df rows, each entry a sorted list of slot ints.
    """
    req_map = build_requirements_map(req_df, all_slots)
    effective = []
    for idx, row in req_df.iterrows():
        metric = row['metric_name']
        rv = float(row['required_value'])
        hr = float(row['headroom'])
        slots = [s for s, (r, h) in req_map[metric].items() if r == rv and h == hr]
        effective.append(sorted(slots))
    return effective


def format_slot_list(slots, all_slots, max_display=4):
    """Format a slot list for display. Shows 'all' only if it truly covers every slot."""
    int_slots = [int(s) for s in slots]
    if set(int_slots) == set(int(s) for s in all_slots):
        return 'all'
    if len(int_slots) <= max_display:
        return ','.join(str(s) for s in int_slots)
    return ','.join(str(s) for s in int_slots[:max_display-1]) + ',..(' + str(len(int_slots)) + ')'


def load_requirements(requirements_file):
    """
    Load a perf_requirements CSV file.

    Validates required columns (metric_name, required_value, headroom),
    adds slots column if not present.

    Returns:
        pd.DataFrame with the requirements data.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing or the file cannot be parsed.
    """
    if not os.path.exists(requirements_file):
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    try:
        req_df = pd.read_csv(requirements_file)
    except Exception as e:
        raise ValueError(f"Failed to read requirements file: {e}")

    required_cols = ['metric_name', 'required_value', 'headroom']
    for col in required_cols:
        if col not in req_df.columns:
            raise ValueError(f"Requirements file missing required column: {col}")

    if 'slots' not in req_df.columns:
        req_df['slots'] = ''

    return req_df
