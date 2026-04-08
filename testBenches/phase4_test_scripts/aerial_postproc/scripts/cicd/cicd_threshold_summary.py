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

import argparse
import pandas as pd
import numpy as np
import sys
import os

from aerial_postproc.logparse import CICD_NONE_FORMAT
from aerial_postproc.cicd_utils import (
    is_higher_is_better,
    is_percentage_metric,
    compute_worst_case,
    build_effective_slot_groups,
    format_slot_list,
    load_requirements,
)


def build_metric_group_order(req_dfs):
    """
    Build an ordered list of (metric_name, slots_str) tuples from the union
    of all requirements files. Order follows the first file, with additional
    groups from subsequent files appended.
    """
    seen = set()
    order = []
    for req_df in req_dfs:
        for _, row in req_df.iterrows():
            metric = row['metric_name']
            slots_str = str(row['slots']).strip()
            if slots_str == 'nan':
                slots_str = ''
            key = (metric, slots_str)
            if key not in seen:
                seen.add(key)
                order.append(key)
    return order


def compute_threshold(req_df, metric, slots_str):
    """
    Compute threshold_value from required_value and headroom for a given
    metric + slots combination in a requirements DataFrame.
    Returns (threshold_value, required_value) or (None, None) if not found.
    """
    for _, row in req_df.iterrows():
        row_slots = str(row['slots']).strip()
        if row_slots == 'nan':
            row_slots = ''
        if row['metric_name'] == metric and row_slots == slots_str:
            required_value = float(row['required_value'])
            headroom = float(row['headroom'])
            if is_higher_is_better(metric):
                return required_value + headroom, required_value
            else:
                return required_value - headroom, required_value
    return None, None


def main(args):
    perf_csv = args.performance_csv
    req_files = args.requirements_csvs

    if not os.path.exists(perf_csv):
        print(f"ERROR: Performance CSV not found: {perf_csv}")
        return 1

    if args.labels:
        if len(args.labels) != len(req_files):
            print(f"ERROR: Number of labels ({len(args.labels)}) must match number of requirements files ({len(req_files)})")
            return 1
        labels = args.labels
    else:
        labels = [os.path.basename(f).replace('.csv', '') for f in req_files]

    perf_df = pd.read_csv(perf_csv, dtype=np.float64, na_values=CICD_NONE_FORMAT)
    all_slots = sorted(perf_df['slot'].unique())

    req_dfs = []
    for req_file in req_files:
        try:
            req_dfs.append(load_requirements(req_file))
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}")
            return 1

    metric_groups = build_metric_group_order(req_dfs)

    ref_groups = set(build_metric_group_order([req_dfs[0]]))
    for i in range(1, len(req_dfs)):
        file_groups = set(build_metric_group_order([req_dfs[i]]))
        if file_groups != ref_groups:
            extra = file_groups - ref_groups
            missing = ref_groups - file_groups
            print(f"ERROR: Requirements file '{req_files[i]}' has different metric groups than '{req_files[0]}'")
            if extra:
                print(f"  Extra:   {sorted(extra)}")
            if missing:
                print(f"  Missing: {sorted(missing)}")
            return 1

    effective_slots = build_effective_slot_groups(req_dfs[0], all_slots)
    effective_slots_lookup = {}
    for row_idx, row in req_dfs[0].iterrows():
        slots_key = str(row['slots']).strip()
        if slots_key == 'nan':
            slots_key = ''
        effective_slots_lookup[(row['metric_name'], slots_key)] = effective_slots[row_idx]

    rows = []
    for metric, slots_str in metric_groups:
        target_slots = effective_slots_lookup.get((metric, slots_str), all_slots)
        slots_display = format_slot_list(target_slots, all_slots)

        worst = np.nan
        if metric in perf_df.columns:
            slot_data = perf_df[perf_df['slot'].isin(target_slots)][metric]
            worst = compute_worst_case(slot_data, metric)

        row_data = {
            'metric_name': metric,
            'slots': slots_display,
            'perf_worst': worst,
        }

        for i, req_df in enumerate(req_dfs):
            thresh, _ = compute_threshold(req_df, metric, slots_str)
            row_data[labels[i]] = thresh

        rows.append(row_data)

    max_metric_len = max(len(r['metric_name']) for r in rows) if rows else 20
    max_slots_len = max(len(r['slots']) for r in rows) if rows else 5
    col_width = 12

    header_cols = ['perf_worst'] + labels
    header_metric = 'metric_name'.ljust(max_metric_len)
    header_slots = 'slots'.ljust(max_slots_len)
    header_line = f"  {header_metric}  {header_slots}"
    for col in header_cols:
        if col == 'perf_worst':
            header_line += f"  {'perf.csv':>{col_width}}"
        else:
            header_line += f"  {col:>{col_width}}"

    sep = "=" * len(header_line)

    print(f"\nFiles:")
    print(f"  perf.csv:  {perf_csv}")
    for i, label in enumerate(labels):
        print(f"  {label}:  {req_files[i]}")
    print(f"\n{sep}")
    print("Threshold Summary")
    print(sep)
    print(header_line)
    print(f"  {'':>{max_metric_len}}  {'':>{max_slots_len}}  {'(worst)':>{col_width}}" +
          ''.join(f"  {'(thresh)':>{col_width}}" for _ in labels))
    print(sep)

    for row_data in rows:
        metric = row_data['metric_name']
        pct = is_percentage_metric(metric)

        padded_metric = metric.ljust(max_metric_len)
        padded_slots = row_data['slots'].ljust(max_slots_len)
        line = f"  {padded_metric}  {padded_slots}"

        worst = row_data['perf_worst']
        hib = is_higher_is_better(metric)

        for col in header_cols:
            if col == 'perf_worst':
                val = worst
                failed = False
            else:
                val = row_data.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)) and not (isinstance(worst, float) and np.isnan(worst)):
                    failed = worst < val if hib else worst > val
                else:
                    failed = False

            mark = "*" if failed else " "
            val_width = col_width - 1

            if val is None or (isinstance(val, float) and np.isnan(val)):
                line += f"  {'---':>{val_width}}{mark}"
            elif pct:
                line += f"  {val:{val_width}.3f}{mark}"
            else:
                line += f"  {val:{val_width}.1f}{mark}"

        print(line)

    print(sep)

    if args.output_csv:
        csv_rows = []
        for row_data in rows:
            csv_row = {
                'metric_name': row_data['metric_name'],
                'slots': row_data['slots'],
                'perf_worst': row_data['perf_worst'],
            }
            for label in labels:
                csv_row[f'{label}_threshold'] = row_data.get(label)
            csv_rows.append(csv_row)
        out_df = pd.DataFrame(csv_rows)
        out_df.to_csv(args.output_csv, index=False)
        print(f"\nSummary written to: {args.output_csv}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display threshold values from multiple perf_requirements files alongside worst-case performance"
    )
    parser.add_argument(
        "performance_csv", type=str,
        help="Input perf.csv file"
    )
    parser.add_argument(
        "requirements_csvs", nargs="+", type=str,
        help="One or more perf_requirements files to compare"
    )
    parser.add_argument(
        "-l", "--labels", nargs="+", type=str,
        help="Labels for each requirements file (must match count)"
    )
    parser.add_argument(
        "-o", "--output_csv", type=str,
        help="Optional: write summary to CSV file"
    )
    args = parser.parse_args()

    sys.exit(main(args))
