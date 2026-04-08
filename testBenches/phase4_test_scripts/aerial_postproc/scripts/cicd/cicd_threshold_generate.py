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
from aerial_postproc.cicd_utils import is_higher_is_better, is_percentage_metric, compute_worst_case, build_effective_slot_groups


def generate_thresholds(perf_csvs, requirements_csv, gating_output, warning_output):
    """
    Generate gating/warning perf_requirements from one or more baseline perf.csv files.

    For each metric group defined in the requirements file:
    1. Compute worst-case value per perf.csv across applicable slots
    2. Take the mean of those worst-case values as the baseline
    3. Compute threshold = mean_worst + mean_headroom (clamped to absolute)
    4. Compute headroom = required_value - threshold
    5. Write output files with updated headroom and mean_headroom pass-through
    """
    if isinstance(perf_csvs, str):
        perf_csvs = [perf_csvs]

    if not os.path.exists(requirements_csv):
        print(f"ERROR: Requirements CSV not found: {requirements_csv}")
        return 1

    perf_dfs = []
    for perf_csv in perf_csvs:
        if not os.path.exists(perf_csv):
            print(f"WARNING: Performance CSV not found, skipping: {perf_csv}")
            continue
        perf_dfs.append(pd.read_csv(perf_csv, dtype=np.float64, na_values=CICD_NONE_FORMAT))

    if len(perf_dfs) == 0:
        print("ERROR: No valid performance CSV files found")
        return 1

    req_df = pd.read_csv(requirements_csv)

    required_cols = ['metric_name', 'required_value', 'headroom']
    for col in required_cols:
        if col not in req_df.columns:
            print(f"ERROR: Requirements file missing required column: {col}")
            return 1

    if 'slots' not in req_df.columns:
        req_df['slots'] = ''

    for col in ['gating_mean_headroom', 'warning_mean_headroom']:
        if col not in req_df.columns:
            print(f"WARNING: No {col} column found, using 0 for all metrics")
            req_df[col] = 0.0

    all_slots = sorted(perf_dfs[0]['slot'].unique())

    print(f"Loaded {len(perf_dfs)} perf.csv file(s) with {len(perf_dfs[0])} slots each")
    print(f"Loaded {len(req_df)} requirement entries from {requirements_csv}")

    effective_slots = build_effective_slot_groups(req_df, all_slots)

    gating_headrooms = []
    warning_headrooms = []
    gating_mean_hr_values = []
    warning_mean_hr_values = []
    mean_worsts = []
    min_worsts = []
    max_worsts = []

    for idx, row in req_df.iterrows():
        metric = row['metric_name']
        required_value = float(row['required_value'])
        gating_mean_hr = float(row['gating_mean_headroom'])
        warning_mean_hr = float(row['warning_mean_headroom'])
        target_slots = effective_slots[idx]

        worst_cases = []
        for perf_df in perf_dfs:
            if metric not in perf_df.columns:
                continue
            slot_data = perf_df[perf_df['slot'].isin(target_slots)][metric]
            worst = compute_worst_case(slot_data, metric)
            if not np.isnan(worst):
                worst_cases.append(worst)

        if len(worst_cases) == 0:
            print(f"WARNING: No valid data for {metric}, keeping original headroom")
            gating_headrooms.append(row['headroom'])
            warning_headrooms.append(row['headroom'])
            mean_worsts.append(np.nan)
            min_worsts.append(np.nan)
            max_worsts.append(np.nan)
            gating_mean_hr_values.append(gating_mean_hr)
            warning_mean_hr_values.append(warning_mean_hr)
            continue

        mean_worst = np.mean(worst_cases)
        precision = 4 if is_percentage_metric(metric) else 1
        mean_worsts.append(round(mean_worst, precision))
        min_worsts.append(min(worst_cases))
        max_worsts.append(max(worst_cases))
        gating_mean_hr_values.append(gating_mean_hr)
        warning_mean_hr_values.append(warning_mean_hr)

        abs_headroom = float(row['headroom'])

        if is_higher_is_better(metric):
            abs_threshold = required_value + abs_headroom
            gating_threshold = max(mean_worst - gating_mean_hr, abs_threshold)
            warning_threshold = max(mean_worst - warning_mean_hr, abs_threshold)
            g_headroom = round(gating_threshold - required_value, precision)
            w_headroom = round(warning_threshold - required_value, precision)
        else:
            abs_threshold = required_value - abs_headroom
            gating_threshold = min(mean_worst + gating_mean_hr, abs_threshold)
            warning_threshold = min(mean_worst + warning_mean_hr, abs_threshold)
            g_headroom = round(required_value - gating_threshold, precision)
            w_headroom = round(required_value - warning_threshold, precision)

        gating_headrooms.append(g_headroom)
        warning_headrooms.append(w_headroom)

    output_cols = ['metric_name', 'required_value', 'headroom', 'slots', 'mean_worst', 'mean_headroom']

    gating_df = pd.DataFrame({
        'metric_name': req_df['metric_name'],
        'required_value': req_df['required_value'],
        'headroom': gating_headrooms,
        'slots': req_df['slots'].fillna(''),
        'mean_worst': mean_worsts,
        'mean_headroom': gating_mean_hr_values,
    })[output_cols]

    warning_df = pd.DataFrame({
        'metric_name': req_df['metric_name'],
        'required_value': req_df['required_value'],
        'headroom': warning_headrooms,
        'slots': req_df['slots'].fillna(''),
        'mean_worst': mean_worsts,
        'mean_headroom': warning_mean_hr_values,
    })[output_cols]

    pct_rows = [is_percentage_metric(m) for m in req_df['metric_name']]
    for df in [gating_df, warning_df]:
        for col in ['required_value', 'headroom', 'mean_worst', 'mean_headroom']:
            df[col] = [f"{v:.4f}" if pct and not (isinstance(v, float) and np.isnan(v)) else v
                       for v, pct in zip(df[col], pct_rows)]

    gating_df.to_csv(gating_output, index=False)
    warning_df.to_csv(warning_output, index=False)

    print(f"\nGating thresholds written to:  {gating_output}")
    print(f"Warning thresholds written to: {warning_output}")

    print("\n" + "=" * 140)
    print("Generated Threshold Summary")
    print("=" * 140)

    max_name_len = max(len(row['metric_name']) for _, row in req_df.iterrows())
    for idx, row in req_df.iterrows():
        metric = row['metric_name']
        padded = metric.ljust(max_name_len)
        pct = is_percentage_metric(metric)
        unit = 'pp' if pct else 'us'
        g_hr = gating_headrooms[idx]
        w_hr = warning_headrooms[idx]
        req_val = row['required_value']
        abs_hr = row['headroom']
        mw = mean_worsts[idx]
        mn = min_worsts[idx]
        mx = max_worsts[idx]
        g_mhr = gating_mean_hr_values[idx]
        w_mhr = warning_mean_hr_values[idx]

        if is_higher_is_better(metric):
            abs_thresh = req_val + abs_hr
            g_thresh = req_val + g_hr
            w_thresh = req_val + w_hr
        else:
            abs_thresh = req_val - abs_hr
            g_thresh = req_val - g_hr
            w_thresh = req_val - w_hr

        if pct:
            na = "       N/A"
            mw_s = f"{mw:10.4f}" if not np.isnan(mw) else na
            mn_s = f"{mn:10.4f}" if not np.isnan(mn) else na
            mx_s = f"{mx:10.4f}" if not np.isnan(mx) else na
            print(f"  {padded}  min={mn_s}  mean={mw_s}  max={mx_s}  g_mhr={g_mhr:8.4f}{unit}  w_mhr={w_mhr:8.4f}{unit}  abs={abs_thresh:10.4f}  gating={g_thresh:10.4f}  warning={w_thresh:10.4f}")
        else:
            na = "      N/A"
            mw_s = f"{mw:9.1f}" if not np.isnan(mw) else na
            mn_s = f"{mn:9.1f}" if not np.isnan(mn) else na
            mx_s = f"{mx:9.1f}" if not np.isnan(mx) else na
            print(f"  {padded}  min={mn_s}  mean={mw_s}  max={mx_s}  g_mhr={g_mhr:7.1f}{unit}  w_mhr={w_mhr:7.1f}{unit}  abs={abs_thresh:9.1f}  gating={g_thresh:9.1f}  warning={w_thresh:9.1f}")

    print("=" * 140)

    return 0


def main(args):
    return generate_thresholds(
        args.performance_csvs,
        args.requirements_csv,
        args.gating_output_csv,
        args.warning_output_csv,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate gating/warning perf_requirements files from baseline perf.csv file(s)"
    )
    parser.add_argument(
        "requirements_csv", type=str,
        help="Perf_requirements CSV (with gating_mean_headroom and warning_mean_headroom columns)"
    )
    parser.add_argument(
        "gating_output_csv", type=str,
        help="Output path for gating perf_requirements file"
    )
    parser.add_argument(
        "warning_output_csv", type=str,
        help="Output path for warning perf_requirements file"
    )
    parser.add_argument(
        "performance_csvs", nargs="+", type=str,
        help="One or more baseline perf.csv files (mean of worst-cases used as baseline)"
    )
    args = parser.parse_args()

    sys.exit(main(args))
