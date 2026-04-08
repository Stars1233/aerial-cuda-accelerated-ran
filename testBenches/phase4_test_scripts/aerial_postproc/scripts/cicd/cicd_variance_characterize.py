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
import math

from aerial_postproc.logparse import CICD_NONE_FORMAT
from aerial_postproc.cicd_utils import is_percentage_metric, compute_worst_case, build_effective_slot_groups, format_slot_list

K_GATING = 12.0
K_WARNING = 9.0


def main(args):
    input_file = args.input_requirements
    perf_csvs = args.performance_csvs

    if len(perf_csvs) < 2:
        print("WARNING: At least 2 perf.csv files recommended for meaningful spread calculation")

    if not os.path.exists(input_file):
        print(f"ERROR: Input requirements file not found: {input_file}")
        return 1

    req_df = pd.read_csv(input_file)

    required_cols = ['metric_name', 'required_value', 'headroom']
    for col in required_cols:
        if col not in req_df.columns:
            print(f"ERROR: Requirements file missing required column: {col}")
            return 1

    if 'slots' not in req_df.columns:
        req_df['slots'] = ''

    has_gating_mhr = 'gating_mean_headroom' in req_df.columns
    has_warning_mhr = 'warning_mean_headroom' in req_df.columns

    if not has_gating_mhr:
        print("WARNING: No gating_mean_headroom column in requirements file")
    if not has_warning_mhr:
        print("WARNING: No warning_mean_headroom column in requirements file")

    perf_dfs = []
    for csv_path in perf_csvs:
        if not os.path.exists(csv_path):
            print(f"WARNING: Performance CSV not found, skipping: {csv_path}")
            continue
        perf_dfs.append(pd.read_csv(csv_path, dtype=np.float64, na_values=CICD_NONE_FORMAT))

    if len(perf_dfs) == 0:
        print("ERROR: No valid performance CSV files found")
        return 1

    print(f"Loaded {len(req_df)} requirement entries from {input_file}")
    print(f"Loaded {len(perf_dfs)} performance CSV files")
    print(f"Using K_gating={K_GATING}, K_warning={K_WARNING} for suggested headroom computation")

    all_slots = sorted(perf_dfs[0]['slot'].unique())
    effective_slots = build_effective_slot_groups(req_df, all_slots)

    results = []
    any_flags = False

    for idx, row in req_df.iterrows():
        metric = row['metric_name']
        target_slots = effective_slots[idx]
        pct = is_percentage_metric(metric)

        worst_cases = []
        for perf_df in perf_dfs:
            if metric not in perf_df.columns:
                continue
            slot_data = perf_df[perf_df['slot'].isin(target_slots)][metric]
            worst = compute_worst_case(slot_data, metric)
            if not np.isnan(worst):
                worst_cases.append(worst)

        if len(worst_cases) >= 2:
            raw_std = np.std(worst_cases, ddof=1)
        elif len(worst_cases) == 1:
            raw_std = 0.0
        else:
            raw_std = 0.0

        if pct:
            suggested_gating = math.floor(raw_std * K_GATING * 1000) / 1000
            suggested_warning = math.floor(raw_std * K_WARNING * 1000) / 1000
        else:
            suggested_gating = math.ceil(raw_std * K_GATING)
            suggested_warning = math.ceil(raw_std * K_WARNING)

        current_gating = float(row['gating_mean_headroom']) if has_gating_mhr else None
        current_warning = float(row['warning_mean_headroom']) if has_warning_mhr else None

        gating_flag = ''
        warning_flag = ''
        if current_gating is not None and suggested_gating > current_gating:
            gating_flag = ' <<'
            any_flags = True
        if current_warning is not None and suggested_warning > current_warning:
            warning_flag = ' <<'
            any_flags = True

        results.append({
            'metric': metric,
            'slots_display': format_slot_list(target_slots, all_slots),
            'n_runs': len(worst_cases),
            'std': raw_std,
            'suggested_gating': suggested_gating,
            'suggested_warning': suggested_warning,
            'current_gating': current_gating,
            'current_warning': current_warning,
            'gating_flag': gating_flag,
            'warning_flag': warning_flag,
            'pct': pct,
        })

    print("\n" + "=" * 160)
    print("Variance Characterization Diagnostic")
    print("=" * 160)

    header = (f"  {'metric_name':35s} {'slots':16s} {'n':>3s}  {'std':>11s}  "
              f"{'sug_gate':>9s} {'cur_gate':>12s}  {'sug_warn':>9s} {'cur_warn':>12s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        metric = r['metric']
        pct = r['pct']
        unit = 'pp' if pct else 'us'

        if pct:
            std_s = f"{r['std']:8.3f}{unit}"
            sg = f"{r['suggested_gating']:9.3f}"
            sw = f"{r['suggested_warning']:9.3f}"
            cg = f"{r['current_gating']:9.3f}" if r['current_gating'] is not None else "      N/A"
            cw = f"{r['current_warning']:9.3f}" if r['current_warning'] is not None else "      N/A"
        else:
            std_s = f"{r['std']:8.1f}{unit}"
            sg = f"{r['suggested_gating']:9.0f}"
            sw = f"{r['suggested_warning']:9.0f}"
            cg = f"{r['current_gating']:9.0f}" if r['current_gating'] is not None else "      N/A"
            cw = f"{r['current_warning']:9.0f}" if r['current_warning'] is not None else "      N/A"

        print(f"  {metric:35s} {r['slots_display']:16s} {r['n_runs']:3d}  {std_s:>11s}  "
              f"{sg} {cg}{r['gating_flag']:3s}  {sw} {cw}{r['warning_flag']:3s}")

    print("=" * 160)

    if any_flags:
        print("\n  << = suggested headroom exceeds current configured value (consider updating)")
    else:
        print("\n  All configured headroom values are at or above suggested values.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic: characterize empirical variance and compare against configured mean_headroom values"
    )
    parser.add_argument(
        "input_requirements", type=str,
        help="Perf_requirements CSV template (with gating_mean_headroom/warning_mean_headroom columns)"
    )
    parser.add_argument(
        "performance_csvs", nargs="+", type=str,
        help="One or more perf.csv files from historical runs"
    )
    args = parser.parse_args()

    sys.exit(main(args))
