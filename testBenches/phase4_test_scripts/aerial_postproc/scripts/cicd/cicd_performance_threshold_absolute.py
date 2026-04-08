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
from aerial_postproc.cicd_utils import is_higher_is_better, is_percentage_metric, load_requirements, build_requirements_map


def get_metric_order(req_df):
    """
    Extract metric order from requirements file.
    
    Returns: list of metrics in order of first appearance
    """
    metric_order = []
    for metric in req_df['metric_name']:
        if metric not in metric_order:
            metric_order.append(metric)
    return metric_order


def validate_performance(perf_df, requirements_map):
    """
    Validate performance metrics against requirements.
    
    Returns a dictionary with results per metric, with sub-results per threshold:
    {metric: [
        {
            'threshold_value': float,
            'slots': [int],
            'worst_value': float,
            'worst_headroom': float,
            'worst_slot': int,
            'failed_slots': [int],
            'pass_fail': 'PASS'/'FAIL'
        },
        ...
    ]}
    """
    results = {}
    
    # Get list of slots from perf_df
    all_slots = sorted(perf_df['slot'].unique())
    
    # Process each metric in requirements
    for metric in requirements_map.keys():
        if metric not in perf_df.columns:
            print(f"WARNING: Metric '{metric}' not found in performance CSV")
            continue
        
        # Group slots by their (required_value, headroom) tuple, preserving insertion order
        threshold_groups = {}  # {(required_value, headroom): [slots]}
        
        for slot in all_slots:
            if slot not in requirements_map[metric]:
                continue
            
            key = requirements_map[metric][slot]
            if key not in threshold_groups:
                threshold_groups[key] = []
            threshold_groups[key].append(slot)
        
        # Process each threshold group in insertion order (matches CSV row order)
        metric_results = []
        
        for (required_value, headroom), group_slots in threshold_groups.items():
            if is_higher_is_better(metric):
                threshold_value = required_value + headroom
            else:
                threshold_value = required_value - headroom
            worst_headroom = float('inf')
            worst_value = None
            worst_slot = None
            failed_slots = []
            
            # Check each slot in this threshold group
            for slot in group_slots:
                # Get actual value from perf_df
                slot_data = perf_df[perf_df['slot'] == slot]
                if len(slot_data) == 0:
                    continue
                
                actual_value = slot_data[metric].iloc[0]
                
                # Skip NaN values
                if pd.isna(actual_value):
                    continue
                
                if is_higher_is_better(metric):
                    actual_headroom = actual_value - threshold_value
                else:
                    actual_headroom = threshold_value - actual_value
                
                # Check if pass (actual_headroom >= 0 means we met/exceeded threshold)
                passed = actual_headroom >= 0
                
                # Track worst case within this threshold group
                if actual_headroom < worst_headroom:
                    worst_headroom = actual_headroom
                    worst_value = actual_value
                    worst_slot = slot
                
                # Track failures
                if not passed:
                    failed_slots.append(slot)
            
            # Store results for this threshold group
            if worst_slot is not None:
                metric_results.append({
                    'threshold_value': threshold_value,
                    'slots': group_slots,
                    'worst_value': worst_value,
                    'worst_headroom': worst_headroom,
                    'worst_slot': worst_slot,
                    'failed_slots': failed_slots,
                    'pass_fail': 'FAIL' if len(failed_slots) > 0 else 'PASS'
                })
        
        # Store all threshold groups for this metric
        if metric_results:
            results[metric] = metric_results
    
    return results


def print_summary(results, metric_order):
    """Print console summary of validation results in requirements file order."""
    print("\n" + "=" * 100)
    print("=== Performance Validation Results ===")
    print("=" * 100)
    print()
    
    # Count total threshold groups and failures
    total_threshold_groups = sum(len(threshold_list) for threshold_list in results.values())
    failed_threshold_groups = sum(
        1 for threshold_list in results.values() 
        for threshold_result in threshold_list 
        if threshold_result['pass_fail'] == 'FAIL'
    )
    
    # Calculate max metric name length for alignment
    max_name_len = max(len(metric) for metric in results.keys()) if results else 0
    
    # Iterate in the order from requirements file
    for metric in metric_order:
        if metric not in results:
            continue
        
        # Get all threshold groups for this metric
        threshold_list = results[metric]
        
        unit = 'pp' if is_percentage_metric(metric) else 'us'
        
        # Print one line per threshold group
        for threshold_result in threshold_list:
            status = threshold_result['pass_fail']
            threshold_value = threshold_result['threshold_value']
            worst_val = threshold_result['worst_value']
            worst_hr = threshold_result['worst_headroom']
            worst_slot = threshold_result['worst_slot']
            
            padded_metric = metric.ljust(max_name_len)
            slot_str = f"{int(worst_slot):2d}"
            
            if is_percentage_metric(metric):
                thresh_str = f"thresh: {threshold_value:9.3f}{unit}"
                val_str = f"worst: {worst_val:9.3f}{unit}"
                hr_str = f"{worst_hr:7.3f}{unit}"
            else:
                thresh_str = f"thresh: {threshold_value:9.1f}{unit}"
                val_str = f"worst: {worst_val:9.1f}{unit}"
                hr_str = f"{worst_hr:7.1f}{unit}"
            
            if status == 'PASS':
                print(f"{padded_metric} : {status}  ({thresh_str}, {val_str}, headroom: {hr_str} @ slot {slot_str})")
            else:
                failed_list = ','.join(map(str, [int(s) for s in threshold_result['failed_slots']]))
                print(f"{padded_metric} : {status}  ({thresh_str}, {val_str}, headroom: {hr_str} @ slot {slot_str}, failed: [{failed_list}])")
    
    print()
    print("=" * 100)
    if failed_threshold_groups > 0:
        print(f"Overall: FAIL ({failed_threshold_groups}/{total_threshold_groups} threshold checks failed)")
    else:
        print(f"Overall: PASS ({total_threshold_groups}/{total_threshold_groups} threshold checks passed)")
    print("=" * 100)
    
    return failed_threshold_groups


def write_csv_output(results, output_file, metric_order):
    """Write detailed results to CSV file in requirements file order."""
    rows = []
    for metric in metric_order:
        if metric not in results:
            continue
        
        # Process each threshold group for this metric
        threshold_list = results[metric]
        for threshold_result in threshold_list:
            failed_str = ','.join(map(str, threshold_result['failed_slots'])) if threshold_result['failed_slots'] else ''
            rows.append({
                'metric_name': metric,
                'threshold_value': threshold_result['threshold_value'],
                'worst_value': threshold_result['worst_value'],
                'worst_headroom': threshold_result['worst_headroom'],
                'worst_slot': threshold_result['worst_slot'],
                'failed_slots': failed_str,
                'pass_fail': threshold_result['pass_fail']
            })
    
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_file, index=False)
    print(f"\nDetailed results written to: {output_file}")


def main(args):
    try:
        req_df = load_requirements(args.requirements_csv)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Loaded {len(req_df)} requirement entries from {args.requirements_csv}")
    
    # Load performance CSV
    if not os.path.exists(args.performance_csv):
        print(f"ERROR: Performance CSV not found: {args.performance_csv}")
        sys.exit(1)
    
    try:
        perf_df = pd.read_csv(args.performance_csv, dtype=np.float64, na_values=CICD_NONE_FORMAT)
    except Exception as e:
        print(f"ERROR: Failed to read performance CSV: {e}")
        sys.exit(1)
    
    print(f"Loaded performance data with {len(perf_df)} slots")
    
    # Extract metric order from requirements file
    metric_order = get_metric_order(req_df)
    
    # Build requirements map with slot-specific overrides
    all_slots = sorted(perf_df['slot'].unique())
    requirements_map = build_requirements_map(req_df, all_slots)
    
    print(f"Built requirements for {len(requirements_map)} metrics")
    
    # Validate performance against requirements
    results = validate_performance(perf_df, requirements_map)
    
    # Print summary in requirements file order
    failed_count = print_summary(results, metric_order)
    
    # Write CSV output if requested
    if args.output_csv:
        write_csv_output(results, args.output_csv, metric_order)
    
    # Return exit code
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate performance metrics against absolute requirements"
    )
    parser.add_argument(
        "performance_csv", 
        help="Input performance CSV file (e.g., from cicd_performance_metrics.py)"
    )
    parser.add_argument(
        "requirements_csv",
        help="Requirements CSV file with required values and headroom (e.g., scripts/requirements_4tr.csv)"
    )
    parser.add_argument(
        "-o", "--output_csv",
        help="Optional: Write detailed results to CSV file"
    )
    
    args = parser.parse_args()
    
    exit_code = main(args)
    sys.exit(exit_code)
