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

import re
import pandas as pd
import numpy as np
from functools import partial
import cxxfilt
from bokeh.layouts import column, row

from aerial_postproc.logplot import whisker_comparison, ccdf_breakout, comparison_legend, plot_all_tasks_cpu_timeline
from aerial_postproc.perfplotter import create_interactive_stats_table

# Saving these patterns here in case we add communications task messages back in
#EXTERNAL_CORRELATION pattern
corr_pattern = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[CUPHY\.CUPTI\] EXTERNAL_CORRELATION externalKind (\d+), correlationId (\d+), externalId (\d+)")
#CONCURRENT_KERNEL pattern
kernel_pattern = re.compile(r"(\d+\:\d+\:\d+\.\d+) I.*\[CUPHY\.CUPTI\] CONCURRENT_KERNEL \[ (\d+), (\d+) \] duration (\d+), \"(.*)\", correlationId (\d+), grid \[ (\d+), (\d+), (\d+) \], block \[ (\d+), (\d+), (\d+) \], sharedMemory \(static (\d+), dynamic (\d+)\), deviceId (\d+), contextId (\d+), streamId (\d+), graphId (\d+), graphNodeId (\d+), channelId (\d+)")

def cupti_parse_corr_line(line):
    global corr_pattern
    found = corr_pattern.match(line)
    result = []
    if found:
        log_timestamp = found[1]
        external_kind = int(found[2])
        correlation_id = int(found[3])
        external_id = int(found[4])
        # sfn = external_id >> 8
        # slot = external_id & 0xFF
        result.append({'log_timestamp':log_timestamp,
                       'external_kind': external_kind,
                       'correlation_id':correlation_id,
                       'external_id':external_id,})

    return result

def get_full_name(mangled_name):
    return cxxfilt.demangle(mangled_name)

def get_compact_name(mangled_name):
    """
    Extract a compact, readable kernel name from mangled C++ name.
    
    Examples:
        void block_fft_kernel<...>(float2*) -> block_fft_kernel
        void prach_compute_pdp<float2, float>(...) -> prach_compute_pdp
        aerial_fh::memset_kernel(...) -> memset_kernel
        memset_kernel -> memset_kernel
    """
    # Run demangling
    name = cxxfilt.demangle(mangled_name)
    
    # Strategy: Extract just the function name, ignoring templates and parameters
    # 1. Remove return type (everything before the function name)
    # 2. Extract function name (up to '<' or '(')
    # 3. Strip namespace prefix if present
    
    # Find the function name by looking for the pattern before '(' or '<'
    # Common patterns:
    #   - "void function_name<...>(...)"
    #   - "namespace::function_name(...)"
    #   - "function_name"
    
    # Remove leading return type (find last space before function name)
    # But be careful with templates like "void function<T>()" where space is in the return type
    parts = name.split('(', 1)  # Split at first '('
    if len(parts) > 0:
        before_params = parts[0]
        
        # Extract just the function name part (before any '<' for templates)
        func_with_templates = before_params.split('<', 1)[0]
        
        # Now strip the return type - find the last token
        # Handle cases like "void function_name" or "namespace::Class::function_name"
        tokens = func_with_templates.split()
        if len(tokens) > 1:
            # Has return type, get last token
            func_part = tokens[-1]
        else:
            # No space, just the function name
            func_part = func_with_templates
        
        # Strip namespace prefix (take text after last '::')
        if '::' in func_part:
            func_part = func_part.split('::')[-1]
        
        return func_part.strip()
    
    # Fallback: return the full demangled name if we can't parse it
    return name


def cupti_parse_kernel_line(line, name_filters=None):

    if name_filters is not None:
        name_filters = [aa.lower() for aa in name_filters]

    global kernel_pattern
    found = kernel_pattern.match(line)
    result = []

    if found:

        mangled_name = found[5]
        should_include = True
        if name_filters is not None:
            should_include = len([aa for aa in name_filters if mangled_name.lower().find(aa)>=0]) > 0

        if should_include:
            log_timestamp = found[1]
            start_time = int(found[2])
            end_time = int(found[3])
            duration = int(found[4])
            correlation_id = int(found[6])
            grid_dim = (int(found[7]), int(found[8]), int(found[9]))
            block_dim = (int(found[10]), int(found[11]), int(found[12]))
            shared_memory = (int(found[13]), int(found[14]))
            device_id = int(found[15])
            context_id = int(found[16])
            stream_id = int(found[17])
            graph_id = int(found[18])
            graph_node_id = int(found[19])
            channel_id = int(found[20])

            result.append({'log_timestamp':log_timestamp,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration/1e3,
                        'mangled_name': mangled_name,
                        'full_name': get_full_name(mangled_name),
                        'compact_name': get_compact_name(mangled_name),
                        'correlation_id': correlation_id,
                        'grid_dim': grid_dim,
                        'block_dim': block_dim,
                        'shared_memory': shared_memory,
                        'device_id': device_id,
                        'context_id': context_id,
                        'stream_id': stream_id,
                        'graph_id': graph_id,
                        'graph_node_id': graph_node_id,
                        'channel_id': channel_id})

    return result

def phy_parse_ti_and_cupti(phy_filename,start_tir,end_tir,num_proc=8,name_filters=None):
    import tempfile
    import os
    from aerial_postproc.parsenator import Parsenator
    from aerial_postproc.logparse import get_ref_t0, parse_ti_line

    #Establish arbitrary reference
    ref_t0 = get_ref_t0(phy_filename)

    # Create temporary files for separated log content
    # CUPTI messages are delayed in the log relative to TI messages
    # So we need to separate them first, then parse independently
    # Using single-pass Python approach (2.46x faster than two-pass grep for I/O-bound operations)
    cupti_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_cupti.log', dir='/tmp')
    ti_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_ti.log', dir='/tmp')
    
    print(f"Extracting CUPTI messages to {cupti_temp.name}")
    print(f"Extracting TI messages to {ti_temp.name}")
    
    # Single pass through the file - extract both CUPTI and TI messages
    with open(phy_filename, 'r') as infile:
        for line in infile:
            if '[CUPHY.CUPTI]' in line:
                cupti_temp.write(line)
            elif '{TI}' in line:
                ti_temp.write(line)
    
    cupti_temp.close()
    ti_temp.close()
    
    try:
        # Parse TI messages with time filtering
        print("Parsing TI messages...")
        p1 = Parsenator(ti_temp.name,
                       [parse_ti_line],
                       [None],
                       num_processes=num_proc,
                       ref_t0=ref_t0,
                       start_tir=start_tir,
                       end_tir=end_tir)
        parsed_ti = p1.parse()
        df_ti = parsed_ti[0]
        
        # Parse CUPTI messages without time filtering (correlation IDs link them)
        print("Parsing CUPTI messages...")
        p2 = Parsenator(cupti_temp.name,
                       [cupti_parse_corr_line,
                        partial(cupti_parse_kernel_line, name_filters=name_filters)],
                       [None,
                        ['log_timestamp','start_time','end_time','duration','mangled_name',
                         'full_name','compact_name','correlation_id']],
                       num_processes=num_proc,
                       ref_t0=ref_t0,
                       start_tir=None,  # Don't filter CUPTI by time
                       end_tir=None)
        parsed_cupti = p2.parse()
        df_corr = parsed_cupti[0]
        df_kernel = parsed_cupti[1]
        
    finally:
        # Clean up temporary files
        if os.path.exists(cupti_temp.name):
            os.unlink(cupti_temp.name)
        if os.path.exists(ti_temp.name):
            os.unlink(ti_temp.name)

    # STEP 1: Detect missing correlation records using outer join
    # This allows us to see kernels that don't have EXTERNAL_CORRELATION records
    print("\nChecking for missing EXTERNAL_CORRELATION records...")
    check_df = pd.merge(df_corr, df_kernel, on=["correlation_id"], 
                        how="outer", validate="one_to_many", indicator=True)
    
    corr_only = check_df[check_df['_merge'] == 'left_only']
    kernel_only = check_df[check_df['_merge'] == 'right_only']
    both = check_df[check_df['_merge'] == 'both']
    
    total_kernels = len(df_kernel)
    matched_kernels = len(both)
    missing_corr_kernels = len(kernel_only)
    
    print(f"\nMerge statistics (CORR + KERNEL):")
    print(f"  Total KERNEL records parsed: {total_kernels}")
    print(f"  Matched (both): {matched_kernels} rows ({100*matched_kernels/total_kernels:.1f}%)")
    print(f"  CORR only (no kernel): {len(corr_only)} rows")
    print(f"  KERNEL only (no corr): {missing_corr_kernels} rows ({100*missing_corr_kernels/total_kernels:.1f}%)")
    
    if len(corr_only) > 0:
        print(f"\n  WARNING: {len(corr_only)} EXTERNAL_CORRELATION records without matching kernel!")
        print(f"  Sample correlation IDs in CORR but not in KERNEL:")
        print(f"    {corr_only['correlation_id'].head(5).tolist()}")
    
    if len(kernel_only) > 0:
        missing_pct = 100 * missing_corr_kernels / total_kernels
        print(f"\n  WARNING: {missing_corr_kernels} kernel executions missing EXTERNAL_CORRELATION records!")
        print(f"  This is {missing_pct:.1f}% of all kernel executions.")
        print(f"  Sample correlation IDs in KERNEL but not in CORR:")
        print(f"    {kernel_only['correlation_id'].head(5).tolist()}")
        
        # Show which kernels are affected
        if 'compact_name' in kernel_only.columns:
            kernel_counts = kernel_only['compact_name'].value_counts().head(10)
            print(f"\n  Top kernel types without correlation records:")
            for kernel_name, count in kernel_counts.items():
                print(f"    {kernel_name}: {count}")
        
        # Raise error if more than 1% of kernels are missing correlation records
        if missing_pct > 1.0:
            raise ValueError(
                f"\nERROR: {missing_pct:.1f}% of kernel executions are missing EXTERNAL_CORRELATION records!\n"
                f"This indicates incomplete CUPTI instrumentation. The log may be corrupted or truncated.\n"
                f"Missing: {missing_corr_kernels} out of {total_kernels} kernel executions.\n"
                f"Without EXTERNAL_CORRELATION records, kernels cannot be mapped to slots/timestamps."
            )
    
    # STEP 2: Perform the actual merge using inner join (only usable data)
    # We can only process kernels that have correlation records, since we need external_id
    # to link to slot timing information
    print(f"\nProceeding with {matched_kernels} matched kernel records...")
    merged_df_kernel = pd.merge(df_corr, df_kernel, on=["correlation_id"], validate="one_to_many")
    
    # Merge sfn/slot from df_ti into df_kernel by matching external_id with t0_timestamp
    # Note: df_ti is filtered by start_tir/end_tir, but df_kernel is not
    # So some df_kernel rows may have NaN sfn/slot if they fall outside the TI time range
    df_ti_unique = df_ti[['t0_timestamp', 'sfn', 'slot']].drop_duplicates()
    
    merged_df_kernel = pd.merge(merged_df_kernel, df_ti_unique, 
                                left_on='external_id', right_on='t0_timestamp', 
                                how='left')
    
    # Drop redundant t0_timestamp column (we already have external_id)
    merged_df_kernel.drop(columns=['t0_timestamp'], inplace=True)
    
    # Filter out CUPTI data that has NaN sfn or slot (falls outside TI time range)
    rows_before = len(merged_df_kernel)
    merged_df_kernel = merged_df_kernel[merged_df_kernel['sfn'].notna()].copy()
    rows_after = len(merged_df_kernel)
    
    if rows_before > rows_after:
        print(f"\nFiltered CUPTI data: removed {rows_before - rows_after} rows with NaN sfn/slot")
        print(f"  Remaining: {rows_after} CUPTI kernel invocations")
    
    # Add kernel_name and relative timing columns to df_kernel (after filtering for efficiency)
    # call_number: which call of the same kernel within a slot (0, 1, 2, ...) ordered by start_time
    # kernel_name: combines compact_name with call_number (e.g., "ldpc_encode (Call 0)")
    # start_deadline: start time relative to T0 (external_id) in microseconds
    # end_deadline: end time relative to T0 (external_id) in microseconds
    
    # Sort by external_id, compact_name, and start_time to ensure chronological ordering
    merged_df_kernel = merged_df_kernel.sort_values(['external_id', 'compact_name', 'start_time']).reset_index(drop=True)
    
    # Assign call numbers based on chronological order within each slot
    merged_df_kernel['call_number'] = merged_df_kernel.groupby(['external_id', 'compact_name']).cumcount()
    merged_df_kernel['kernel_name'] = merged_df_kernel['compact_name'] + ' (Call ' + merged_df_kernel['call_number'].astype(str) + ')'
    merged_df_kernel['start_deadline'] = (merged_df_kernel['start_time'] - merged_df_kernel['external_id']) / 1e3
    merged_df_kernel['end_deadline'] = (merged_df_kernel['end_time'] - merged_df_kernel['external_id']) / 1e3
    
    # Create aggregate records for kernels with multiple calls per slot
    # For kernels with multiple calls (Call 0, Call 1, ...), create an "(All Calls)" record
    # that spans from the first call's start to the last call's end
    grouped = merged_df_kernel.groupby(['external_id', 'compact_name'])
    multi_call_groups = grouped.filter(lambda x: len(x) > 1).groupby(['external_id', 'compact_name'])
    
    if len(multi_call_groups) > 0:
        # For each multi-call group, create an aggregate record
        aggregate_records = []
        for (external_id, compact_name), group in multi_call_groups:
            # Take the first row as a template and modify the relevant fields
            agg_record = group.iloc[0].copy()
            agg_record['start_time'] = group['start_time'].min()
            agg_record['end_time'] = group['end_time'].max()
            agg_record['start_deadline'] = (agg_record['start_time'] - external_id) / 1e3
            agg_record['end_deadline'] = (agg_record['end_time'] - external_id) / 1e3
            agg_record['duration'] = (agg_record['end_time'] - agg_record['start_time']) / 1e3
            agg_record['call_number'] = -1  # Special value to indicate aggregate
            agg_record['kernel_name'] = compact_name + ' (All Calls)'
            aggregate_records.append(agg_record)
        
        # Append aggregate records to the main dataframe
        if aggregate_records:
            aggregate_df = pd.DataFrame(aggregate_records)
            merged_df_kernel = pd.concat([merged_df_kernel, aggregate_df], ignore_index=True)
            print(f"\nCreated {len(aggregate_records)} aggregate '(All Calls)' records for multi-call kernels (from filtered data)")

    return df_ti,merged_df_kernel

def create_t0_coverage_plots(df_ti, df_kernel):
    """
    Create T0 coverage visualization plots comparing TI and CUPTI data.
    
    Args:
        df_ti: DataFrame with TI data (must have 't0_timestamp' column)
        df_kernel: DataFrame with CUPTI kernel data (must have 'external_id' column)
    
    Returns:
        List of bokeh figure objects [coverage_plot, first_diff_plot]
    """
    from bokeh.plotting import figure
    
    # Get unique t0 timestamps from both dataframes
    ti_t0_unique = df_ti['t0_timestamp'].unique()
    ti_t0_sorted = np.sort(ti_t0_unique)
    
    kernel_t0_unique = df_kernel['external_id'].unique()
    kernel_t0_sorted = np.sort(kernel_t0_unique)
    
    print(f"\nPlot data preparation:")
    print(f"  TI unique t0s: {len(ti_t0_sorted)}")
    print(f"  CUPTI unique t0s: {len(kernel_t0_sorted)}")
    
    # Plot 1: T0 timestamps scatter plot (two traces)
    p1 = figure(width=1200, height=400, 
                title="T0 Timestamp Coverage",
                x_axis_label="T0 Timestamp (nanoseconds)",
                y_axis_label="Source")
    
    # TI data (y=1)
    p1.circle(ti_t0_sorted, np.ones(len(ti_t0_sorted)), 
              size=3, color='blue', alpha=0.5, legend_label='TI Data')
    
    # CUPTI data (y=0)
    p1.circle(kernel_t0_sorted, np.zeros(len(kernel_t0_sorted)), 
              size=3, color='red', alpha=0.5, legend_label='CUPTI Data')
    
    p1.legend.location = "top_right"
    p1.legend.click_policy = "hide"
    
    # Plot 2: First difference (delta between consecutive t0s)
    p2 = figure(width=1200, height=400, 
                title="T0 Timestamp First Difference (Gap between consecutive T0s)",
                x_axis_label="T0 Timestamp (nanoseconds)",
                y_axis_label="Delta (nanoseconds)",
                x_range=p1.x_range)  # Sync x-axis with p1
    
    # Calculate first differences
    if len(ti_t0_sorted) > 1:
        ti_diff = np.diff(ti_t0_sorted)
        ti_t0_mid = ti_t0_sorted[1:]  # x-coordinates for differences
        p2.circle(ti_t0_mid, ti_diff, 
                  size=3, color='blue', alpha=0.5, legend_label='TI Data')
    
    if len(kernel_t0_sorted) > 1:
        kernel_diff = np.diff(kernel_t0_sorted)
        kernel_t0_mid = kernel_t0_sorted[1:]  # x-coordinates for differences
        p2.circle(kernel_t0_mid, kernel_diff, 
                  size=3, color='red', alpha=0.5, legend_label='CUPTI Data')
    
    p2.legend.location = "top_right"
    p2.legend.click_policy = "hide"
    
    return [p1, p2]

def add_datetimes(io_df):
    """
    Adds start_datetime and end_datetime columns using t0_timestamp and deadlines.
    
    Args:
        io_df: DataFrame with t0_timestamp, start_deadline, end_deadline columns
    """
    start_timestamp = io_df['t0_timestamp'] + (io_df['start_deadline']*1000).astype(int)
    end_timestamp = io_df['t0_timestamp'] + (io_df['end_deadline']*1000).astype(int)
    io_df['start_datetime'] = pd.to_datetime(start_timestamp, unit='ns')
    io_df['end_datetime'] = pd.to_datetime(end_timestamp, unit='ns')

def create_gpu_kernel_timeline(df_kernel, x_range=None, crosshair_tool=None):
    """
    Create a timeline plot showing GPU kernel execution.
    
    Args:
        df_kernel: DataFrame with GPU kernel data
        x_range: Optional x_range to share with other plots
        crosshair_tool: Optional CrosshairTool to share with other plots
    
    Returns:
        Bokeh column layout
    """
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Category10_10
    
    # Prepare data
    df_plot = df_kernel.copy()
    df_plot['start_datetime'] = pd.to_datetime(df_plot['start_time'], unit='ns')
    df_plot['end_datetime'] = pd.to_datetime(df_plot['end_time'], unit='ns')
    
    # Sort kernel names by median start_deadline (across all slots) and create y-position mapping
    # This orders kernels by their typical execution time relative to slot start
    kernel_median_start = df_plot.groupby('kernel_name')['start_deadline'].median().sort_values()
    kernel_names_sorted = kernel_median_start.index.tolist()
    kernel_to_y = {name: idx for idx, name in enumerate(kernel_names_sorted)}
    
    # Create y positions and slot-based colors
    df_plot['yy'] = df_plot['kernel_name'].map(kernel_to_y)
    df_plot['slot_mod'] = (df_plot['slot'] % 10).astype(int)
    
    # Create tooltips
    TOOLTIPS = [
        ("kernel", "@kernel_name"),
        ("slot", "@slot"),
        ("duration", "@duration"),
    ]
    
    # Create figure (exactly matching CPU timeline style)
    fig1 = figure(
        title="GPU Kernel Timeline",
        x_axis_type="datetime",
        tooltips=TOOLTIPS,
        tools="tap,hover,xwheel_zoom,ywheel_zoom,xwheel_pan,ywheel_pan,box_zoom,reset",
        plot_width=1600,
        plot_height=600,
        x_range=x_range
    )
    
    # Draw bars for each slot color (0-9)
    for slot_mod in range(10):
        df_slot = df_plot[df_plot['slot_mod'] == slot_mod]
        if len(df_slot) > 0:
            # Create ColumnDataSource with only needed columns
            source = ColumnDataSource({
                'yy': df_slot['yy'].values,
                'start_datetime': df_slot['start_datetime'].values,
                'end_datetime': df_slot['end_datetime'].values,
                'kernel_name': df_slot['kernel_name'].values,
                'slot': df_slot['slot'].values,
                'duration': df_slot['duration'].values,
            })
            fig1.hbar(
                y='yy',
                height=0.8,
                left='start_datetime',
                right='end_datetime',
                source=source,
                color=Category10_10[slot_mod],
                line_width=1,
                line_color='black',
                legend_label=f'Slot % 10 = {slot_mod}'
            )
    
    # Configure y-axis with numeric labels (kernel names visible in tooltip)
    fig1.yaxis.ticker = list(range(len(kernel_names_sorted)))
    
    # Add crosshair tool if provided
    if crosshair_tool:
        fig1.add_tools(crosshair_tool)
    
    fig1.legend.location = "top_right"
    fig1.legend.click_policy = "hide"
    fig1.xaxis.axis_label = 'Time'
    fig1.yaxis.axis_label = 'Kernel Index'
    
    # Return as column layout (matching CPU timeline)
    return column([fig1])

def generate_kernel_duration_csv(df_kernel, csv_filename, percentile=0.50):
    """
    Generate CSV with kernel durations per slot.
    
    Args:
        df_kernel: DataFrame with kernel execution data (must have kernel_name, full_name, 
                   compact_name, slot, duration columns)
        csv_filename: Output CSV file path
        percentile: Percentile to compute (0.0-1.0, e.g., 0.99 for 99th percentile)
    
    CSV Format:
        Columns: kernel_name, full_name, compact_name, duration_0, duration_1, ..., duration_79
        Values: pth percentile of duration for that kernel in that slot (empty string if not present)
    """
    import csv
    
    print(f"\nGenerating CSV output to {csv_filename}")
    print(f"  Computing {percentile*100:.1f}th percentile for each kernel per slot...")
    
    # Group by kernel_name and slot, compute percentile
    # Note: kernel_name already includes call number (e.g., "ldpc_encode (Call 0)")
    grouped = df_kernel.groupby(['kernel_name', 'slot'])['duration'].quantile(percentile).reset_index()
    grouped.columns = ['kernel_name', 'slot', 'duration_percentile']
    
    # Get unique kernel names and their associated full_name and compact_name
    kernel_info = df_kernel[['kernel_name', 'full_name', 'compact_name']].drop_duplicates()
    
    # Pivot to get one row per kernel with columns for each slot
    pivoted = grouped.pivot(index='kernel_name', columns='slot', values='duration_percentile')
    
    # Merge with kernel info
    result = kernel_info.merge(pivoted, left_on='kernel_name', right_index=True, how='left')
    
    # Ensure all 80 slots are present as columns (0-79)
    for slot in range(80):
        if slot not in result.columns:
            result[slot] = ''
    
    # Reorder columns: kernel_name, full_name, compact_name, then slots 0-79
    column_order = ['kernel_name', 'full_name', 'compact_name'] + list(range(80))
    result = result[column_order]
    
    # Rename slot columns to duration_XX
    result.columns = ['kernel_name', 'full_name', 'compact_name'] + [f'duration_{i}' for i in range(80)]
    
    # Round all duration columns to 1 decimal place (0.1 usec resolution)
    duration_cols = [f'duration_{i}' for i in range(80)]
    for col in duration_cols:
        # Round numeric values, leave NaN as is for now
        result[col] = result[col].round(1)
    
    # Replace NaN with empty string
    result = result.fillna('')
    
    # Sort by kernel_name for consistency
    result = result.sort_values('kernel_name')
    
    # Write to CSV
    result.to_csv(csv_filename, index=False)
    
    print(f"  Wrote {len(result)} kernel types to CSV")
    print(f"  CSV saved to: {csv_filename}")


def unit_test(args):
    import os
    from bokeh.plotting import show, save
    from bokeh.io import output_file

    df_ti,df_kernel = phy_parse_ti_and_cupti(args.phy_filename,args.ignore_duration,args.max_duration,num_proc=args.num_proc,name_filters=args.name_filters)

    print(df_ti)
    print(f"\ndf_kernel shape: {df_kernel.shape}")
    print(f"df_kernel columns: {df_kernel.columns.tolist()}")

    #Print information on all names found
    print("\nKernel Summary:")
    print(f"  Total kernel invocations: {len(df_kernel)}")
    print(f"  Unique kernel names: {df_kernel['compact_name'].nunique()}")
    print(f"  Unique kernel stages: {df_kernel['kernel_name'].nunique()}")
    print(f"\nFull list of mangled/full/compact names:")
    for name in df_kernel.mangled_name.unique():
        print(f"  mangled: {name}")
        print(f"  full: {get_full_name(name)}")
        print(f"  compact: {get_compact_name(name)}")
        print("")
    
    print(f"\nDataframe columns: {df_kernel.columns.tolist()}")
    print(f"\nSample data:")
    print(df_kernel[['sfn', 'slot', 'compact_name', 'call_number', 'kernel_name', 
                     'start_deadline', 'end_deadline', 'duration']].head(10))

    # Generate CSV output if requested
    if args.csv_output:
        generate_kernel_duration_csv(df_kernel, args.csv_output, args.percentile)

    # #Add full duration (start to end of all calls for a slot)
    # ldpc_summary_df = df_kernel[df_kernel.compact_name=="ldpc2_BG1_split_index_fp_x2_desc_dyn_sm86_tb"]
    # ldpc_summary_df = ldpc_summary_df.assign(min_start_time=ldpc_summary_df.groupby('external_id').start_time.transform(min),
    #                                          max_end_time=ldpc_summary_df.groupby('external_id').end_time.transform(max))
    # ldpc_summary_df = ldpc_summary_df[['external_id','min_start_time','max_end_time']]
    # ldpc_summary_df['duration'] = (ldpc_summary_df['max_end_time'] - ldpc_summary_df['min_start_time']) / 1e3
    # ldpc_summary_df['slot'] = (ldpc_summary_df['external_id'] % (80*500000)) / 500000
    # ldpc_summary_df.drop_duplicates()

    # Create visualizations
    from bokeh.layouts import column
    
    # T0 coverage plots (if enabled via --enable_t0_plots flag)
    coverage_plots = []
    if args.enable_t0_plots:
        coverage_plots = create_t0_coverage_plots(df_ti, df_kernel)
    
    # Timeline plots (if enabled via -t flag)
    timeline_plots = []
    if args.enable_timeline:
        from bokeh.models import CrosshairTool
        
        # Create shared crosshair tool
        crosshair = CrosshairTool()
        
        # Create GPU kernel timeline first (establishes x_range)
        gpu_timeline_fig = create_gpu_kernel_timeline(df_kernel, x_range=None, crosshair_tool=crosshair)
        timeline_plots.append(gpu_timeline_fig)
        
        # Create CPU timeline sharing x_range from GPU timeline
        df_ti_subtask = df_ti[df_ti['subtask'] != 'Full Task'].copy()
        add_datetimes(df_ti_subtask)
        
        # Create empty dataframes for l2, tick, testmac
        df_l2_empty = pd.DataFrame()
        df_tick_empty = pd.DataFrame()
        df_testmac_empty = pd.DataFrame()
        
        # Get x_range from GPU timeline figure (it's wrapped in column, so extract it)
        gpu_fig = gpu_timeline_fig.children[0] if hasattr(gpu_timeline_fig, 'children') else gpu_timeline_fig
        
        # Create CPU timeline plot with shared x_range and crosshair
        cpu_timeline_fig = plot_all_tasks_cpu_timeline(
            df_ti_subtask, 
            df_l2_empty, 
            df_tick_empty, 
            df_testmac_empty,
            x_range=gpu_fig.x_range,  # Share x-axis with GPU timeline
            traces_only=True,
            cpu_only=False,
            crosshair_tool=crosshair,  # Share crosshair tool
            ul_only=False
        )
        timeline_plots.append(cpu_timeline_fig)
    
    # Kernel summary table with interactive CCDF display
    kernel_table = create_interactive_stats_table(
        df_kernel,
        name_field='kernel_name',
        value_field='duration',
        slot_field='slot',
        percentiles=[50, 99, 99.99]
    )
    
    # TI task summary tables with interactive CCDF and whisker displays
    # Filter to "Full Task" subtask only
    df_ti_full_task = df_ti[df_ti['subtask'] == 'Full Task'].copy()
    
    print(f"\nFiltered TI data to Full Task only: {len(df_ti_full_task)} rows")
    
    # ti_task_duration_table = create_interactive_stats_table(
    #     df_ti_full_task,
    #     name_field='task',
    #     value_field='duration',
    #     slot_field='slot',
    #     percentiles=[50, 99, 99.99]
    # )
    
    # ti_task_start_deadline_table = create_interactive_stats_table(
    #     df_ti_full_task,
    #     name_field='task',
    #     value_field='start_deadline',
    #     slot_field='slot',
    #     percentiles=[50, 99, 99.99]
    # )
    
    # ti_task_end_deadline_table = create_interactive_stats_table(
    #     df_ti_full_task,
    #     name_field='task',
    #     value_field='end_deadline',
    #     slot_field='slot',
    #     percentiles=[50, 99, 99.99]
    # )
    
    # Combine all visualizations
    fig_list = coverage_plots + timeline_plots + [kernel_table]
    
    # Output plots
    if(args.out_filename):
        out_filename = args.out_filename
    else:
        out_filename = "result.html"
    output_file(filename=out_filename, title=os.path.split(__file__)[1])
    print(f"\nWriting output to {out_filename}")
    if(args.out_filename):
        save(column(fig_list))
    else:
        show(column(fig_list))

if __name__ == '__main__':
    import argparse
    default_ignore_duration = 0.0
    default_max_duration = 999999999.0
    parser = argparse.ArgumentParser(
        description="Process and visualize Aerial log files."
    )
    parser.add_argument(
        "phy_filename", type=str, help="PHY filename"
    )
    parser.add_argument(
        "-o", "--out_filename", default="result.html", help="Filename for the output"
    )
    parser.add_argument(
        "-m", "--max_duration", default=default_max_duration, type=float, help="Maximum run duration to process"
    )
    parser.add_argument(
        "-i", "--ignore_duration", default=default_ignore_duration, type=float, help="Duration at beginning of run to ignore"
    )
    parser.add_argument(
        "-n", "--num_proc", default=8, type=int, help="Number of processes to use for parsing"
    )
    parser.add_argument(
        "-f", "--name_filters", nargs="+", default=None, help="Space separated list of mangled names (allows any partial match)"
    )
    parser.add_argument(
        "-t", "--enable_timeline", action="store_true", help="Enable CPU timeline plot"
    )
    parser.add_argument(
        "--enable_t0_plots", action="store_true", help="Enable overlay of CUPTI/TI timelines to verify there are no gaps in either dataset (T0 Timestamp Coverage and First Difference plots)"
    )
    parser.add_argument(
        "-c", "--csv_output", default=None, type=str, help="CSV output file for kernel duration statistics"
    )
    parser.add_argument(
        "-p", "--percentile", default=0.50, type=float, help="Percentile to compute for CSV output (0.0-1.0, default=0.50 for median)"
    )
    args = parser.parse_args()

    unit_test(args)
