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

"""
Performance Plotting Utilities

This module provides reusable utilities for creating interactive performance
analysis visualizations, particularly focused on statistics tables with 
integrated CCDF (Complementary Cumulative Distribution Function) viewers
and optional whisker plots.

Main functionality:
- create_interactive_stats_table: Generate sortable/searchable table with CCDF viewer
                                  and optional whisker plot by slot
- format_slot_ranges: Format slot numbers into compact range notation
"""

import pandas as pd
import numpy as np
from bokeh.models import DataTable, TableColumn, ColumnDataSource, NumberFormatter
from bokeh.models.widgets import TextInput
from bokeh.layouts import column, row
from bokeh.models.callbacks import CustomJS
from aerial_postproc.logplot import ccdf_comparison, whisker_comparison


def format_slot_ranges(slots):
    """
    Format a list of slots into a compact range string.
    Example: [0,1,2,3,45,46,47,78,79] -> "0-3,45-47,78-79"
    
    Args:
        slots: iterable of slot numbers
    
    Returns:
        String with comma-separated ranges
    """
    if len(slots) == 0:
        return ""
    
    slots_sorted = sorted(set(slots))
    ranges = []
    start = slots_sorted[0]
    end = slots_sorted[0]
    
    for i in range(1, len(slots_sorted)):
        if slots_sorted[i] == end + 1:
            # Continue the current range
            end = slots_sorted[i]
        else:
            # End current range and start new one
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = slots_sorted[i]
            end = slots_sorted[i]
    
    # Add the last range
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{end}")
    
    return ",".join(ranges)


def create_interactive_stats_table(df, name_field, value_field, slot_field=None, percentiles=[50, 99, 99.99]):
    """
    Create an interactive statistics table with integrated CCDF viewer and optional whisker plot.
    
    The table displays statistics grouped by name_field, with clickable rows
    that display a CCDF plot of the value_field distribution for that group.
    If slot_field is provided, a whisker plot showing distribution by slot
    is displayed below the table.
    
    Args:
        df: Input DataFrame containing the data
        name_field: Column name for grouping (e.g., "pipeline_name")
        value_field: Column name for the metric to analyze (e.g., "duration")
        slot_field: Optional column name for slots (e.g., "slot"). If provided,
                   adds a column showing slot ranges in compact notation and
                   includes a whisker plot underneath showing distribution by slot.
        percentiles: List of percentiles to compute (default: [50, 99, 99.99])
    
    Returns:
        Bokeh layout with:
        - Table (left) and interactive CCDF viewer (right) on top
        - Whisker plot underneath (only if slot_field provided)
    
    Raises:
        ValueError: If required fields don't exist in dataframe
    """
    # Validate inputs
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    if name_field not in df.columns:
        raise ValueError(f"name_field '{name_field}' not found in dataframe")
    
    if value_field not in df.columns:
        raise ValueError(f"value_field '{value_field}' not found in dataframe")
    
    if slot_field and slot_field not in df.columns:
        raise ValueError(f"slot_field '{slot_field}' not found in dataframe")
    
    # Calculate statistics for each unique name
    stats_list = []
    ccdf_plots = {}
    whisker_plots = {}
    
    print(f"\nGenerating statistics table for {df[name_field].nunique()} unique {name_field} values...")
    
    for name in sorted(df[name_field].unique()):
        group_data = df[df[name_field] == name].copy()
        
        # Calculate percentiles
        values = group_data[value_field].values
        stats = {'name': name, 'count': len(group_data)}
        
        for p in percentiles:
            stats[f'p{p}'] = np.percentile(values, p)
        
        # Add slot ranges if requested
        if slot_field:
            slots = group_data[slot_field].dropna().astype(int).values
            stats['slots'] = format_slot_ranges(slots)
        
        stats_list.append(stats)
        
        # Pre-compute CCDF plot for this group
        # Calculate data-driven axis range with 10% padding
        if len(values) > 0:
            val_min = values.min()
            val_max = values.max()
            val_range = val_max - val_min
            xstart = val_min - val_range / 10
            xstop = val_max + val_range / 10
        else:
            xstart = 0
            xstop = 100
        
        ccdf_fig = ccdf_comparison([group_data], value_field,
                                   title=f"{value_field.upper()} CCDF: {name}",
                                   xlabel=f"{value_field} (µs)",
                                   xdelta=1,
                                   xstart=xstart,
                                   xstop=xstop,
                                   height=500,
                                   width=700)
        ccdf_fig.visible = False
        ccdf_plots[name] = ccdf_fig
        
        # Pre-compute whisker plot for this group (if slot_field provided)
        if slot_field:
            whisker_fig = whisker_comparison(
                [group_data],
                value_field,
                ['file_index', slot_field],
                title=f"{value_field.title()} by {slot_field.title()}: {name}",
                ylabel=f"{value_field.title()} (µs)"
            )
            
            # Set y-axis range with 10% padding
            if len(values) > 0:
                whisker_fig.y_range.start = xstart
                whisker_fig.y_range.end = xstop
            
            whisker_fig.visible = False
            whisker_plots[name] = whisker_fig
    
    print(f"  Pre-computed {len(ccdf_plots)} CCDF plots")
    if slot_field:
        print(f"  Pre-computed {len(whisker_plots)} whisker plots")
    
    # Create DataFrame for table
    stats_df = pd.DataFrame(stats_list)
    
    # Create ColumnDataSource
    source = ColumnDataSource(stats_df)
    
    # Define table columns dynamically
    columns = [
        TableColumn(field="name", title=name_field.replace('_', ' ').title(), width=350),
        TableColumn(field="count", title="Count", width=80, 
                   formatter=NumberFormatter(format="0")),
    ]
    
    # Add percentile columns
    for p in percentiles:
        if p == int(p):
            title = f"P{int(p)} {value_field.title()} (µs)"
        else:
            title = f"P{p:.2f} {value_field.title()} (µs)"
        
        columns.append(
            TableColumn(field=f"p{p}", title=title, width=130,
                       formatter=NumberFormatter(format="0.00"))
        )
    
    # Add slots column if requested
    if slot_field:
        columns.append(
            TableColumn(field="slots", title="Slots", width=450)
        )
    
    # Calculate table width
    table_width = sum(col.width for col in columns)
    
    # Create DataTable (sortable by default)
    data_table = DataTable(source=source, columns=columns, width=table_width, height=600,
                          sortable=True, reorderable=False)
    
    # Add search box
    search_input = TextInput(placeholder=f"Search {name_field}...", width=400)
    
    # Build field list for JavaScript callback
    field_list = ['name', 'count'] + [f'p{p}' for p in percentiles]
    if slot_field:
        field_list.append('slots')
    
    # JavaScript callback for filtering
    filter_callback = CustomJS(args=dict(source=source, table=data_table, 
                                        stats_df=stats_df.to_dict('list'),
                                        field_list=field_list), code=f"""
        const search_text = cb_obj.value.toLowerCase();
        const orig_data = stats_df;
        
        // Filter data based on search text
        const indices = [];
        for (let i = 0; i < orig_data.name.length; i++) {{
            if (orig_data.name[i].toLowerCase().includes(search_text)) {{
                indices.push(i);
            }}
        }}
        
        // Update source with filtered data
        const new_data = {{}};
        for (const field of field_list) {{
            new_data[field] = [];
        }}
        
        for (const i of indices) {{
            for (const field of field_list) {{
                new_data[field].push(orig_data[field][i]);
            }}
        }}
        
        source.data = new_data;
        source.change.emit();
    """)
    
    search_input.js_on_change('value', filter_callback)
    
    # Create table layout (search box + table)
    table_layout = column(search_input, data_table)
    
    # Create container for all CCDF plots
    sorted_names = sorted(ccdf_plots.keys())
    
    # Show first CCDF by default
    if sorted_names:
        ccdf_plots[sorted_names[0]].visible = True
    
    # Stack all CCDF plots (only one visible at a time)
    ccdf_container = column(*[ccdf_plots[name] for name in sorted_names])
    
    # Handle whisker plots if slot_field provided
    if slot_field and whisker_plots:
        # Show first whisker plot by default
        if sorted_names:
            whisker_plots[sorted_names[0]].visible = True
        
        # Stack all whisker plots (only one visible at a time)
        whisker_container = column(*[whisker_plots[name] for name in sorted_names])
        
        # Add JavaScript callback for row selection (updates both CCDF and whisker)
        selection_callback = CustomJS(
            args=dict(
                source=source,
                ccdf_plots={name: ccdf_plots[name] for name in sorted_names},
                whisker_plots={name: whisker_plots[name] for name in sorted_names},
                name_list=sorted_names
            ),
            code="""
            // Get selected indices
            const selected = source.selected.indices;
            
            if (selected.length > 0) {
                // Preserve scroll position
                const scrollX = window.scrollX || window.pageXOffset;
                const scrollY = window.scrollY || window.pageYOffset;
                
                // Get the name from the selected row
                const selected_idx = selected[0];
                const selected_name = source.data['name'][selected_idx];
                
                // Hide all CCDF plots
                for (const name of name_list) {
                    ccdf_plots[name].visible = false;
                }
                
                // Show the selected CCDF plot
                if (ccdf_plots[selected_name]) {
                    ccdf_plots[selected_name].visible = true;
                }
                
                // Hide all whisker plots
                for (const name of name_list) {
                    whisker_plots[name].visible = false;
                }
                
                // Show the selected whisker plot
                if (whisker_plots[selected_name]) {
                    whisker_plots[selected_name].visible = true;
                }
                
                // Restore scroll position after a brief delay to let DOM update
                setTimeout(() => {
                    window.scrollTo(scrollX, scrollY);
                }, 10);
            }
            """
        )
        
        # Attach callback to table source selection
        source.selected.js_on_change('indices', selection_callback)
        
        # Create main layout with table on left, CCDF on right
        main_layout = row(table_layout, ccdf_container)
        
        # Return column layout: [table+CCDF] on top, whisker below
        return column(main_layout, whisker_container)
    else:
        # No whisker plots - just CCDF callback
        selection_callback = CustomJS(
            args=dict(
                source=source,
                ccdf_plots={name: ccdf_plots[name] for name in sorted_names},
                name_list=sorted_names
            ),
            code="""
            // Get selected indices
            const selected = source.selected.indices;
            
            if (selected.length > 0) {
                // Preserve scroll position
                const scrollX = window.scrollX || window.pageXOffset;
                const scrollY = window.scrollY || window.pageYOffset;
                
                // Get the name from the selected row
                const selected_idx = selected[0];
                const selected_name = source.data['name'][selected_idx];
                
                // Hide all CCDF plots
                for (const name of name_list) {
                    ccdf_plots[name].visible = false;
                }
                
                // Show the selected CCDF plot
                if (ccdf_plots[selected_name]) {
                    ccdf_plots[selected_name].visible = true;
                }
                
                // Restore scroll position after a brief delay to let DOM update
                setTimeout(() => {
                    window.scrollTo(scrollX, scrollY);
                }, 10);
            }
            """
        )
        
        # Attach callback to table source selection
        source.selected.js_on_change('indices', selection_callback)
        
        # Return just the table+CCDF layout
        return row(table_layout, ccdf_container)
