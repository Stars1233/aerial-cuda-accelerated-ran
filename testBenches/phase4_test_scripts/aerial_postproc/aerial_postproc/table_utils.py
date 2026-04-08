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
Table utility for creating consistent tables in bokeh
"""

import pandas as pd
from bokeh.models import Div


def create_html_table(df, title=None, 
                     include_units_in_headers=True, unit_suffix="(μs)",
                     exclude_unit_columns=None, table_width="100%"):
    """
    Create an HTML table from a pandas DataFrame.
    
    The generated table is formatted for easy copy-paste into Excel or other applications.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to convert to an HTML table
    title : str, optional
        Title to display above the table. If None, no title is shown (default: None)
    include_units_in_headers : bool, optional
        Whether to append unit suffix to column headers (default: True)
    unit_suffix : str, optional
        The unit suffix to append to headers (default: "(μs)")
    exclude_unit_columns : list, optional
        List of column names to exclude from unit suffix (default: None)
    table_width : str, optional
        CSS width value for the table (default: "100%")
        
    Returns:
    --------
    bokeh.models.Div
        A Bokeh Div widget containing the HTML table
    """
    
    if exclude_unit_columns is None:
        exclude_unit_columns = []
    
    # Create HTML table
    html_table = f"<div style='margin-top: 20px; overflow-x: auto;'>"
    if title is not None:
        html_table += f"<h4>{title}:</h4>"
    html_table += f"<table border='1' style='border-collapse: collapse; font-family: Arial, sans-serif; width: {table_width}; table-layout: auto;'>"
    
    # Calculate dynamic column widths for better multi-column handling
    num_cols = len(df.columns)
    first_col_width = max(20, 40 - num_cols * 2)  # Shrink first column as more files are added
    other_col_width = max(8, (60 - first_col_width) / (num_cols - 1)) if num_cols > 1 else 60
    
    # Header row
    html_table += "<tr style='background-color: #f0f0f0; font-weight: bold;'>"
    for i, col in enumerate(df.columns):
        if include_units_in_headers and col not in exclude_unit_columns:
            header_text = f"{col} {unit_suffix}"
        else:
            header_text = str(col)
        
        # Use different widths for first column (metric names) vs data columns
        col_width = f"{first_col_width}%" if i == 0 else f"{other_col_width}%"
        html_table += f"<td style='padding: 6px 4px; border: 1px solid #ccc; width: {col_width}; text-align: center; font-size: 12px;'>{header_text}</td>"
    html_table += "</tr>"
    
    # Data rows
    for _, row in df.iterrows():
        html_table += "<tr>"
        for i, col in enumerate(df.columns):
            value = row[col] if pd.notna(row[col]) and row[col] != '' else ''
            # Left align first column (metrics), center align data columns
            text_align = "left" if i == 0 else "center"
            html_table += f"<td style='padding: 6px 4px; border: 1px solid #ccc; text-align: {text_align}; font-size: 12px;'>{value}</td>"
        html_table += "</tr>"
    
    html_table += "</table></div>"
    html_table += "<p style='font-size: 12px; color: #666;'>Tip: Select the table above and copy (Ctrl+C) to paste into Excel</p>"
    
    return Div(text=html_table)


 
