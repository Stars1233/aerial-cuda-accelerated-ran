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
Section styling utility for consistent header generation across log analysis tools.
"""

from bokeh.models import Div

# Default grey shade styling
GrayShades = {
    'section': {
        'font-size': '200%',
        'text-align': 'center',
        'color': 'white',
        'background-color': '#2C3E50',     # Dark grey for main section headers
        'padding': '20px',
        'border-radius': '8px'
    },
    'sub_section': {
        'font-size': '140%',
        'font-weight': 'bold',
        'color': 'white',
        'background-color': '#7F8C8D',     # Medium grey for subsection headers
        'padding': '10px 15px',
        'margin': '15px 0 10px 0',
        'border-radius': '5px'
    },
    'sub_sub_section': {
        'font-size': '120%',
        'font-weight': 'bold',
        'color': 'white',
        'background-color': '#BDC3C7',     # Light grey for sub-subsection headers
        'padding': '8px 12px',
        'margin': '20px 0 15px 20px',
        'border-radius': '3px'
    }
}


class SectionGenerator:
    """
    Generates consistently styled section headers for Bokeh layouts.
    
    Args:
        styles (dict, optional): Custom styling dictionary. Defaults to GrayShades.
        
    Example:
        generator = SectionGenerator()
        header = generator.getSections("section", "Main Analysis")
        # header is now a styled Div ready for layout use
    """
    
    def __init__(self, styles=None):
        """
        Initialize the section generator with styling.
        
        Args:
            styles (dict, optional): Style dictionary with keys 'section', 'sub_section', 'sub_sub_section'.
                                   Defaults to GrayShades.
        """
        self.styles = styles if styles is not None else GrayShades
        
    def getSections(self, section_type, text):
        """
        Generate a styled section header.
        
        Args:
            section_type (str): Type of section ('section', 'sub_section', 'sub_sub_section')
            text (str): Header text content
            
        Returns:
            Div: Styled Bokeh Div element
            
        Raises:
            ValueError: If section_type is not recognized
        """
        if section_type not in self.styles:
            raise ValueError(f"Unknown section type '{section_type}'. Available: {list(self.styles.keys())}")
            
        return Div(text=text, style=self.styles[section_type]) 
