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

"""pyAerial library - Visualization utilities."""
from typing import Optional
from typing import Tuple
from typing import Union

import cupy as cp  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np


def plot_constellation(
    constellation: Union[np.ndarray, cp.ndarray],
    mod_order: int,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    title: Optional[str] = None,
    show_grid: bool = True,
    marker_size: int = 100,
    font_size: int = 10
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a QAM constellation diagram with bit labels.

    Displays constellation points on an I-Q (In-phase/Quadrature) grid with
    the corresponding bit string labels above each point.

    The constellation array must be indexed by bit pattern integer (MSB first),
    i.e., constellation[i] is the symbol for the bit pattern represented by
    the integer i. For example, with mod_order=2 (QPSK):
    - constellation[0] is the symbol for bits "00"
    - constellation[1] is the symbol for bits "01"
    - constellation[2] is the symbol for bits "10"
    - constellation[3] is the symbol for bits "11"

    This convention is followed by ModulationMapper.constellation.

    Args:
        constellation: Complex constellation points array of shape (2^mod_order,).
            Must be indexed by bit pattern integer (MSB first).
            Can be NumPy or CuPy array.
        mod_order: Modulation order (number of bits per symbol).
            Supported values: 2 (QPSK), 4 (16-QAM), 6 (64-QAM), 8 (256-QAM).
        ax: Optional matplotlib Axes to plot on. If None, a new figure is created.
        figsize: Figure size as (width, height) in inches. Only used if ax is None.
        title: Optional title for the plot. If None, a default title is generated.
        show_grid: Whether to show grid lines.
        marker_size: Size of constellation point markers.
        font_size: Font size for bit string labels.

    Returns:
        Tuple of (figure, axes) matplotlib objects.

    Example:
        >>> from aerial.phy5g.algorithms import ModulationMapper
        >>> from aerial.util.visualization import plot_constellation
        >>> mapper = ModulationMapper(mod_order=2)  # QPSK
        >>> fig, ax = plot_constellation(mapper.constellation, mod_order=2)
        >>> plt.show()
    """
    # Convert CuPy to NumPy if needed
    if isinstance(constellation, cp.ndarray):
        constellation = constellation.get()

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    # Extract I and Q components
    i_vals = np.real(constellation)
    q_vals = np.imag(constellation)

    # Plot constellation points
    ax.scatter(i_vals, q_vals, s=marker_size, c='blue', marker='o', zorder=3)

    # Add bit string labels above each point
    num_symbols = len(constellation)
    axis_range = max(np.max(np.abs(i_vals)), np.max(np.abs(q_vals))) * 1.3

    for idx in range(num_symbols):
        # Convert index to bit string (MSB first)
        bit_string = format(idx, f'0{mod_order}b')
        ax.annotate(
            bit_string,
            (i_vals[idx], q_vals[idx]),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=font_size,
            fontfamily='monospace'
        )

    # Configure axes
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(-axis_range, axis_range)
    ax.set_ylim(-axis_range, axis_range)
    ax.set_aspect('equal')
    ax.set_xlabel('In-phase (I)')
    ax.set_ylabel('Quadrature (Q)')

    if show_grid:
        ax.grid(True, alpha=0.3)

    # Set title
    if title is None:
        modulation_names = {2: 'QPSK', 4: '16-QAM', 6: '64-QAM', 8: '256-QAM'}
        mod_name = modulation_names.get(mod_order, f'{2**mod_order}-QAM')
        title = f'{mod_name} Constellation (Gray Coded)'
    ax.set_title(title)

    return fig, ax
