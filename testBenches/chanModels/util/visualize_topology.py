# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Visualize network topology from YAML or H5 file."""

import yaml
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from matplotlib.patches import RegularPolygon
from collections import defaultdict


def parse_h5_topology(h5_file_path):
    """Parse topology data from H5 file"""
    with h5py.File(h5_file_path, 'r') as f:
        # Parse topology parameters
        if 'topology' in f:
            topo_group = f['topology']
            topology = {
                'ISD': float(topo_group['ISD'][()]),
                'nSite': int(topo_group['nSite'][()]),
                'nSector': int(topo_group['nSector'][()]),
                'nUT': int(topo_group['nUT'][()]),
                'bsHeight': float(topo_group['bsHeight'][()]),
                'minBsUeDist2d': float(topo_group['minBsUeDist2d'][()]),
                'maxBsUeDist2dIndoor': float(topo_group['maxBsUeDist2dIndoor'][()]),
                'indoorUtPercent': float(topo_group['indoorUtPercent'][()])
            }
            
            # Parse cell parameters (base stations)
            base_stations = []
            if 'cellParams' in topo_group:
                cell_dataset = topo_group['cellParams']
                for record in cell_dataset:
                    # Handle nested location structure
                    if hasattr(record['loc'], 'dtype') and record['loc'].dtype.names:
                        # Structured array
                        loc = record['loc']
                        location = {'x': float(loc['x']), 'y': float(loc['y']), 'z': float(loc['z'])}
                    else:
                        # Simple array [x, y, z]
                        loc = record['loc']
                        location = {'x': float(loc[0]), 'y': float(loc[1]), 'z': float(loc[2])}
                    
                    # Sector orientation (prefer H5 antPanelOrientation if present)
                    cid = int(record['cid'])
                    site_id = int(record['siteId'])
                    if 'antPanelOrientation' in record.dtype.names:
                        orientation = float(record['antPanelOrientation'][1])
                    else:
                        # Fallback: match the C++ model convention in sls_chan_topology.cu:
                        # sectorOrien = {30, 150, 270} for 3 sectors (boresights), i.e. 30 + 120*k.
                        n_site = int(topo_group['nSite'][()]) if 'nSite' in topo_group else 1
                        n_sector = int(topo_group['nSector'][()]) if 'nSector' in topo_group else 3
                        n_sector_per_site = max(1, int(round(n_sector / max(1, n_site))))
                        sector_id = cid % n_sector_per_site
                        step = 360.0 / float(n_sector_per_site)
                        orientation = 30.0 + sector_id * step
                    
                    base_stations.append({
                        'cid': cid,
                        'siteId': site_id,
                        'location': location,
                        'antPanelIdx': int(record['antPanelIdx']),
                        'antPanelOrientation': [0, orientation]  # [tilt, azimuth]
                    })
            
            # Parse UT parameters (user equipment)
            user_equipment = []
            if 'utParams' in topo_group:
                ut_group = topo_group['utParams']
                # Read arrays from the group
                uids = ut_group['uid'][:]
                locs_x = ut_group['loc_x'][:]
                locs_y = ut_group['loc_y'][:]
                locs_z = ut_group['loc_z'][:]
                outdoor_inds = ut_group['outdoor_ind'][:]
                ant_panel_idxs = ut_group['antPanelIdx'][:]
                
                for i in range(len(uids)):
                    user_equipment.append({
                        'uid': int(uids[i]),
                        'location': {
                            'x': float(locs_x[i]),
                            'y': float(locs_y[i]),
                            'z': float(locs_z[i])
                        },
                        'outdoor_ind': int(outdoor_inds[i]),
                        'antPanelIdx': int(ant_panel_idxs[i])
                    })

            # Parse ST parameters (sensing targets)
            sensing_targets = []
            if 'stParams' in topo_group:
                st_group = topo_group['stParams']
                sids = st_group['sid'][:]
                st_x = st_group['loc_x'][:]
                st_y = st_group['loc_y'][:]
                st_z = st_group['loc_z'][:]
                target_type = st_group['target_type'][:] if 'target_type' in st_group else None

                for i in range(len(sids)):
                    sensing_targets.append({
                        'sid': int(sids[i]),
                        'location': {'x': float(st_x[i]), 'y': float(st_y[i]), 'z': float(st_z[i])},
                        'target_type': int(target_type[i]) if target_type is not None else None,
                    })
        else:
            raise ValueError("No topology data found in H5 file")
    
    return {
        'topology': topology,
        'base_stations': base_stations,
        'user_equipment': user_equipment,
        'sensing_targets': sensing_targets
    }

def visualize_topology(
    input_file: str | Path,
    ue_label: bool = False,
    cell_label: bool = False,
) -> None:
    """Visualize network topology and save the plot as a PNG file.

    Reads base-station, UE, and sensing-target positions from a YAML or
    H5 topology file, draws hexagonal cells with sector boundaries, and
    saves the result next to the input file with a ``.png`` extension.

    Args:
        input_file: Path to the topology file. Supported formats are
            ``.yaml`` / ``.yml`` and ``.h5`` / ``.hdf5``.
        ue_label: If ``True``, annotate each UE marker with its index.
            Defaults to ``False``.
        cell_label: If ``True``, annotate each sector with its cell ID.
            Defaults to ``False``.

    Returns:
        None

    Raises:
        FileNotFoundError: If *input_file* does not exist.
        ValueError: If the file format is unsupported or the topology
            data is missing / malformed.
    """
    file_path = Path(input_file)
    
    # Determine file type and parse accordingly
    if file_path.suffix.lower() in ['.yaml', '.yml']:
        # Read YAML file
        with open(input_file, 'r') as f:
            data = yaml.safe_load(f)
    elif file_path.suffix.lower() in ['.h5', '.hdf5']:
        # Parse H5 file
        data = parse_h5_topology(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .yaml, .yml, .h5, .hdf5")

    # Extract data
    topology = data['topology']
    base_stations = data['base_stations']
    user_equipment = data['user_equipment']
    sensing_targets = data.get('sensing_targets', [])

    # Guard against empty / malformed topology before doing any geometry.
    if not topology:
        raise ValueError("Empty topology: missing 'topology' section")
    isd = topology.get('ISD')
    if isd is None:
        raise ValueError("Empty topology: missing 'ISD'")
    if not base_stations and not user_equipment and not sensing_targets:
        raise ValueError("Empty topology: no BS/UE/ST entries to visualize")

    # Create figure
    _fig, ax = plt.subplots(figsize=(12, 8))

    # Group all BSs by unique (x, y) location (all sites)
    loc_to_cids = defaultdict(list)
    loc_to_orientations = defaultdict(list)
    for bs in base_stations:
        x, y = bs['location']['x'], bs['location']['y']
        loc = (round(x, 6), round(y, 6))  # rounding to avoid float precision issues
        loc_to_cids[loc].append(bs['cid'])
        loc_to_orientations[loc].append(float(bs['antPanelOrientation'][1]))
    hex_side = isd / np.sqrt(3)

    # Calculate adaptive marker sizes based on number of UTs
    n_ut = len(user_equipment)
    base_size = 50  # marker size when n_ut == 1 (inverse scaling)
    min_size = 10   # minimum size to ensure visibility
    
    # Inverse scaling for all n_ut:
    # - n_ut == 0: use base_size
    # - n_ut >= 1: marker_size = base_size / n_ut (clamped to min_size)
    marker_size = max(min_size, base_size / max(n_ut, 1))
    
    # BS marker size should be larger than UE markers
    bs_marker_size = max(30, min(100, marker_size * 1.5))  # 150% of UE marker size, with min 30 and max 100

    # Use blue for all BSs and boundaries
    bs_color = 'tab:blue'

    for loc, cids in loc_to_cids.items():
        x, y = loc
        orientations_deg = sorted(set(loc_to_orientations.get(loc, [])))
        
        # Plot BS marker, assuming at least one BS at origin
        ax.scatter(x, y, c=bs_color, marker='^', s=bs_marker_size, label="BS" if x == 0 and y == 0 else "")
        # Draw hexagon (solid blue line)
        hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_side,
                                orientation=np.pi/6, edgecolor=bs_color, facecolor='none', lw=2, zorder=2)
        ax.add_patch(hexagon)
        # Draw sector boundaries using each sector boresight (±60° around boresight).
        # This matches the dropping logic in dropCoordinateInCell().
        line_length = hex_side / 2 * np.sqrt(3)
        for bore_deg in orientations_deg:
            for delta in (-60.0, 60.0):
                ang = (bore_deg + delta) * np.pi / 180.0
                x_end = x + line_length * np.cos(ang)
                y_end = y + line_length * np.sin(ang)
                ax.plot([x, x_end], [y, y_end], linestyle=':', color=bs_color, lw=1.5, zorder=3)

    if cell_label:
        for bs in base_stations:
            bx, by = bs['location']['x'], bs['location']['y']
            azimuth = float(bs['antPanelOrientation'][1])
            offset_r = hex_side * 0.25
            ox = offset_r * np.cos(np.radians(azimuth))
            oy = offset_r * np.sin(np.radians(azimuth))
            ax.annotate(str(bs['cid']),
                        (bx + ox, by + oy),
                        fontsize=7, fontweight='bold', color=bs_color,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=bs_color, alpha=0.7))

    # Plot UEs
    ue_x = [ue['location']['x'] for ue in user_equipment]
    ue_y = [ue['location']['y'] for ue in user_equipment]
    ue_outdoor = [ue['outdoor_ind'] for ue in user_equipment]
    
    # Plot outdoor and indoor UEs with different markers
    outdoor_mask = np.array(ue_outdoor) == 1
    indoor_mask = ~outdoor_mask
    
    ax.scatter(np.array(ue_x)[outdoor_mask], np.array(ue_y)[outdoor_mask],
              c='blue', marker='o', s=marker_size, label='Outdoor UE')
    ax.scatter(np.array(ue_x)[indoor_mask], np.array(ue_y)[indoor_mask],
              c='red', marker='s', s=marker_size, label='Indoor UE')

    if ue_label:
        print(f"Note: Labeling {n_ut} UEs -- readability of figure may be impacted")
        label_fontsize = max(4, min(8, 120 / max(n_ut, 1)))
        for ue in user_equipment:
            ax.annotate(str(ue['uid']),
                        (ue['location']['x'], ue['location']['y']),
                        fontsize=label_fontsize, ha='left', va='bottom',
                        xytext=(2, 2), textcoords='offset points')

    # Plot STs
    st_x = [st['location']['x'] for st in sensing_targets]
    st_y = [st['location']['y'] for st in sensing_targets]
    if len(st_x) > 0:
        ax.scatter(np.array(st_x), np.array(st_y),
                   c='tab:green', marker='x', s=max(20, marker_size * 1.2), label='ST')
        # Print STs for direct comparison with SHOW_TRAJECTORY logs
        def _ang_deg(xv: float, yv: float) -> float:
            return (np.degrees(np.arctan2(yv, xv)) + 360.0) % 360.0

        def _ang_diff_deg(a: float, b: float) -> float:
            return ((a - b + 180.0) % 360.0) - 180.0

        # Use site at (0,0) sector boresights (single-site visualization case)
        origin_bores = sorted(set(loc_to_orientations.get((0.0, 0.0), [])))
        for st in sensing_targets:
            loc = st["location"]
            ang = _ang_deg(loc["x"], loc["y"])
            sector = None
            if origin_bores:
                # pick first boresight whose wedge contains the point
                for b in origin_bores:
                    if abs(_ang_diff_deg(ang, b)) <= 60.0:
                        sector = b
                        break
            extra = f" angle={ang:.1f}deg sector_bore={sector}" if sector is not None else f" angle={ang:.1f}deg"
            print(f"ST sid={st['sid']} loc=({loc['x']:.3f}, {loc['y']:.3f}, {loc['z']:.3f}){extra}")

    # Add labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Network Topology Visualization ({file_path.name})')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Collect all x/y positions from both UEs and BSs
    bs_x = [loc[0] for loc in loc_to_cids.keys()]
    bs_y = [loc[1] for loc in loc_to_cids.keys()]
    if not (ue_x or st_x or bs_x):
        raise ValueError("Empty topology: no points available to compute axis limits")
    all_x = np.array(ue_x + st_x + bs_x)
    all_y = np.array(ue_y + st_y + bs_y)
    # Account for hexagon radius (hex_side) and a small padding
    padding = hex_side * 0.05
    ax.set_xlim(all_x.min() - hex_side - padding, all_x.max() + hex_side + padding)
    ax.set_ylim(all_y.min() - hex_side - padding, all_y.max() + hex_side + padding)

    # Save the plot
    input_path = Path(input_file)
    output_file = input_path.with_suffix('.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Topology visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize network topology from YAML or H5 file')
    parser.add_argument('input_file', help='Path to the input file containing topology data (.yaml, .yml, .h5, .hdf5)')
    parser.add_argument('--ue_label', action='store_true', help='Show UE index labels on the plot')
    parser.add_argument('--cell_label', action='store_true', help='Show sector/cell index labels on the plot')
    args = parser.parse_args()
    
    try:
        visualize_topology(args.input_file, ue_label=args.ue_label, cell_label=args.cell_label)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main() 