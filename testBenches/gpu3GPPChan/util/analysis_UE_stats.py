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
UE elevation angle CDF analysis from SLS channel H5 files.

Reads linkParams (theta_LOS_ZOD = zenith angle of departure from BS, i.e. elevation-related
angle per link) from the H5 file. Collects elevation angles from all UEs and all sites,
then plots a single CDF curve for all UEs.

Usage:
  python analysis_UE_stats.py <h5_file>
  python analysis_UE_stats.py slsChanData_3sites_192uts_cuMAC_slot0.h5

Output:
  PNG file: UE_stat_<h5_file_basename>.png
  e.g. UE_stat_slsChanData_3sites_192uts_cuMAC_slot0.h5.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _validate_link_topology(n_links: int, n_site: int, n_ut: int) -> None:
    """Fail fast when linkParams length does not match a full n_site x n_ut grid."""
    if n_site <= 0:
        raise ValueError(f"n_site must be positive, got n_site={n_site} (n_links={n_links})")
    if n_ut <= 0:
        raise ValueError(f"n_ut must be positive, got n_ut={n_ut} (n_links={n_links}, n_site={n_site})")
    if n_links % n_site != 0:
        raise ValueError(
            f"n_links={n_links} is not divisible by n_site={n_site} "
            f"(remainder {n_links % n_site}); cannot infer n_ut without dropping links"
        )
    if n_links != n_site * n_ut:
        raise ValueError(
            f"n_links={n_links} != n_site={n_site} * n_ut={n_ut} (= {n_site * n_ut}); inconsistent link grid"
        )


def get_topology(h5_file: h5py.File) -> tuple[Optional[int], Optional[int]]:
    """Read nSite and nUT from topology group; fallback from linkParams size."""
    n_site = None
    n_ut = None
    if "topology" in h5_file:
        topo = h5_file["topology"]
        if "nSite" in topo:
            n_site = int(topo["nSite"][()])
        if "nUT" in topo:
            n_ut = int(topo["nUT"][()])
    if (n_site is None or n_ut is None) and "linkParams" in h5_file:
        n_links = h5_file["linkParams"].shape[0]
        if n_site is not None and n_ut is None:
            if n_site <= 0:
                raise ValueError(f"n_site must be positive, got n_site={n_site} (n_links={n_links})")
            if n_links % n_site != 0:
                raise ValueError(
                    f"n_links={n_links} is not divisible by n_site={n_site} "
                    f"(remainder {n_links % n_site}); cannot infer n_ut without dropping links"
                )
            n_ut = n_links // n_site
        elif n_site is None and n_ut is not None:
            if n_ut <= 0:
                raise ValueError(f"n_ut must be positive, got n_ut={n_ut} (n_links={n_links})")
            if n_links % n_ut != 0:
                raise ValueError(
                    f"n_links={n_links} is not divisible by n_ut={n_ut} "
                    f"(remainder {n_links % n_ut}); cannot infer n_site without dropping links"
                )
            n_site = n_links // n_ut
        elif n_site is None and n_ut is None:
            n_site = 1
            n_ut = n_links
    if n_site is not None and n_ut is not None and "linkParams" in h5_file:
        _validate_link_topology(h5_file["linkParams"].shape[0], n_site, n_ut)
    return n_site, n_ut


def load_ue_elevation_angles(h5_path: Path) -> tuple[int, int, list[np.ndarray]]:
    """
    Load elevation angle (theta_LOS_ZOD) per link and group by UE.
    linkIdx = siteIdx * nUT + utIdx, so for UE utIdx we use links [utIdx, nUT+utIdx, ...].
    Returns (n_ut, list of arrays): for each UE, array of elevation angles in degrees.
    """
    with h5py.File(h5_path, "r") as f:
        if "linkParams" not in f:
            raise ValueError(f"No 'linkParams' dataset in {h5_path}")
        n_site, n_ut = get_topology(f)
        if n_site is None or n_ut is None:
            raise ValueError("Could not determine nSite/nUT from topology or linkParams")
        ds = f["linkParams"]
        n_links = ds.shape[0]
        _validate_link_topology(n_links, n_site, n_ut)
        theta_zod = np.array([ds[i]["theta_LOS_ZOD"] for i in range(n_links)], dtype=np.float64)
    # Group by UE: for ut_idx, links are ut_idx, n_ut+ut_idx, 2*n_ut+ut_idx, ...
    ue_elevations = []
    for ut_idx in range(n_ut):
        indices = [ut_idx + s * n_ut for s in range(n_site)]
        vals = theta_zod[indices]
        vals = vals[np.isfinite(vals)]
        ue_elevations.append(vals)
    return n_site, n_ut, ue_elevations


def plot_ue_elevation_cdf(
    ue_elevations, n_site: int, n_ut: int, out_path: Path, h5_name: str
) -> None:
    """Plot a single CDF of elevation angles across all UEs (elevation angle in degrees)."""
    non_empty = [e for e in ue_elevations if len(e) > 0]
    if not non_empty:
        raise ValueError("No valid elevation angles found")
    all_elev = np.concatenate(non_empty)
    all_elev = 90.0 - all_elev  # Convert zenith angle to elevation from horizontal
    sorted_x = np.sort(all_elev)
    cdf_y = (np.arange(1, len(sorted_x) + 1) / len(sorted_x)) * 100.0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_x, cdf_y, color="blue", linewidth=2, label="All UEs")
    ax.set_xlabel("Elevation angle (in deg)", fontsize=12)
    ax.set_ylabel("CDF (%)", fontsize=12)
    ax.set_title(f"Elevation angle CDF (all {n_ut} UEs, {n_site} sites)\n{h5_name}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(float(np.min(sorted_x)), float(np.max(sorted_x)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CDF of elevation angle (all UEs combined) from SLS channel H5 file."
    )
    parser.add_argument(
        "h5_file",
        type=Path,
        help="Path to SLS channel H5 file (e.g. slsChanData_3sites_192uts_cuMAC_slot0.h5)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: UE_stat_<h5_file_name>.png in current directory)",
    )
    args = parser.parse_args()

    h5_path = args.h5_file.resolve()
    if not h5_path.is_file():
        print(f"Error: file not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    h5_name = h5_path.name
    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = Path.cwd() / f"UE_stat_{h5_name}.png"

    try:
        n_site, n_ut, ue_elevations = load_ue_elevation_angles(h5_path)
    except (ValueError, OSError, KeyError) as e:
        print(f"Error loading H5: {e}", file=sys.stderr)
        sys.exit(1)

    plot_ue_elevation_cdf(ue_elevations, n_site, n_ut, out_path, h5_name)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
