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
Verify the TR 38.901 Step-8 sub-cluster coupling invariant from an sls_chan_ex H5 dump.

Per TR 38.901 Step 8, the random coupling permutations of the two strongest clusters
must stay within each Table 7.5-5 sub-cluster ray set (R1 = {0-7,18,19}, R2 =
{8-11,16,17}, R3 = {12-15}, 0-based), so every delay sub-cluster keeps its own designed
Table 7.5-3 offset footprint. Regular clusters use a full 20-ray permutation.

For every link and cluster, this script recovers the offset-index permutation from the
per-ray vs per-cluster angles (AoA, AoD, ZOD) in the dump and checks:
  - split (strongest-2) clusters: the permutation maps each R_i onto itself
  - regular clusters: reported as a negative control (a random 20-permutation conforms
    with probability ~3.6e-8, so the expected count is ~0)
Clusters where the offset recovery is ambiguous due to angle wrapping/reflection are
skipped and counted.

Usage:
    python3 util/verify_subcluster_coupling.py <dump.h5>

Exit code 0 if no violations, 1 otherwise.
"""
import sys

import h5py
import numpy as np

OFF = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715,
                0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481,
                1.5195, -1.5195, 2.1551, -2.1551])
R = [set([0, 1, 2, 3, 4, 5, 6, 7, 18, 19]), set([8, 9, 10, 11, 16, 17]), set([12, 13, 14, 15])]


def wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def recover(per_ray: np.ndarray, per_cluster: float) -> np.ndarray | None:
    """Return the offset-index assignment for one cluster, or None if not a clean bijection."""
    delta = wrap(per_ray - per_cluster)
    c = np.max(np.abs(delta)) / 2.1551
    if c < 1e-6:
        return None
    idx = np.array([np.argmin(np.abs(d - c * OFF)) for d in delta])
    if sorted(idx) != list(range(20)):
        return None  # recovery failed (angle wrap/reflection)
    return idx


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    with h5py.File(sys.argv[1], 'r') as f:
        cp = f['clusterParams'][:]
    split_ok = split_bad = reg_conform = reg_total = unrecovered = 0
    for li in range(len(cp)):
        n_cluster = int(cp[li]['nCluster'])
        s2 = {int(x) for x in cp[li]['strongest2clustersIdx']}
        for ang_ray, ang_cl in [('phi_n_m_AoA', 'phi_n_AoA'), ('phi_n_m_AoD', 'phi_n_AoD'),
                                ('theta_n_m_ZOD', 'theta_n_ZOD')]:
            for n in range(n_cluster):
                idx = recover(cp[li][ang_ray][n * 20:(n + 1) * 20], cp[li][ang_cl][n])
                if idx is None:
                    unrecovered += 1
                    continue
                conforms = all(set(idx[list(Ri)]) == Ri for Ri in R)
                if n in s2:
                    if conforms:
                        split_ok += 1
                    else:
                        split_bad += 1
                        print(f"VIOLATION link {li} cluster {n} {ang_ray}: {idx.tolist()}")
                else:
                    reg_total += 1
                    if conforms:
                        reg_conform += 1
    print(f"split clusters conforming: {split_ok}, violations: {split_bad}")
    print(f"regular clusters: {reg_total}, accidentally conforming: {reg_conform} (expect ~0)")
    print(f"unrecoverable (angle wrap): {unrecovered}")
    sys.exit(1 if split_bad else 0)


if __name__ == '__main__':
    main()
