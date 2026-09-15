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
CPU-vs-GPU comparison for the SLS channel model: KS equivalence tests + CDF plots.

Compares two sls_chan_ex H5 dumps (typically one generated with cpu_only_mode: 0
and one with cpu_only_mode: 1, same config otherwise) across the pipeline stages:

  - large-scale: pathloss, shadow fading
  - LSPs: log10(DS), ASA, ASD, ZSA, ZSD, LOS ratio
  - small-scale realization: per-link RMS delay spread from cluster delays/powers
  - final coefficients: per-link CIR power (cell 0)

Every metric gets a two-sample Kolmogorov-Smirnov test (two-proportion z-test
for the binary LOS indicator). CPU and GPU use different RNGs (std::mt19937 vs
cuRAND), so the comparison is statistical, not per-sample. All tests are
expected to accept at alpha = 0.01.

With --plot, additionally saves a one-page CDF comparison (GPU vs CPU) of the
headline statistics.

Usage:
    python3 util/compare_cpu_vs_gpu.py <gpu_dump.h5> <cpu_dump.h5> [--plot [out.png]]

Exit code 0 if all tests pass, 1 otherwise.
"""
import argparse
import sys

import h5py
import numpy as np
from scipy import stats

ALPHA = 0.01

# (metric key, axis label) for the --plot CDF panels
PLOT_PANELS = [
    ('pathloss', 'Pathloss [dB]'),
    ('SF', 'Shadow fading [dB]'),
    ('rms_DS_ns', 'Realized RMS DS [ns]'),
    ('ASA', 'ASA LSP [deg]'),
    ('ZSA', 'ZSA LSP [deg]'),
    ('cir_pow_dB_cell0', 'CIR power cell 0 [dB]'),
]


def load(fn: str) -> dict[str, np.ndarray]:
    with h5py.File(fn, 'r') as f:
        lp = f['linkParams'][:]
        cp = f['clusterParams'][:]
        c = f['cirPerCell/cirCoe_cell0'][:]
    out = {
        'pathloss': lp['pathloss'].astype(float),
        'SF': lp['SF'].astype(float),
        'log10_DS': np.log10(np.maximum(lp['DS'].astype(float), 1e-12)),
        'ASA': lp['ASA'].astype(float),
        'ASD': lp['ASD'].astype(float),
        'ZSA': lp['ZSA'].astype(float),
        'ZSD': lp['ZSD'].astype(float),
        'losInd': lp['losInd'].astype(float),
    }
    # realized RMS delay spread from cluster delays/powers
    rms = []
    for i in range(len(cp)):
        n = int(cp[i]['nCluster'])
        d = cp[i]['delays'][:n].astype(float)
        p = cp[i]['powers'][:n].astype(float)
        if p.sum() <= 0:
            continue
        m = (d * p).sum() / p.sum()
        rms.append(np.sqrt(((d - m) ** 2 * p).sum() / p.sum()))
    out['rms_DS_ns'] = np.array(rms)
    # per-link CIR power, cell 0
    h = c['real'].astype(np.float64) + 1j * c['imag'].astype(np.float64)
    pw = (np.abs(h) ** 2).sum(axis=tuple(range(1, h.ndim)))
    out['cir_pow_dB_cell0'] = 10 * np.log10(pw[pw > 0])
    return out


def run_ks(g: dict[str, np.ndarray], c: dict[str, np.ndarray]) -> int:
    print(f"{'metric':18s} {'GPU mean':>10s} {'GPU med':>9s} {'CPU mean':>10s} {'CPU med':>9s} {'KS p':>7s}")
    fail = 0
    for k in g:
        a, b = g[k], c[k]
        if len(a) == 0 or len(b) == 0:
            raise ValueError(f"{k} has no samples to compare")
        if k == 'losInd':
            # binary indicator: two-proportion z-test instead of KS
            pa, pb = a.mean(), b.mean()
            if pa == pb and pa in (0.0, 1.0):
                p = 1.0
            elif pa in (0.0, 1.0) and pb in (0.0, 1.0):
                p = 0.0
            else:
                pooled = (a.sum() + b.sum()) / (len(a) + len(b))
                se = np.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
                z = (pa - pb) / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
            flag = "  <-- FAIL" if p < ALPHA else ""
            if p < ALPHA:
                fail += 1
            print(f"{'LOS ratio':18s} {pa:10.4f} {'':>9s} {pb:10.4f} {'':>9s} {p:7.3f}{flag}")
            continue
        _, p = stats.ks_2samp(a, b)
        flag = "  <-- FAIL" if p < ALPHA else ""
        if p < ALPHA:
            fail += 1
        print(f"{k:18s} {a.mean():10.3f} {np.median(a):9.3f} {b.mean():10.3f} {np.median(b):9.3f} {p:7.3f}{flag}")
    print(f"\n{'ALL PASS' if fail == 0 else f'{fail} FAILURES'} (two-sample KS, alpha={ALPHA})")
    return fail


def plot_cdfs(g: dict[str, np.ndarray], c: dict[str, np.ndarray], out: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def cdf(ax: plt.Axes, x: np.ndarray, **kw: object) -> None:
        xs = np.sort(x)
        ax.plot(xs, 100 * np.arange(1, len(xs) + 1) / len(xs), **kw)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (key, label) in zip(axes.flat, PLOT_PANELS, strict=False):
        cdf(ax, g[key], color='blue', lw=2.2, label='Simulation GPU')
        cdf(ax, c[key], color='red', lw=2.2, ls='--', label='Simulation CPU')
        ax.set_xlabel(label)
        ax.set_ylabel('CDF (%)')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle('CPU-only vs GPU channel statistics (same drop, independent RNG)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gpu_h5', help='H5 dump from the GPU run (cpu_only_mode: 0)')
    ap.add_argument('cpu_h5', help='H5 dump from the CPU-only run (cpu_only_mode: 1)')
    ap.add_argument('--plot', nargs='?', const='cpu_vs_gpu.png', metavar='out.png',
                    help='also save a one-page CDF comparison figure (default name: cpu_vs_gpu.png)')
    args = ap.parse_args()
    g, c = load(args.gpu_h5), load(args.cpu_h5)
    fail = run_ks(g, c)
    if args.plot:
        plot_cdfs(g, c, args.plot)
    sys.exit(0 if fail == 0 else 1)


if __name__ == '__main__':
    main()
