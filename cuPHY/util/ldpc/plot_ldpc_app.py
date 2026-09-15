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

"""plot_ldpc_app.py  ref_h5  dut_h5  [--cb CB] [--out PREFIX]
Plot LLR_initial + APP_history.
Scatter grids: <PREFIX>_{ref,dut,reldiff,absdiff}_cb<N>.png
CDF grids:     <PREFIX>_{ref,dut,reldiff,absdiff}_cdf_cb<N>.png
               (all CDFs use sign(x)*log2|x| on x-axis)
ref: blue   DUT: black
"""
import argparse
import math
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_params(h5_path: str) -> dict[str, float]:
    """Read the structural parameters describing an APP history file.

    Prefers the compound /params dataset written by the MATLAB test-vector flow.
    When it is absent (a DUT file from write_dut_h5), the shape of /APP_history
    supplies maxItr, C and K_prime, and the individually stored scalars BGN, Zc,
    Kb, mb and F are picked up when present.

    Args:
        h5_path: Path to a REF (test-vector) or DUT HDF5 file.

    Returns:
        Field name to value. Always contains maxItr, C and K_prime; contains
        BGN, Zc, Kb, mb and F when the file carries them. Values are int or
        float depending on the field.

    Raises:
        OSError: If the file cannot be opened.
        KeyError: If neither /params nor /APP_history is present.

    Examples:
        >>> load_params('tv.h5')['maxItr']  # doctest: +SKIP
        20
    """
    # Fall back to shape-derived defaults when /params is absent (DUT h5).
    with h5py.File(h5_path, 'r') as f:
        if 'params' in f:
            p = f['params'][()]
            return {k: p[k].item() for k in p.dtype.names}
        C, maxItr, N = f['APP_history'].shape
        out = {'maxItr': maxItr, 'C': C, 'K_prime': N}
        # write_dut_h5 stashes structural scalars individually (no compound /params);
        # they enable the core/ext column regions, clipping and boundary markers.
        for k in ('BGN', 'Zc', 'Kb', 'mb', 'F'):
            if k in f:
                out[k] = int(f[k][()])
        # With the structure known, K_prime = info bits excluding filler (better ylim);
        # otherwise K_prime stays N and the ylim is computed from the full row.
        if 'Kb' in out and 'Zc' in out:
            out['K_prime'] = out['Kb'] * out['Zc'] - out.get('F', 0)
        return out


def _save_grid(slices: list[np.ndarray], titles: list[str], suptitle: str,
               ylabel: str, color: str, fname: str,
               axhline: bool = False,
               ylim_init: float | None = None,
               ylim_iter: float | None = None,
               vline_x: float | None = None) -> None:
    n     = len(slices)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3, nrows * 3),
                             squeeze=False)
    axes = axes.flatten()
    fig.suptitle(suptitle, fontsize=10)
    for i, (data, title) in enumerate(zip(slices, titles)):
        ax = axes[i]
        ax.plot(data, '.', color=color, markersize=1, linestyle='None')
        if axhline:
            ax.axhline(0, color='gray', lw=0.4, ls='--')
        if vline_x is not None:
            ax.axvline(vline_x, color='red', lw=0.5, ls='--')
        ylim = ylim_init if i == 0 else ylim_iter
        if ylim is not None:
            ax.set_ylim(-ylim, ylim)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
    for i in range(n, nrows * ncols):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'Saved: {fname}')


def _compute_cdf_xy(data: np.ndarray,
                    signed_log2_eps: float | None
                    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    rel = np.asarray(data, dtype=np.float64).ravel()
    if signed_log2_eps is not None:
        zero = rel == 0
        mag = np.maximum(np.abs(rel), signed_log2_eps)
        flat = np.sign(rel) * np.log2(mag)
        flat[zero] = 0.0
    else:
        flat = rel
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None, None
    xs = np.sort(flat)
    ys = np.arange(1, flat.size + 1, dtype=np.float64) / flat.size
    return xs, ys


def _save_cdf_grid(slices: list[np.ndarray], titles: list[str], suptitle: str,
                   xlabel: str, color: str, fname: str,
                   xlim_init: float | None = None,
                   xlim_iter: float | None = None,
                   signed_log2_eps: float | None = None,
                   group_size: int = 4,
                   llr_stands_alone: bool = False) -> None:
    # Layout:
    #   llr_stands_alone=True : slices[0] = LLR_initial -> own subplot;
    #                           slices[1:] grouped `group_size` per subplot.
    #   llr_stands_alone=False: all slices grouped `group_size` per subplot.
    if llr_stands_alone and len(slices) > 0:
        head_slice, head_title = slices[0], titles[0]
        rest_slices, rest_titles = slices[1:], titles[1:]
    else:
        head_slice = None
        rest_slices, rest_titles = slices, titles

    groups = []
    for start in range(0, len(rest_slices), group_size):
        end = min(start + group_size, len(rest_slices))
        groups.append((rest_slices[start:end], rest_titles[start:end], start, end - 1))

    n = (1 if head_slice is not None else 0) + len(groups)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3, nrows * 3),
                             squeeze=False)
    axes = axes.flatten()
    fig.suptitle(suptitle, fontsize=10)

    idx = 0
    if head_slice is not None:
        ax = axes[idx]
        xs, ys = _compute_cdf_xy(head_slice, signed_log2_eps)
        if xs is not None:
            ax.plot(xs, ys, color=color, linewidth=1)
        xlim = xlim_init
        if xlim is None and signed_log2_eps is not None and xs is not None:
            xlim = max(1.0, float(np.max(np.abs(xs))))
        if xlim is not None:
            ax.set_xlim(-xlim, xlim)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel('CDF', fontsize=7)
        ax.set_title(head_title, fontsize=7)
        ax.tick_params(labelsize=6)
        idx += 1

    curve_colors = plt.cm.viridis(np.linspace(0.15, 0.85, group_size))

    for g_slices, g_titles, g_start, g_end in groups:
        ax = axes[idx]
        max_abs = 0.0
        for c, (data, label) in enumerate(zip(g_slices, g_titles)):
            xs, ys = _compute_cdf_xy(data, signed_log2_eps)
            if xs is None:
                continue
            ax.plot(xs, ys, color=curve_colors[c], linewidth=1, label=label)
            max_abs = max(max_abs, float(np.max(np.abs(xs))))
        xlim = xlim_iter
        if xlim is None and signed_log2_eps is not None:
            xlim = max(1.0, max_abs)
        if xlim is not None:
            ax.set_xlim(-xlim, xlim)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel('CDF', fontsize=7)
        ax.set_title(f'iter {g_start}..{g_end}', fontsize=7)
        ax.legend(fontsize=5, loc='lower right')
        ax.tick_params(labelsize=6)
        idx += 1

    for i in range(n, nrows * ncols):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f'Saved: {fname}')


def plot_cb(cb: int | None, ref_h5: str, dut_h5: str, params: dict[str, float],
            out_prefix: str) -> None:
    """Write the per-iteration APP comparison figures for one or all codeblocks.

    Emits value grids, CDFs and relative/absolute difference plots as PNGs named
    ``<out_prefix>_<kind>_cb<n>.png``. REF and DUT may carry different iteration
    counts; the lower of the two bounds every APP loop.

    Args:
        cb: Codeblock index to plot, or None to plot every codeblock.
        ref_h5: Reference (test-vector) HDF5 path.
        dut_h5: DUT HDF5 path written by write_dut_h5.
        params: Structural parameters from :func:`load_params`.
        out_prefix: Prefix for the generated PNG filenames.

    Returns:
        None. Figures are written to disk and their paths printed.

    Raises:
        OSError: If either file cannot be opened.
        KeyError: If an expected dataset is missing.
        IndexError: If cb is outside the codeblock range.

    Examples:
        >>> plot_cb(0, 'tv.h5', 'ldpc_dut.h5', p, 'ldpc')  # doctest: +SKIP
    """
    C       = params['C']
    K_prime = params['K_prime']   # info-bit count; excludes filler saturation from ylim

    # End of "core" variable nodes: (Kb + 4) * Zc — matches N_core_var in
    # ldpc_interm_check.cpp. Kernels don't update extension parity columns
    # during iters, so any APP comparison/visualization beyond this is noise.
    # Only available when /params is present on the REF h5; otherwise skip.
    if 'BGN' in params and 'Zc' in params:
        Kb         = 22 if params['BGN'] == 1 else 10
        N_core_var = (Kb + 4) * params['Zc']
    else:
        N_core_var = None

    with h5py.File(ref_h5, 'r') as rf, h5py.File(dut_h5, 'r') as df:
        llr_input_ref = rf['inputLLR'][()]      # [C, N]
        llr_input_dut = df['inputLLR'][()] if 'inputLLR' in df else llr_input_ref
        # REF and DUT can carry different iteration counts; read both truncated to
        # the shorter one so every use below is bounded without repeating the limit.
        max_itr       = min(rf['APP_history'].shape[1], df['APP_history'].shape[1])
        APP_ref       = rf['APP_history'][:, :max_itr]   # [C, max_itr, N]
        APP_dut       = df['APP_history'][:, :max_itr]  # [C, max_itr, N]

    if cb is not None and not (0 <= cb < C):
        raise SystemExit(f"--cb {cb} out of range: data has {C} codeblock(s) (valid 0..{C - 1}).")
    cbs = range(C) if cb is None else [cb]

    for cw in cbs:
        llr_app_titles = ['LLR_initial'] + [f'APP iter {i}' for i in range(max_itr)]
        app_titles     = [f'APP iter {i}' for i in range(max_itr)]

        # [:, :K_prime]: info bits only (cols K_prime..N-1 are filler/parity
        # with large saturated values that would dominate the scale)
        ylim_init_ref = max(1.0, float(np.max(np.abs(llr_input_ref[cw, :K_prime]))))
        ylim_init_dut = max(1.0, float(np.max(np.abs(llr_input_dut[cw, :K_prime]))))
        ylim_iter_ref = max(1.0, float(np.max(np.abs(APP_ref[cw, :, :K_prime]))))
        ylim_iter_dut = max(1.0, float(np.max(np.abs(APP_dut[cw, :, :K_prime]))))

        ref_slices = [llr_input_ref[cw]] + [APP_ref[cw, i] for i in range(max_itr)]
        dut_slices = [llr_input_dut[cw]] + [APP_dut[cw, i] for i in range(max_itr)]

        _save_grid(
            ref_slices, llr_app_titles, f'CB {cw} — ref', 'ref', 'blue',
            f'{out_prefix}_ref_cb{cw}.png',
            ylim_init=ylim_init_ref, ylim_iter=ylim_iter_ref,
            vline_x=N_core_var)

        _save_grid(
            dut_slices, llr_app_titles, f'CB {cw} — DUT', 'dut', 'black',
            f'{out_prefix}_dut_cb{cw}.png',
            ylim_init=ylim_init_dut, ylim_iter=ylim_iter_dut,
            vline_x=N_core_var)

        EPS = 1.0 / 1024
        # CDFs are restricted to core columns [:N_core_var] = (Kb+4)*Zc, since
        # extension parity columns are not updated by the decoder per iter and
        # would skew the CDF tails. Scatter grids still show the full N with a
        # red dashed marker at the core boundary.
        def _core(arr: np.ndarray) -> np.ndarray:
            return arr[..., :N_core_var] if N_core_var is not None else arr
        core_tag = ' [core clms]' if N_core_var is not None else ''

        ref_cdf_slices = [_core(s) for s in ref_slices]
        dut_cdf_slices = [_core(s) for s in dut_slices]
        _save_cdf_grid(
            ref_cdf_slices, llr_app_titles,
            f'CB {cw} — sign(ref)*log2|ref| CDF{core_tag}', 'sign(ref)*log2|ref|', 'blue',
            f'{out_prefix}_ref_cdf_cb{cw}.png',
            signed_log2_eps=EPS, llr_stands_alone=True)
        _save_cdf_grid(
            dut_cdf_slices, llr_app_titles,
            f'CB {cw} — sign(dut)*log2|dut| CDF{core_tag}', 'sign(dut)*log2|dut|', 'black',
            f'{out_prefix}_dut_cdf_cb{cw}.png',
            signed_log2_eps=EPS, llr_stands_alone=True)

        denom = [np.maximum(np.maximum(np.abs(APP_ref[cw, i]), np.abs(APP_dut[cw, i])), EPS)
                 for i in range(max_itr)]
        rel_signed = [(APP_ref[cw, i] - APP_dut[cw, i]) / d for i, d in enumerate(denom)]
        reldiff = [np.abs(r) for r in rel_signed]
        ylim_rel = max(1e-4, float(np.max(np.array(reldiff)[:, :K_prime])))

        _save_grid(
            reldiff, app_titles, f'CB {cw} — |ref-dut|/max(|ref|,|dut|,EPS)', 'rel_err', 'purple',
            f'{out_prefix}_reldiff_cb{cw}.png',
            ylim_init=ylim_rel, ylim_iter=ylim_rel,
            vline_x=N_core_var)
        _save_cdf_grid(
            [_core(r) for r in rel_signed], app_titles,
            f'CB {cw} — sign(rel)*log2|rel| CDF{core_tag}', 'sign(rel)*log2|rel|', 'purple',
            f'{out_prefix}_reldiff_cdf_cb{cw}.png',
            signed_log2_eps=EPS)

        absdiff = [APP_ref[cw, i] - APP_dut[cw, i] for i in range(max_itr)]
        ylim_abs = max(1e-4, float(np.max(np.abs(np.array(absdiff)[:, :K_prime]))))
        _save_grid(
            absdiff, app_titles, f'CB {cw} — ref-dut', 'ref-dut', 'black',
            f'{out_prefix}_absdiff_cb{cw}.png', axhline=True,
            ylim_init=ylim_abs, ylim_iter=ylim_abs,
            vline_x=N_core_var)
        _save_cdf_grid(
            [_core(a) for a in absdiff], app_titles,
            f'CB {cw} — sign(ref-dut)*log2|ref-dut| CDF{core_tag}', 'sign(ref-dut)*log2|ref-dut|', 'black',
            f'{out_prefix}_absdiff_cdf_cb{cw}.png',
            signed_log2_eps=EPS)


def main() -> None:
    """Parse the command line and plot the requested codeblock comparison.

    Reads the REF and DUT paths, and plots one or all codeblocks.

    Returns:
        None.

    Raises:
        SystemExit: Raised by argparse on --help or invalid arguments.
        OSError: If either HDF5 file cannot be opened.

    Examples:
        Plot every codeblock::

            python3 plot_ldpc_app.py tv.h5 ldpc_dut.h5 --out ldpc

        Plot one codeblock::

            python3 plot_ldpc_app.py tv.h5 ldpc_dut.h5 --cb 0
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('ref_h5')
    ap.add_argument('dut_h5')
    ap.add_argument('--cb',       type=int, default=None)
    ap.add_argument('--out',      default='ldpc')
    args = ap.parse_args()

    params = load_params(args.ref_h5)
    plot_cb(args.cb, args.ref_h5, args.dut_h5, params, args.out)


if __name__ == '__main__':
    main()
