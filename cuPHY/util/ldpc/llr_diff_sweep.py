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

# TV-free per-iteration APP (LLR) diff sweep: compare a reference LDPC decoder
# (default algo40, the generic box-plus reference) against a list of DUT algos
# across the full Z x p grid for BG1 and BG2, using self-generated
# inputs (random codeword + AWGN, no PUSCH rate matching).
#
# Drives `cuphy_ex_ldpc --llr_diff <ref,dut1,...>` once per (BG, Z, p): the tool
# generates one input, captures the reference APP history, and compares each DUT
# against it. A DUT that cannot take the config reports N/A (not a failure).
#
# BUILD PREREQUISITE: a decoder's ET-free dump twin must be compiled in, or its
# capture comes back empty. Build with -DCUPHY_LDPC_SPLIT_DUMP_KERNELS=ON (OFF by
# default). Which decoders carry a twin is listed in ldpc2_decoder_bands.yaml.
# If a capture is all zeros the tool aborts.
#
# Usage:
#   python3 llr_diff_sweep.py --algos 35,41
#   python3 llr_diff_sweep.py --algos 35 --bg 1 --z 384 --p 4-9
#   python3 llr_diff_sweep.py --algos 41 --ref 40 --snr 12 --iters 10
#   python3 llr_diff_sweep.py --algos 35 --csv sweep.csv   # record per-cell metrics
#
# Exit code: 0 if no FAIL, ERR or NO-DUMP (N/A and REF-N/A don't count), 1 otherwise.

import argparse
import contextlib
import math
import os
import re
import subprocess
import sys
import time
from typing import TextIO

from ldpc_bench_common import resolve_bin

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# The 51 standard NR lifting sizes (ldpc_decode_test_vec.cpp).
Z_LIFTINGS = [
      2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
     12,  13,  14,  15,  16,  18,  20,  22,  24,  26,
     28,  30,  32,  36,  40,  44,  48,  52,  56,  60,
     64,  72,  80,  88,  96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 288, 320, 352,
    384,
]
P_MAX = {1: 46, 2: 42}   # max parity nodes per base graph
P_MIN = 4


def parse_int_csv(s):
    return [int(t) for t in s.split(",") if t.strip() != ""]


def parse_z_spec(spec):
    if spec == "all":
        return list(Z_LIFTINGS)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            lo, hi = (int(x) for x in tok.split("-"))
            out += [z for z in Z_LIFTINGS if lo <= z <= hi]
        elif tok:
            out.append(int(tok))
    return out


def parse_p_spec(spec, bg):
    if spec == "all":
        return list(range(P_MIN, P_MAX[bg] + 1))
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            lo, hi = (int(x) for x in tok.split("-"))
            out += list(range(lo, hi + 1))
        elif tok:
            out.append(int(tok))
    # clamp to the base graph's legal parity range
    return [p for p in out if P_MIN <= p <= P_MAX[bg]]


def code_rate(bg, p):
    """5G NR LDPC rate = Kb/(Kb+p-2): Kb info cols (22 BG1, 10 BG2), 2 punctured,
    p parity cols. Matches N = Z*(Kb+p-2) in ldpc_decode_test_vec_gen.cpp."""
    Kb = 22 if bg == 1 else 10
    return Kb / (Kb + p - 2)


def shannon_snr_db(bg, p, bits_per_sym=2, margin=0.0):
    """Es/N0 (dB) at the complex-AWGN Shannon limit for rate code_rate(bg,p) and
    QPSK (2 b/sym), plus margin. -S in this tool is Es/N0 (noise_var=10^(-SNR/10))."""
    R = code_rate(bg, p)
    return 10.0 * math.log10(2.0 ** (R * bits_per_sym) - 1.0) + margin


# Type aliases for run_all_z's return value.
CellResult = tuple[str, float | None, float | None]   # (status, worst_p99err, min_snr)
PerZ = dict[int, dict[tuple[int, int], CellResult]]    # {z: {(dut, p): CellResult}}


def run_all_z(
    binpath: str,
    bg: int,
    zlist: list[int],
    plist: list[int],
    ref: int,
    duts: list[int],
    iters: int,
    snr: float | None,
    p99err: float | None,
    snr_thr: float | None,
    rawlog: TextIO | None = None,
    snr_list: list[float] | None = None,
) -> tuple[PerZ, float | None, float | None]:
    """Run the whole ``Z x p`` grid of one base graph in a single decoder process.

    Issues one ``cuphy_ex_ldpc --llr_diff_zrange`` invocation so the CUDA context
    and ~60 MB module load are paid once for the entire sweep, not once per Z. The
    per-cell verdict line format is unchanged (no C++ API change); Z is recovered
    from each cell's ``llr_diff (TV-free): ... Z=<z> ...`` header, which precedes
    that cell's check_APP table and verdict.

    Args:
        binpath: Path to the ``cuphy_ex_ldpc`` example binary.
        bg: Base graph (1 or 2).
        zlist: Lifting sizes (Z) to sweep; ``zlist[0]`` also seeds the binary's ``-Z``.
        plist: Parity-node counts (p) to sweep.
        ref: Reference algorithm id (e.g. 40, the box-plus reference).
        duts: DUT algorithm ids compared against the reference.
        iters: LDPC decode iterations (binary ``-n``).
        snr: Scalar AWGN SNR dB seeding ``-S``; ``None`` falls back to the first
            entry of ``snr_list`` (which, when given, drives the per-p schedule).
        p99err: Override for the p99err_rms gate, or ``None`` to use the binary's
            built-in gate.
        snr_thr: Override for the per-iteration SNR(dB) gate, or ``None`` for the
            binary's built-in gate.
        rawlog: Open writable text stream to append the raw per-grid stdout+stderr
            to, or ``None`` to skip raw logging.
        snr_list: Per-p AWGN SNR schedule (dB) passed as ``--llr_diff_snrs``, or
            ``None`` for a flat ``snr``.

    Returns:
        A 3-tuple ``(per_z, gate_p99, gate_snr)``. ``per_z`` maps
        ``z -> {(dut, p): (status, worst_p99err, min_snr)}`` where ``status`` is one
        of ``"PASS"``, ``"FAIL"``, ``"N/A"``, ``"NO-DUMP"``, ``"REF-N/A"``, ``"ERR"``. The two
        gates are the thresholds the binary echoed (``float``), or ``None`` if not
        found in the output.

    Raises:
        OSError: If ``subprocess.run`` cannot launch ``binpath`` (e.g. it does not
            exist or is not executable; callers usually validate it first via
            ``resolve_bin``).

    Examples:
        >>> per_z, gp, gs = run_all_z(  # doctest: +SKIP
        ...     "build/cuPHY/examples/error_correction/cuphy_ex_ldpc",
        ...     1, [384, 352], [4, 5, 6], ref=40, duts=[55], iters=10,
        ...     snr=12.0, p99err=None, snr_thr=None)
        >>> per_z[384][(55, 4)][0]  # doctest: +SKIP
        'PASS'
    """
    zrange = ",".join(str(z) for z in zlist)
    prange = ",".join(str(p) for p in plist)
    algo_list = ",".join(str(a) for a in [ref] + duts)
    s_scalar = snr_list[0] if snr_list else snr            # -S fallback (list drives per-p)
    cmd = [binpath, "-g", str(bg), "-Z", str(zlist[0]), "--llr_diff", algo_list,
           "--llr_diff_zrange", zrange, "--llr_diff_prange", prange,
           "-n", str(iters), "-S", str(s_scalar)]
    if p99err is not None:
        cmd += ["--p99err_rms_threshold", str(p99err)]
    if snr_thr is not None:
        cmd += ["--snr_threshold", str(snr_thr)]
    if snr_list:
        cmd += ["--llr_diff_snrs", ",".join(f"{s:.4f}" for s in snr_list)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    blob = res.stdout + res.stderr
    if rawlog is not None:
        rawlog.write(f"\n{'#' * 78}\n### GRID bg={bg} Z={zlist[0]}..{zlist[-1]} "
                     f"p={plist[0]}-{plist[-1]} ref={ref} duts={','.join(map(str, duts))}"
                     f"\n{'#' * 78}\n")
        rawlog.write(blob)
    if "all zeros" in blob and "Reference" in blob:
        print(f"WARNING: reference algo{ref} APP all zeros -- the build lacks the dump "
              f"path (rebuild with -DCUPHY_LDPC_SPLIT_DUMP_KERNELS=ON).", file=sys.stderr)

    per_z = {}                 # {z: {(dut,p): (status, p99, snr)}}
    cur_z = None
    worst_p99 = None
    min_snr = None
    p99_col = snr_col = None
    hdr_re = re.compile(r"llr_diff \(TV-free\): BG=\d+ Z=(\d+) p=\d+")
    verdict_re = re.compile(r"^LLR_DIFF algo(\d+) vs algo%d p=(\d+): (.*)$" % ref)
    for line in blob.splitlines():
        hm = hdr_re.search(line)
        if hm:
            cur_z = int(hm.group(1))       # new (Z,p) cell -> track its Z from the header
            continue
        parts = line.split("|")
        if len(parts) >= 12 and parts[0].strip() == "Iter":       # check_APP header row
            hdr = [c.strip() for c in parts]
            if "p99err_rms" in hdr:
                p99_col = hdr.index("p99err_rms")
            if "SNR(dB)" in hdr:
                snr_col = hdr.index("SNR(dB)")
            continue
        if (len(parts) >= 12 and parts[0].strip().isdigit()
                and p99_col is not None and snr_col is not None):
            p99 = _to_float(parts[p99_col].strip())
            sn  = _to_float(parts[snr_col].strip())
            if p99 is not None:
                worst_p99 = p99 if worst_p99 is None else max(worst_p99, p99)
            if sn is not None:
                min_snr = sn if min_snr is None else min(min_snr, sn)
            continue
        m = verdict_re.match(line)
        if m:
            d, pp, tail = int(m.group(1)), int(m.group(2)), m.group(3).strip()
            if tail.startswith("PASS"):
                status = "PASS"
            elif tail.startswith("FAIL"):
                status = "FAIL"
            elif tail.startswith("N/A"):
                low = tail.lower()
                if "no dump path" in low:
                    status = "NO-DUMP"       # twin not compiled: harness misconfigured
                elif "reference" in low:
                    status = "REF-N/A"
                else:
                    status = "N/A"           # decoder refused the config: legitimate skip
            else:
                status = "ERR"
            per_z.setdefault(cur_z, {})[(d, pp)] = (status, worst_p99, min_snr)
            worst_p99 = None
            min_snr = None
    gate_p99 = _parse_gate(blob, r"p99err_rms_threshold = ([\d.]+)")
    gate_sn = _parse_gate(blob, r"snr_threshold\s*=\s*([\d.]+)")
    return per_z, gate_p99, gate_sn


def _to_float(s):
    if s in ("inf", "+inf"):
        return float("inf")
    if s == "-inf":
        return float("-inf")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_gate(blob, pat):
    """Recover a gate the binary echoed, so the sweep never keeps its own copy."""
    m = re.search(pat, blob)
    return float(m.group(1)) if m else None


def _fmt_snr(sn: float | None) -> str:
    """Human-readable min-SNR(dB). +inf = bit-exact vs the reference (zero error);
    -inf = reference had no signal but the DUT differed."""
    if sn is None:
        return "n/a"
    if sn == float("inf"):
        return "inf(bit-exact)"
    if sn == float("-inf"):
        return "-inf"
    return f"{sn:.1f}dB"


MARGINAL_FRAC = 0.9   # PASS cells with p99err >= this*gate render as marginal


def _symbol(st, p99, gate):
    if st == "PASS":
        return "~" if (gate is not None and p99 is not None and p99 != float("inf") and p99 >= MARGINAL_FRAC * gate) else "."
    return {"FAIL": "X", "N/A": "-", "NO-DUMP": "D", "REF-N/A": " ", "ERR": "E"}.get(st, "?")


# SNR(dB) density ramp, descending (floor, glyph): ink increases as margin
# shrinks, so the grid reads as a gradient without consulting the legend.
# Bands are absolute dB; anything under the gate is 'X' regardless of band.
SNR_BANDS = [(60.0, ":"), (45.0, "+"), (30.0, "*")]


def _symbol_snr(st: str, sn: float | None, snr_gate: float | None) -> str:
    """One cell of the SNR map. Encodes the SNR magnitude itself, not the cell's
    overall verdict -- a cell that FAILs on p99err but has healthy SNR still shows
    its SNR band here (the p99err map is where that failure surfaces).
    '.' = bit-exact (+inf), kept lightest so a rare degraded cell draws the eye."""
    if st in ("PASS", "FAIL"):
        if sn is None:
            return "?"
        if sn == float("inf"):
            return "."
        if snr_gate is not None and sn < snr_gate:
            return "X"
        for floor, glyph in SNR_BANDS:
            if sn >= floor:
                return glyph
        return "X"
    return {"N/A": "-", "NO-DUMP": "D", "REF-N/A": " ", "ERR": "E"}.get(st, "?")


def render_ascii_heatmap(dut: int, ref: int, bg: int,
                         cell: dict[tuple[int, int], CellResult],
                         gate: float | None, metric: str = "p99err",
                         snr_gate: float | None = None) -> None:
    """cell: {(z,p): (status, p99err, snr)}  -> one Z x p grid printed to stdout.
    metric selects which of the two per-cell numbers the glyphs encode."""
    zs = sorted({z for (z, p) in cell})
    ps = sorted({p for (z, p) in cell})
    if not zs or not ps:
        return
    if metric == "p99err":
        title  = "Z x p margin heatmap (p99err, lower = better)"
        legend = ("  legend: '.' pass   '~' marginal (p99err>=%d%% gate)   X fail   "
                  "'-' N/A   D no dump twin   ' ' ref-N/A   E err" % int(MARGINAL_FRAC * 100))
    else:
        title  = "Z x p margin heatmap (SNR dB, higher = better)"
        legend = ("  legend: '.' bit-exact   ':' >=60dB   '+' 45-60   '*' 30-45 (thin)   "
                  "X <%.0fdB fail   '-' N/A   D no dump twin   ' ' ref-N/A   E err"
                  % (snr_gate if snr_gate is not None else 30.0))
    print(f"\nalgo{dut} vs algo{ref}  (BG{bg})  {title}")
    print(legend)
    ruler = "".join("+" if p % 10 == 0 else ("'" if p % 5 == 0 else " ") for p in ps)
    units = "".join(str(p % 10) for p in ps)
    print("   p   " + ruler + "   (+ marks p%10==0)")
    print("       " + units)
    for z in zs:
        row = ""
        for p in ps:
            rec = cell.get((z, p))
            if rec is None:
                row += " "
            elif metric == "p99err":
                row += _symbol(rec[0], rec[1], gate)
            else:
                row += _symbol_snr(rec[0], rec[2], snr_gate)
        print(f"  Z{z:<4}{row}")


def _comparable(cell):
    return [(st, p99, sn) for (st, p99, sn) in cell.values() if st in ("PASS", "FAIL")]


def render_histogram(dut, ref, cell, gate, metric="p99err", snr_gate=None):
    if metric == "p99err":
        vals = [p99 for (st, p99, sn) in _comparable(cell) if p99 not in (None, float("inf"))]
        if not vals:
            return
        print(f"\nalgo{dut} vs algo{ref}  worst-p99err distribution (x gate={gate}):")
        edges = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1e9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            n = sum(1 for v in vals if lo * gate <= v < hi * gate)
            label = f"{lo:.2f}-{hi:.2f}x" if hi < 1e9 else f">={lo:.2f}x"
            print(f"  {label:>11}: {n:5d} {'#' * min(60, n)}")
        mx = max(vals)
        print(f"  worst p99err = {mx:.4f} ({mx / gate:.2f}x gate)")
    else:  # snr
        vals = [sn for (st, p99, sn) in _comparable(cell)
                if sn not in (None, float("inf"), float("-inf"))]
        if not vals:
            return
        print(f"\nalgo{dut} vs algo{ref}  min-SNR(dB) distribution (gate={snr_gate}):")
        edges = [-1e9, snr_gate, snr_gate + 5, snr_gate + 10, snr_gate + 20, 1e9]
        labels = [f"<{snr_gate:.0f} (FAIL)", f"{snr_gate:.0f}-{snr_gate+5:.0f}",
                  f"{snr_gate+5:.0f}-{snr_gate+10:.0f}", f"{snr_gate+10:.0f}-{snr_gate+20:.0f}",
                  f">={snr_gate+20:.0f}"]
        for (lo, hi), lab in zip(zip(edges[:-1], edges[1:]), labels):
            n = sum(1 for v in vals if lo <= v < hi)
            print(f"  {lab:>14}: {n:5d} {'#' * min(60, n)}")
        print(f"  min SNR = {min(vals):.2f} dB")


def _metric_grid(np, cell, zs, ps, metric, gate, snr_vmax):
    """NaN = no data (N/A -> white). For SNR, +inf (bit-exact) -> a value above
    vmax so it renders as the 'over' color (black), distinct from N/A; finite SNR
    is clipped to vmax so only +inf triggers 'over'."""
    grid = np.full((len(zs), len(ps)), np.nan)
    for i, z in enumerate(zs):
        for j, p in enumerate(ps):
            rec = cell.get((z, p))
            if not rec or rec[0] not in ("PASS", "FAIL"):
                continue
            if metric == "p99err":
                if rec[1] not in (None, float("inf")):
                    grid[i, j] = rec[1] / gate
            else:  # snr
                sn = rec[2]
                if sn == float("inf"):
                    grid[i, j] = snr_vmax + 1.0            # bit-exact -> over -> black
                elif sn not in (None, float("-inf")):
                    grid[i, j] = min(sn, snr_vmax)         # clip so only +inf is 'over'
    return grid


def render_png(path, dut, ref, bg, cell, gate, metrics, snr_gate=None):
    """One figure with a stacked subplot per metric (shared p-axis)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  (matplotlib unavailable; skipping PNG)")
        return None
    zs = sorted({z for (z, p) in cell})
    ps = sorted({p for (z, p) in cell})
    if not zs or not ps:
        return None
    if snr_gate is None:  # gates unset -> dump path not compiled; skip PNG (warning already printed)
        print("  (snr_gate is None -- skipping PNG; dump path likely not compiled)")
        return None
    snr_vmax = snr_gate + 30.0
    cmap_pk = plt.get_cmap("viridis").copy()
    cmap_pk.set_bad("white")
    cmap_sn = plt.get_cmap("viridis_r").copy()
    cmap_sn.set_bad("white")
    cmap_sn.set_over("black")
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, sharex=True, squeeze=False,
                             figsize=(max(6, len(ps) * 0.22), max(3.2, len(zs) * 0.26) * n))
    for ax, m in zip(axes[:, 0], metrics):
        grid = _metric_grid(np, cell, zs, ps, m, gate, snr_vmax)
        if m == "p99err":
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap_pk, vmin=0.0, vmax=1.2)
            clabel = "p99err / gate  (1.0 = fail; white = N/A)"
            title = f"worst p99err / gate ({gate})"
        else:  # higher SNR better -> reversed map so 'bad' is bright, like p99err
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap_sn,
                           vmin=snr_gate, vmax=snr_vmax)
            clabel = "min SNR dB  (Blk=BitExact, White=N/A)"
            title = f"min SNR dB (gate {snr_gate})"
        ax.set_yticks(range(len(zs)))
        ax.set_yticklabels(zs, fontsize=6)
        ax.set_ylabel("Z")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, extend="max" if m == "snr" else "neither", label=clabel)
    axes[-1, 0].set_xticks(range(len(ps)))
    axes[-1, 0].set_xticklabels(ps, fontsize=6, rotation=90)
    axes[-1, 0].set_xlabel("p")
    fig.suptitle(f"algo{dut} vs algo{ref}  BG{bg}  Z{zs[0]}-{zs[-1]} p{ps[0]}-{ps[-1]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _csv_val(x):
    if x is None:
        return ""
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return f"{x:.6g}"


def render_all(cells: dict[int, dict[tuple[int, int, int], CellResult]], duts: list[int],
               ref: int, p99err: float | None, metric: str, snr_gate: float | None,
               no_heatmap: bool, histogram: bool, png_prefix: str) -> list[str]:
    metrics = ["p99err", "snr"] if metric == "both" else [metric]
    png_paths = []
    for d in duts:
        if d not in cells:
            continue
        for bg in sorted({b for (b, z, p) in cells[d]}):
            sub = {(z, p): v for (b, z, p), v in cells[d].items() if b == bg}
            if not no_heatmap:
                for m in metrics:
                    render_ascii_heatmap(d, ref, bg, sub, p99err, m, snr_gate)
            if histogram:
                for m in metrics:
                    render_histogram(d, ref, sub, p99err, m, snr_gate)
            if png_prefix:
                zz = sorted({z for (z, p) in sub})
                pp = sorted({p for (z, p) in sub})
                tag = (f"z{zz[0]}-{zz[-1]}" if len(zz) > 1 else f"z{zz[0]}") + \
                      ("_p%d-%d" % (pp[0], pp[-1]) if len(pp) > 1 else f"_p{pp[0]}")
                pth = render_png(f"{png_prefix}_a{d}_vs_a{ref}_bg{bg}_{tag}.png", d, ref, bg,
                                 sub, p99err, metrics, snr_gate)
                if pth:
                    png_paths.append(pth)
    return png_paths


EXAMPLES = """
Examples:
  # full BG1 sweep of algo35 vs the algo40 box-plus reference:
  python3 llr_diff_sweep.py --algos 35 --bg 1 --z 32-384 --p 4-46
  # two DUTs at Z384, also saving a per-cell metrics CSV:
  python3 llr_diff_sweep.py --algos 35,41 --bg 1 --z 384 --p 4-8 --csv sweep.csv
"""


def main() -> int:
    """Run the TV-free per-iteration APP diff sweep from command-line arguments.

    Parses ``sys.argv`` (via ``argparse``), then drives ``cuphy_ex_ldpc`` across
    the requested base graphs and ``Z x p`` grid -- one decoder process per base
    graph -- printing a summary, heatmaps/histograms, and drill-down hints.

    Args:
        None. All inputs are read from ``sys.argv`` through ``argparse``.

    Returns:
        Process exit code: ``0`` if no DUT had a FAIL or ERR cell (N/A and
        REF-N/A do not count as failures; NO-DUMP does), ``1`` otherwise.

    Raises:
        SystemExit: On argument errors or ``--help`` (from ``argparse``), or when
            the ``cuphy_ex_ldpc`` binary cannot be resolved (via ``resolve_bin``).

    Examples:
        >>> import sys
        >>> sys.argv = ["llr_diff_sweep.py", "--algos", "55",
        ...             "--bg", "1", "--z", "384", "--p", "4-8"]
        >>> main()  # doctest: +SKIP
        0
    """
    ap = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLES,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", required=True,
                    help="comma list of DUT algos, e.g. 35,41")
    ap.add_argument("--ref", type=int, default=40, help="reference algo (default 40)")
    ap.add_argument("--bg", default="1,2", help="base graphs, comma list (default 1,2)")
    ap.add_argument("--z", default="32-384",
                    help="Z spec: 'all', list '256,384', or range '256-384' (default 32-384). "
                         "Swept high->low (large Z first); p ascending 4..max (BG1 46, BG2 42). "
                         "Z<32 is excluded by default: the decoder returns NOT_SUPPORTED there.")
    ap.add_argument("--p", default="all",
                    help="p spec: 'all', list, or range '4-14' (default all, clamped per BG)")
    ap.add_argument("--iters", type=int, default=10, help="LDPC iterations -n (default 10)")
    snr_mode = ap.add_mutually_exclusive_group()
    snr_mode.add_argument("--snr", type=float, default=None,
                    help="fixed AWGN SNR dB (-S) for ALL p (mutually exclusive with "
                         "--auto-snr-shannon-margin)")
    snr_mode.add_argument("--auto-snr-shannon-margin", type=float, default=3.0,
                    help="per-p auto SNR = complex-AWGN Shannon limit + this dB margin "
                         "(default 3; used when --snr is not given)")
    ap.add_argument("--p99err", type=float, default=None,
                    help="override the p99err_rms gate threshold (default: use the binary's built-in gate)")
    ap.add_argument("--snr-threshold", type=float, default=None,
                    help="override the binary's per-iter SNR(dB) gate (default: use the binary's built-in gate)")
    ap.add_argument("--bin", default=None, help="path to cuphy_ex_ldpc (else autodetect)")
    ap.add_argument("--verbose", action="store_true",
                    help="print every (bg,z,p) DUT result as it runs")
    ap.add_argument("--no-heatmap", action="store_true",
                    help="suppress the per-DUT ASCII Z x p heatmap")
    ap.add_argument("--no-histogram", action="store_true",
                    help="suppress the per-DUT histogram (on by default)")
    ap.add_argument("--csv", default="",
                    help="compact per-cell metrics CSV "
                         "(bg,z,p,dut,status,worst_p99err,min_snr)")
    ap.add_argument("--rawlog", default="",
                    help="full delimited stdout+stderr of every cell (for grep "
                         "drill-down, e.g. grep -A400 'Z=384 p=8' <rawlog>). Large.")
    args = ap.parse_args()
    args.bin = resolve_bin(args.bin)
    do_hist = not args.no_histogram
    png_prefix = "llr_diff_sweep"   # heatmap PNG is always written (auto-named)

    duts = parse_int_csv(args.algos)
    bgs = parse_int_csv(args.bg)
    snr_desc = (f"{args.snr}dB(flat)" if args.snr is not None
                else f"Shannon+{args.auto_snr_shannon_margin}dB(per-p)")
    with contextlib.ExitStack() as stack:
        csv_f = stack.enter_context(open(args.csv, "w")) if args.csv else None
        csv_header_written = False   # deferred until the gate is known (echoed by the binary)
        rawlog_f = stack.enter_context(open(args.rawlog, "w")) if args.rawlog else None

        # results[dut] = status counts; cells[dut] = {(bg,z,p): (status,p99err,snr)}
        results = {d: {"PASS": 0, "FAIL": 0, "N/A": 0, "NO-DUMP": 0, "ERR": 0, "REF-N/A": 0}
                   for d in duts}
        cells = {d: {} for d in duts}
        fails = {d: [] for d in duts}
        n_cfg = 0
        t0 = time.time()
        gate_p99 = args.p99err        # resolved from the gate the binary echoes on the first cell
        gate_sn = args.snr_threshold

        for bg in bgs:
            zs = sorted(parse_z_spec(args.z), reverse=True)   # high->low (large Z first)
            plist = sorted(parse_p_spec(args.p, bg))                    # p ascending 4..max (BG1 46, BG2 42)
            snr_list = (None if args.snr is not None
                        else [shannon_snr_db(bg, p, margin=args.auto_snr_shannon_margin) for p in plist])
            # ONE process for the whole (Z x p) grid of this BG (--llr_diff_zrange):
            # CUDA context + 60 MB module load paid once for the entire sweep, not per Z.
            per_z, gpk, gsn = run_all_z(args.bin, bg, zs, plist, args.ref, duts,
                                        args.iters, args.snr, args.p99err, args.snr_threshold,
                                        rawlog=rawlog_f, snr_list=snr_list)
            if gate_p99 is None:
                gate_p99 = gpk
            if gate_sn is None:
                gate_sn = gsn
            if csv_f and not csv_header_written:
                csv_f.write(f"# llr_diff_sweep ref={args.ref} p99err_gate={gate_p99} "
                            f"snr_gate={gate_sn} iters={args.iters} snr={snr_desc}\n")
                csv_f.write("bg,z,p,dut,status,worst_p99err,min_snr\n")
                csv_header_written = True
            for z in zs:
                r = per_z.get(z, {})
                for d in duts:
                    for p in plist:
                        st, p99, sn = r.get((d, p), ("ERR", None, None))
                        n_cfg += 1
                        results[d][st] += 1
                        cells[d][(bg, z, p)] = (st, p99, sn)
                        if st == "FAIL":
                            fails[d].append((bg, z, p))
                        if csv_f:
                            csv_f.write(f"{bg},{z},{p},{d},{st},"
                                        f"{_csv_val(p99)},{_csv_val(sn)}\n")
                        if args.verbose:
                            m = f" p99err={p99:.4f}" if p99 not in (None, float("inf")) else ""
                            if sn is not None:
                                m += f" snr={_fmt_snr(sn)}"
                            print(f"  bg{bg} Z{z} p{p} algo{d}: {st}{m}")
                if csv_f:
                    csv_f.flush()   # per-Z flush: partial results survive a crash
                if rawlog_f:
                    rawlog_f.flush()

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY: {n_cfg} configs, ref=algo{args.ref}, "
          f"gates p99err<={gate_p99} snr>={gate_sn}dB ===")
    for d in duts:
        c = results[d]
        line = (f"  algo{d}: PASS={c['PASS']} FAIL={c['FAIL']} "
                f"N/A={c['N/A']} ERR={c['ERR']}")
        if c["REF-N/A"]:
            line += f" REF-N/A={c['REF-N/A']}"
        if c["NO-DUMP"]:
            line += f" NO-DUMP={c['NO-DUMP']}"
        # worst p99err / worst (= lowest) SNR among comparable cells
        p99v = [p99 for (st, p99, sn) in cells[d].values()
                if st in ("PASS", "FAIL") and p99 not in (None, float("inf"))]
        if p99v:
            line += f"  worst_p99err={max(p99v):.4f}"
        snv = [sn for (st, p99, sn) in cells[d].values()
               if st in ("PASS", "FAIL") and sn is not None]
        if snv:
            line += f"  min_snr={_fmt_snr(min(snv))}"
        if c["FAIL"]:
            line += f"  fail@{fails[d][:8]}" + (" ..." if len(fails[d]) > 8 else "")
        print(line)

    # --- rendering -------------------------------------------------------
    png_paths = render_all(cells, duts, args.ref, gate_p99, "both",
                           gate_sn, args.no_heatmap, do_hist, png_prefix)

    # --- outputs + drill-down suggestions --------------------------------
    print("\n=== Outputs ===")
    if args.csv:
        print(f"  metrics CSV : {args.csv}")
    else:
        print("  metrics CSV : not saved      (pass --csv <path> to record per-cell metrics)")
    if args.rawlog:
        print(f"  raw log     : {args.rawlog}")
    else:
        print("  raw log     : not saved      (pass --rawlog <path> for per-cell grep drill-down)")
    if png_paths:
        for pp in png_paths:
            print(f"  heatmap PNG : {pp}")
    else:
        print("  heatmap PNG : not written    (no comparable cells, or matplotlib missing)")

    # Pick an example cell to drill: prefer a FAIL, then a marginal, then any.
    example = None
    for d in duts:
        if fails[d]:
            example = (*fails[d][0], d)
            break
    if example is None:
        for d in duts:
            for (bg, z, p), (st, p99, sn) in cells[d].items():
                if st in ("PASS",) and p99 not in (None, float("inf")) and gate_p99 is not None and p99 >= 0.9 * gate_p99:
                    example = (bg, z, p, d)
                    break
            if example:
                break
    if example is None:
        for d in duts:
            for (bg, z, p), (st, p99, sn) in cells[d].items():
                if st in ("PASS", "FAIL"):
                    example = (bg, z, p, d)
                    break
            if example:
                break

    print("\n=== Drill into one (Z,p) cell / next steps ===")
    if example:
        bg, z, p, d = example
        print("  reproduce one cell (prints check_APP table; --dump-h5 also emits a plot cmd):")
        _s = args.snr if args.snr is not None else shannon_snr_db(bg, p, margin=args.auto_snr_shannon_margin)
        print(f"    {args.bin} -g {bg} -Z {z} -p {p} --llr_diff {args.ref},{d} "
              f"-n {args.iters} -S {_s:.2f} --dump-h5")
        if args.rawlog:
            print("  find that cell in the raw log:")
            print(f"    grep -A400 'Z={z} p={p}' {args.rawlog}")

    # AWGN SNR schedule actually used per p (so the operating point is on record).
    print("\n=== AWGN SNR used per p ===")
    for bg in bgs:
        plist = sorted(parse_p_spec(args.p, bg))
        if args.snr is not None:
            print(f"  BG{bg}: {args.snr:.2f} dB (flat, all p)")
        else:
            print(f"  BG{bg}: Shannon + {args.auto_snr_shannon_margin} dB (QPSK):")
            pairs = [f"p{p}={shannon_snr_db(bg, p, margin=args.auto_snr_shannon_margin):.2f}" for p in plist]
            for i in range(0, len(pairs), 8):
                print("    " + "  ".join(pairs[i:i + 8]))

    # --- machine-parseable footer: total time (2nd last), then one verdict
    # line per algo (last lines). An algo PASSes iff it has no FAIL and no ERR
    # or NO-DUMP cells (N/A and REF-N/A do not count as failures).
    print(f"\nTotal test time: {elapsed:.1f}s")
    any_fail = False
    for d in duts:
        c = results[d]
        ok = (c["FAIL"] == 0 and c["ERR"] == 0 and c["NO-DUMP"] == 0)
        any_fail = any_fail or not ok
        print(f"LLR sweep for algo{d}: {'PASS' if ok else 'FAIL'}")
    if any(results[d]["NO-DUMP"] for d in duts):
        print("ERROR: some decoders do not have dump twin. "
              "Rebuild with -DCUPHY_LDPC_SPLIT_DUMP_KERNELS=ON.",
              file=sys.stderr)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
