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

"""Run cuphy_ldpc_find_dispatch over a grid and plot the resolved LDPC dispatch."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
_BINARY_REL = "cuPHY/examples/error_correction/cuphy_ldpc_find_dispatch"
_candidates = sorted(REPO_ROOT.glob(f"build*/{_BINARY_REL}"))
DEFAULT_BINARY = _candidates[0] if _candidates else (REPO_ROOT / f"build.aarch64/{_BINARY_REL}")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("cuBB_SDK", REPO_ROOT)) / "tmp_figures"

# Default sweep, applied only when launching the probe. In --input-csv mode the
# axes come from the CSV rows, so these must not stand in as "explicit" input.
DEFAULT_BG_LIST = ["1", "2"]
DEFAULT_Z_LIST = ["2:16", "18:32:2", "36:64:4", "72:128:8", "144:256:16", "288:384:32"]
DEFAULT_P_LIST = ["4:46"]


def parse_int_list(values: list[str] | None) -> list[int]:
    """Expand CLI int tokens (lists, commas, start:stop[:step]) into a sorted unique list."""
    if not values:
        return []
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                pieces = part.split(":")
                if len(pieces) not in (2, 3):
                    raise ValueError(f"Bad range token: {part}")
                start = int(pieces[0])
                stop = int(pieces[1])
                step = int(pieces[2]) if len(pieces) == 3 else 1
                if step <= 0:
                    raise ValueError(f"Range step must be positive: {part}")
                out.extend(range(start, stop + 1, step))
            else:
                out.append(int(part))
    return sorted(OrderedDict.fromkeys(out))


def sanitize_token(text: str) -> str:
    """Return text with filename-unsafe characters collapsed to '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def range_token(name: str, values: list[int]) -> str:
    """Format a min-max filename token, e.g. range_token('z', ...) -> 'z2-384'."""
    if not values:
        return f"{name}all"
    return f"{name}{min(values)}-{max(values)}"


def output_basename(args: argparse.Namespace) -> str:
    """Build the output file stem encoding BG/Z/p, LLR type, dispatch mode and device(s)."""
    # Append the ordinal only when a label repeats, so single-GPU filenames stay
    # stable but gb203:0 / gb203:1 don't collide.
    label_counts = Counter(label for label, _ in args.devices)
    device_text = "-".join(
        sanitize_token(label if label_counts[label] == 1 else f"{label}-{ordinal}")
        for label, ordinal in args.devices
    )
    # Encode fptype and dispatch mode: latency vs throughput and fp16 vs fp32
    # resolve to different algos, so keep their outputs from colliding.
    parts = [
        "cuphy_dispatch",
        range_token("bg", args.bg_list),
        range_token("z", args.z_list),
        range_token("p", args.p_list),
        args.fptype,
        "thr" if args.throughput else "lat",
    ]
    if args.algo != 0:
        parts.append(f"a{args.algo}")  # forced algo overrides auto-dispatch
    parts.append(f"dev{device_text}")
    return "_".join(parts)


def build_probe_cmd(args: argparse.Namespace, device_ordinal: int) -> list[str]:
    """Build the cuphy_ldpc_find_dispatch command line for one CUDA device ordinal."""
    cmd = [
        str(args.binary),
        "--device",
        str(device_ordinal),
        "--sweep",
        "--fptype",
        args.fptype,
        "--max-iter",
        str(args.max_iter),
        "--num-cw",
        str(args.num_cw),
    ]
    if args.throughput:
        cmd.append("--throughput")
    if args.algo != 0:
        cmd += ["-a", str(args.algo)]
    if args.bg_list:
        cmd += ["--bg-list", *[str(v) for v in args.bg_list]]
    if args.z_list:
        cmd += ["--z-list", *[str(v) for v in args.z_list]]
    if args.p_list:
        cmd += ["--p-list", *[str(v) for v in args.p_list]]
    return cmd


def short_device_label(device_name: str) -> str:
    """Shorten the driver-detected GPU name (cuDeviceGetName), e.g. 'NVIDIA GB203' -> 'GB203'."""
    name = device_name.strip()
    for prefix in ("NVIDIA ", "NVIDIA_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name or "gpu"


def run_probe(args: argparse.Namespace, ordinal: int) -> list[dict[str, str]]:
    """Run the probe on one device and return its CSV rows tagged with device identity."""
    cmd = build_probe_cmd(args, ordinal)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        msg = f"device {ordinal} failed with return code {proc.returncode}"
        if proc.stderr.strip():
            msg += f"\n{proc.stderr.strip()}"
        raise RuntimeError(msg)

    rows = list(csv.DictReader(proc.stdout.splitlines()))
    for row in rows:
        # Label comes from the detected GPU name (device_name column), not a
        # user-supplied flag.
        row["device_label"] = short_device_label(row.get("device_name", ""))
        row["device_ordinal"] = str(ordinal)
        row["probe_cmd"] = shlex.join(cmd)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows to CSV, using the union of keys across all rows as the header."""
    if not rows:
        raise RuntimeError("No rows to write")
    # Union of keys across all rows (first-seen order): --input-csv can merge files
    # with different columns, so keying off rows[0] alone would drop or reject them.
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csvs(paths: list[Path]) -> list[dict[str, str]]:
    """Read probe CSVs, defaulting device identity from the filename when absent."""
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if "device_label" not in row or not row["device_label"]:
                    row["device_label"] = path.stem
                if "device_ordinal" not in row:
                    row["device_ordinal"] = ""
                rows.append(row)
    return rows


def device_key(row: dict[str, str]) -> tuple[str, str]:
    """Return a row's (device_label, device_ordinal) identity.

    Same-model GPUs at different ordinals (GB203:0 vs GB203:1) must stay separate
    throughout the plot, so the ordinal is part of the key.
    """
    return (row.get("device_label", ""), row.get("device_ordinal", ""))


def device_order_from_rows(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Return the distinct device keys present in rows, in first-seen order."""
    keys: OrderedDict[tuple[str, str], None] = OrderedDict()
    for row in rows:
        keys.setdefault(device_key(row), None)
    return list(keys.keys())


def numeric(row: dict[str, str], key: str) -> int:
    """Return row[key] parsed as an int."""
    return int(row[key])


def success_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return only the rows whose dispatch query reported success."""
    return [row for row in rows if row.get("status_name") == "CUPHY_STATUS_SUCCESS"]


def gpu_title(rows: list[dict[str, str]]) -> str:
    """Return a comma-joined list of the distinct GPU names, for the figure title."""
    names: OrderedDict[str, None] = OrderedDict()
    for row in rows:
        name = row.get("device_name", "").strip() or row.get("device_label", "").strip()
        if name:
            names.setdefault(name, None)
    return ", ".join(names) if names else "unknown"


# Scaffolding tokens dropped when shortening the authoritative kernel name (from
# cuFuncGetName) for the legend. Everything else -- the variant tokens like
# x2/lowp/highp/bigreg/msc/dp -- is kept, so NEW variants auto-appear with no
# maintenance. Display only: the CSV always retains the full kernel name.
_DROP_TOKENS = {"BG1", "BG2", "split", "index", "bp", "reg", "box", "plus",
                "desc", "dyn", "fp", "tb",
                # "highp" is not a distinct variant -- it's the same core variant
                # (bp/msc/bigreg) compiled for the high-p region. The p-axis already
                # shows that region, so we fold it into the base label to keep the
                # variant count within the colour/fill budget.
                "highp",
                # "no"/"accessories" come from the "_no_accessories" lean twin
                # (early-termination compiled out). That is ET scaffolding, not a
                # decode variant, so fold the lean twin onto its base variant label.
                "no", "accessories"}


def short_label(row: dict[str, str]) -> str:
    """Shorten an authoritative kernel name to an 'algo-variant' display label."""
    name = row.get("globalfunc", "")
    if not (name.startswith("BG1_") or name.startswith("BG2_")):
        return name  # legacy / already-short CSV: pass through unchanged
    algo = row.get("algo", "?")
    toks = [t for t in name.split("_") if t not in _DROP_TOKENS]
    return f"{algo}-" + "-".join(toks) if toks else str(algo)


def _algo_of(globalfunc: str) -> int:
    """Return the algo index parsed from a label, or -1 when absent."""
    # short label starts with the algo number ("55-x2-bigreg-msc"); legacy labels
    # like "algo35"/"55bp"/"40lowp" also expose it. Parse the first integer.
    m = re.search(r"\d+", globalfunc)
    return int(m.group()) if m else -1


def color_of_algo(algo: int) -> str:
    """Return the plot colour assigned to an algo index."""
    # Colour encodes the ALGO NUMBER (stable / standard). Variants of the same
    # algo share this hue and are told apart by MARKER SHAPE + FILL, not by
    # near-identical colour shades (see variant_marker_map()).
    return {
        # 56 is the low-p neighbour of algo55; keep it well clear of algo55's
        # blue/teal families (see _ALGO55_COLORS) -> deep pink.
        56: "#d81b60", 55: "#0066ff", 51: "#ff66ff", 52: "#8a8a8a",
        35: "#ffb000", 36: "#e08a00", 40: "#ff7a00", 41: "#aa00ff", 46: "#ff3d00",
        26: "#00b050", 27: "#00c060", 29: "#00b050", 30: "#33c080",
        33: "#78ff00", 34: "#40c000",
        100: "#000000", 101: "#7a3cff", 102: "#4f46e5", 103: "#9333ea",
        105: "#dc2626", 106: "#16a34a",
    }.get(algo, "#444444")


# algo55 has many variants; give it TWO colour families so a single shape
# (diamond) with just filled/open still yields 4 clearly-distinct styles.
_ALGO55_COLORS = ("#0066ff", "#0ab0c0")  # blue, teal


def _base_marker(algo: int) -> str:
    """Return the marker shape shared by an algo's family."""
    if algo in {55, 52, 56}:
        return "D"                          # diamond
    # 51 is the bigreg algo35 (no APP/cC2V shmem split), so it shares the triangle.
    if algo in {26, 27, 29, 30, 33, 34, 35, 36, 40, 41, 46, 51}:
        return "^"                          # triangle
    if 100 <= algo <= 106:
        return "o"                          # circle
    return "s"                              # square


def _style_combos(algo: int) -> list[tuple[str, str, bool]]:
    """Return one algo's (colour, marker, filled) style combos.

    One shape per algo; variants differ by filled/open, plus a 2nd colour family
    for algo55.
    """
    if algo == 55:
        return [(_ALGO55_COLORS[0], "D", True), (_ALGO55_COLORS[0], "D", False),
                (_ALGO55_COLORS[1], "D", True), (_ALGO55_COLORS[1], "D", False)]
    c, m = color_of_algo(algo), _base_marker(algo)
    return [(c, m, True), (c, m, False)]


def assign_styles(globalfuncs) -> dict[str, tuple[str, str, bool]]:
    """Map each globalfunc to a distinct (colour, marker, filled) style within its algo."""
    # Distinct (colour, marker, filled) per variant WITHIN each algo, by sorted
    # order -> deterministic. Called per BG subplot so each subplot's <=4 algo55
    # variants map to the 4 distinct combos (shared labels get the same index in
    # both BGs, so the shared legend stays consistent).
    from collections import defaultdict
    by_algo: dict[int, list[str]] = defaultdict(list)
    for gf in sorted(set(globalfuncs)):
        by_algo[_algo_of(gf)].append(gf)
    out: dict[str, tuple[str, str, bool]] = {}
    for algo, gfs in by_algo.items():
        combos = _style_combos(algo)
        for i, gf in enumerate(gfs):
            out[gf] = combos[i % len(combos)]
    return out


# Legacy fallback for CSVs generated before cuphy_ldpc_find_dispatch emitted the
# resolved CB/CTA and occupancy-derived max CTA/SM columns.
LEGACY_GLOBALFUNC_LAUNCH_SHAPE = {
    "55bp": "2/?",
    "55msc": "2/?",
    "55bg1-bigreg-msc": "2/?",
    "40lowp": "1/?",
    "40base": "1/?",
    "40highp22": "1/?",
    "40xhighp40": "1/?",
    "algo29": "1/?",
    "algo33": "var/?",
    "algo35": "2/?",
    "algo51": "2/?",
    "algo52": "2/?",
    "algo56": "2/?",
    "algo100": "2/?",
    "algo101": "2/?",
    "algo102": "2/?",
    "algo103": "2/?",
    "algo105": "1/?",
    "algo106": "1/?",
}


def launch_shape(row: dict[str, str]) -> str:
    """Return the 'cb_per_cta/max_cta_per_sm' launch shape recorded for a row."""
    cb_per_cta = row.get("cb_per_cta", "")
    max_cta_per_sm = row.get("max_cta_per_sm", "")
    if cb_per_cta and max_cta_per_sm:
        try:
            if int(cb_per_cta) > 0 and int(max_cta_per_sm) > 0:
                return f"{cb_per_cta}/{max_cta_per_sm}"
        except ValueError:
            pass
    return LEGACY_GLOBALFUNC_LAUNCH_SHAPE.get(row["globalfunc"], "?/?")


def globalfunc_legend_label(globalfunc: str, rows: list[dict[str, str]]) -> str:
    """Return a kernel's legend label, annotated with its launch shape(s)."""
    shapes = sorted({launch_shape(row) for row in rows if row["globalfunc"] == globalfunc})
    if len(shapes) <= 3:
        shape_text = ", ".join(shapes)
    else:
        numeric_shapes: list[tuple[int, int]] = []
        for shape in shapes:
            fields = shape.split("/")
            if len(fields) != 2 or not all(field.isdigit() for field in fields):
                numeric_shapes = []
                break
            numeric_shapes.append((int(fields[0]), int(fields[1])))
        if numeric_shapes:
            cb_values = [shape[0] for shape in numeric_shapes]
            cta_values = [shape[1] for shape in numeric_shapes]
            cb_text = str(cb_values[0]) if min(cb_values) == max(cb_values) else f"{min(cb_values)}..{max(cb_values)}"
            cta_text = str(cta_values[0]) if min(cta_values) == max(cta_values) else f"{min(cta_values)}..{max(cta_values)}"
            shape_text = f"{cb_text}/{cta_text}"
        else:
            shape_text = ", ".join(shapes)
    return f"{globalfunc} ({shape_text})"


def selected_values(rows: list[dict[str, str]], key: str, fallback: list[int]) -> list[int]:
    """Return the explicit CLI values, else the distinct values present in rows."""
    if fallback:
        return fallback
    return sorted({numeric(row, key) for row in rows})


def configure_ticks(ax, values: list[int], axis: str) -> None:
    """Set readable ticks on one axis, thinning them when values are many."""
    if not values:
        return
    if len(values) <= 16:
        ticks = values
    else:
        ticks = [min(values), max(values)]
        preferred = [4, 10, 16, 22, 28, 34, 40, 46] if axis == "x" else [2, 16, 30, 32, 64, 128, 192, 256, 320, 384]
        ticks = sorted(set(ticks + [v for v in preferred if min(values) <= v <= max(values)]))
    if axis == "x":
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)


def collapse_z_bands(bg: int, ok_rows: list[dict[str, str]], z_sorted: list[int],
                     p_values: list[int], device_keys: list[tuple[str, str]]):
    """Group CONSECUTIVE Z whose full dispatch signature -- globalfunc at every
    (device, p) -- is byte-identical, into one band. Fully automatic: no Z ranges
    or algos are hardcoded; a band breaks the instant any (device, p) cell differs.
    Returns (num_bands, z_to_row, band_labels)."""
    cell = {}
    for r in ok_rows:
        if numeric(r, "bg") == bg:
            cell[(device_key(r), numeric(r, "z"), numeric(r, "p"))] = r["globalfunc"]

    def sig(z: int):
        """Return one Z's dispatch signature across every (device, p) cell."""
        return tuple(cell.get((dev, z, p), "") for dev in device_keys for p in p_values)

    bands: list[dict] = []
    prev = None
    for z in z_sorted:
        s = sig(z)
        if bands and s == prev:
            bands[-1]["hi"] = z
            bands[-1]["zs"].append(z)
        else:
            bands.append({"lo": z, "hi": z, "zs": [z]})
        prev = s

    z_to_row = {z: i for i, b in enumerate(bands) for z in b["zs"]}
    labels = [str(b["lo"]) if b["lo"] == b["hi"] else f'{b["lo"]}–{b["hi"]}'
              for b in bands]
    return len(bands), z_to_row, labels


def plot_rows(args: argparse.Namespace, rows: list[dict[str, str]], png_path: Path) -> None:
    """Render the dispatch map to png_path; requires matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        # Fail fast rather than fall back to the raster renderer, which diverged
        # from this path (no --collapse-z, different band/style mapping) and would
        # emit an ambiguous plot. The CSV is written before we get here, so only
        # the PNG needs matplotlib.
        raise RuntimeError(
            "matplotlib is required to render the dispatch PNG (the CSV is already "
            "written); install it with 'pip install matplotlib'"
        ) from exc

    ok_rows = success_rows(rows)
    if not ok_rows:
        raise RuntimeError("No successful dispatch rows to plot")

    device_keys = [(label, str(ordinal)) for label, ordinal in args.devices
                   if any(device_key(row) == (label, str(ordinal)) for row in ok_rows)]
    # Show the ordinal in titles only when a label repeats, so single-GPU plots
    # stay clean but gb203:0 / gb203:1 are distinguishable.
    label_counts = Counter(label for label, _ in device_keys)

    def device_display(dev_key: tuple[str, str]) -> str:
        """Return the subplot label for a device key, adding the ordinal only when ambiguous."""
        label, ordinal = dev_key
        return label if label_counts[label] == 1 else f"{label}:{ordinal}"

    bg_values = selected_values(ok_rows, "bg", args.bg_list)
    z_values = selected_values(ok_rows, "z", args.z_list)
    p_values = selected_values(ok_rows, "p", args.p_list)

    # Categorical Z axis: every distinct Z gets its own evenly-spaced row
    # (ascending bottom-up), so all 51 lifting sizes are visible regardless of
    # their numeric spacing. Labels show the real Z value.
    z_sorted = sorted(z_values)
    z_index = {z: i for i, z in enumerate(z_sorted)}

    # Precompute each BG's Y rows (collapsed bands or every Z) so the subplot
    # heights can be sized to each BG's row count -- a 2-band BG2 gets a short
    # row instead of the same height as a 30-band BG1.
    bg_layout = {}
    for bg in bg_values:
        if getattr(args, "collapse_z", True):
            bg_layout[bg] = collapse_z_bands(bg, ok_rows, z_sorted, p_values, device_keys)
        else:
            bg_layout[bg] = (len(z_sorted), z_index, [str(z) for z in z_sorted])

    nrows = max(1, len(bg_values))
    ncols = max(1, len(device_keys))
    fig_width = max(7.5, 5.0 * ncols)
    # Row heights proportional to each BG's band count (floor so a tiny BG still
    # has room for its title/axis labels); figure height scales with the total.
    row_counts = [max(4, bg_layout[bg][0]) for bg in bg_values]
    fig_height = max(4.0, 0.3 * sum(row_counts) + 1.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height),
                             squeeze=False, gridspec_kw={"height_ratios": row_counts})
    legend_handles: OrderedDict[str, object] = OrderedDict()
    legend_labels = {
        globalfunc: globalfunc_legend_label(globalfunc, ok_rows)
        for globalfunc in {row["globalfunc"] for row in ok_rows}
    }

    for row_idx, bg in enumerate(bg_values):
        n_y, z_to_row, y_labels = bg_layout[bg]

        for col_idx, dev_key in enumerate(device_keys):
            ax = axes[row_idx][col_idx]
            subset = [
                row for row in ok_rows
                if numeric(row, "bg") == bg and device_key(row) == dev_key
            ]
            # Per-subplot style map so each BG's variants map to distinct combos.
            style_map = assign_styles(row["globalfunc"] for row in subset)
            for globalfunc in sorted({row["globalfunc"] for row in subset}):
                points = [row for row in subset if row["globalfunc"] == globalfunc]
                color, marker, filled = style_map.get(globalfunc,
                                                       (color_of_algo(_algo_of(globalfunc)), "D", True))
                # A triangle/square at equal area reads smaller than a diamond;
                # boost their area so all shapes look comparably sized.
                size_factor = {"^": 1.5, "v": 1.5, "s": 1.25}.get(marker, 1.0)
                handle = ax.scatter(
                    [numeric(row, "p") for row in points],
                    [z_to_row[numeric(row, "z")] for row in points],
                    s=0.6 * args.marker_size * size_factor,
                    marker=marker,
                    facecolors=(color if filled else "none"),
                    edgecolors=color,
                    linewidths=(0.4 if filled else 0.9),
                    label=globalfunc,
                )
                legend_handles.setdefault(legend_labels[globalfunc], handle)

            ax.set_title(f"BG{bg} {device_display(dev_key)}")
            ax.set_xlabel("p (#parity nodes)")
            ax.set_ylabel("Z")
            if p_values:
                ax.set_xlim(min(p_values) - 1, max(p_values) + 1)
            if n_y:
                ax.set_ylim(-1, n_y)
                ax.set_yticks(range(n_y))
                ax.set_yticklabels(y_labels, fontsize=8)
            configure_ticks(ax, p_values, "x")
            # Mirror the p (x) tick labels to the top edge and the Z (y) tick
            # labels to the right edge, so both are readable on all sides.
            ax.tick_params(axis="x", top=True, labeltop=True)
            ax.tick_params(axis="y", right=True, labelright=True, labelsize=8)
            ax.grid(True, linewidth=0.5, alpha=0.45)
            ax.set_axisbelow(True)

    title = args.title or (
        f"cuPHY LDPC dispatch probe, "
        f"fptype={args.fptype}, throughput={int(args.throughput)}"
    )
    fig.suptitle(title)
    legend_bottom = 0.0
    if legend_handles:
        # Aim for ~4 legend rows: pick the column count that yields 4 rows.
        n_leg = len(legend_handles)
        ncol = max(1, math.ceil(n_leg / 4))
        n_leg_rows = math.ceil(n_leg / ncol)
        legend_bottom = min(0.26, 0.015 + 0.028 * n_leg_rows)  # reserve just enough below axes
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=ncol,
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            fontsize=9,
            markerscale=1.8,
            handlelength=1.0,      # shorter marker "badge"
            handletextpad=0.4,     # tighter gap marker->text
            columnspacing=1.0,
            labelspacing=0.4,
            borderpad=0.5,
            title="global func (CB/CTA / max CTA/SM)",
            title_fontsize=10,
        )
    # Reserve the bottom strip for the legend; normal padding between subplots.
    fig.tight_layout(rect=(0, legend_bottom, 1, 0.95), h_pad=1.4)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments and normalise the int-list options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help="Path to cuphy_ldpc_find_dispatch.")
    parser.add_argument("--device", nargs="+", type=int, default=[0], help="CUDA device ordinal(s) to probe (default 0). The column label and filename are auto-detected from the GPU name; no display name needed.")
    parser.add_argument("--input-csv", nargs="+", type=Path, default=[], help="Plot existing cuphy_ldpc_find_dispatch CSV files instead of running the GPU probe.")
    parser.add_argument("--bg-list", nargs="+", default=None, help="BG list, comma list, or ranges like 1:2. Probe mode defaults to 1 2; with --input-csv the CSV rows decide.")
    parser.add_argument("--z-list", nargs="+", default=None, help="Z list, comma list, or inclusive ranges start:stop[:step]. Probe mode defaults to all 38.212 lifting sizes; with --input-csv the CSV rows decide.")
    parser.add_argument("--p-list", nargs="+", default=None, help="p list, comma list, or inclusive ranges start:stop[:step]. Probe mode defaults to 4:46; with --input-csv the CSV rows decide.")
    parser.add_argument("-t", "--throughput", dest="throughput", action="store_true", default=False, help="Set CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT.")
    parser.add_argument("--latency", dest="throughput", action="store_false", help="Do not set CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT.")
    parser.add_argument("-a", "--algo", type=int, default=0, help="Requested algo. Use 0 for auto-dispatch.")
    parser.add_argument("--fptype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--num-cw", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--png", type=Path, default=None, help="Explicit PNG path.")
    parser.add_argument("--title", default="")
    parser.add_argument("--marker-size", type=float, default=28.0)
    parser.add_argument("--collapse-z", action=argparse.BooleanOptionalAction, default=True,
                        help="Collapse consecutive Z with identical full-p dispatch into one "
                             "labeled band (default on; use --no-collapse-z to show every Z).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-failed-devices", action="store_true", help="Warn and continue when a device probe fails.")
    args = parser.parse_args(argv)

    # Device labels are auto-detected from the probe output; populated post-probe.
    args.devices = []
    args.bg_list = parse_int_list(args.bg_list)
    args.z_list = parse_int_list(args.z_list)
    args.p_list = parse_int_list(args.p_list)
    args.binary = args.binary.resolve()
    return args


def main(argv: list[str]) -> int:
    """Collect dispatch rows, write the CSV, and render the PNG. Returns an exit code."""
    args = parse_args(argv)
    if not args.input_csv and not args.binary.exists():
        raise FileNotFoundError(f"cuphy_ldpc_find_dispatch not found: {args.binary}")

    rows: list[dict[str, str]] = []
    if args.input_csv:
        rows = read_csvs([path.resolve() for path in args.input_csv])
    else:
        # Probe path only: an unset list means "sweep the default range".
        args.bg_list = args.bg_list or parse_int_list(DEFAULT_BG_LIST)
        args.z_list = args.z_list or parse_int_list(DEFAULT_Z_LIST)
        args.p_list = args.p_list or parse_int_list(DEFAULT_P_LIST)
        for ordinal in args.device:
            try:
                rows.extend(run_probe(args, ordinal))
            except RuntimeError as exc:
                if not args.skip_failed_devices:
                    raise
                print(f"warning: {exc}", file=sys.stderr)

    if not rows:
        raise RuntimeError("No probe rows collected")

    # Device identity (label, ordinal) is derived from the rows -- the label from
    # the detected GPU name in both the probe and --input-csv paths.
    args.devices = device_order_from_rows(rows)

    basename = output_basename(args)
    png_path = args.png.resolve() if args.png else (args.output_dir / f"{basename}.png").resolve()
    csv_path = png_path.with_suffix(".csv")

    write_csv(csv_path, rows)  # full authoritative kernel names preserved
    # Display uses the shortened variant label; CSV keeps the full name.
    plot_rows(args, [dict(r, globalfunc=short_label(r)) for r in rows], png_path)

    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    return 0


if __name__ == "__main__":
    # No blanket except, so unexpected errors keep their full traceback.
    raise SystemExit(main(sys.argv[1:]))
