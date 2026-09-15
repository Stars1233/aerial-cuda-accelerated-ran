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

"""Generate a local PNG/PDF kernel-latency report from an Nsight Systems trace.

This is the offline plotting companion to ``nsys_parser.py``. It uses the same
context and kernel classification as the Aerial GPU Benchmark dashboard, but
writes all results locally and performs no network access or upload.

Example:
    python3 nsys_gen_report.py /path/to/run.nsys-rep
    python3 nsys_gen_report.py run.nsys-rep -o nsys_report

The recommended input is an ``.nsys-rep``. A ``.sqlite`` file previously
produced by ``nsys export`` is also supported.
When ``-o`` is omitted, report files are written to a directory named after the
input trace, next to the trace.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import nsys_parser


_CHANNEL_ORDER = ("PDSCH", "MAC", "PUSCH", "PUCCH", "PDCCH", "PRACH", "SRS")
_EXPECTED_CHANNELS = ("PDSCH", "PUSCH", "PUCCH", "PDCCH", "PRACH")
_COLORS = tuple(matplotlib.colormaps["tab20"].colors)


def _safe_filename(value: str) -> str:
    """Return a stable lowercase filename component."""
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return result or "unknown"


def _ordered_channels(channels: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(_CHANNEL_ORDER)}
    return sorted(
        channels,
        key=lambda item: (
            order.get(str(item.get("channel")), len(order)),
            -float(item.get("kernel_busy_us") or 0.0),
        ),
    )


def _group_small_slices(
    labels: list[str], values: list[float], top_k: int
) -> tuple[list[str], list[float]]:
    """Keep the largest slices and combine the rest into ``Other``."""
    pairs = sorted(zip(labels, values), key=lambda item: item[1], reverse=True)
    if top_k <= 0 or len(pairs) <= top_k:
        return [item[0] for item in pairs], [item[1] for item in pairs]
    kept = pairs[:top_k]
    other = sum(value for _, value in pairs[top_k:])
    return (
        [label for label, _ in kept] + [f"Other ({len(pairs) - top_k} kernels)"],
        [value for _, value in kept] + [other],
    )


def _kernel_figure(
    report: dict[str, Any], channel: dict[str, Any], top_k: int
) -> plt.Figure:
    """Create one channel page: percentage pie on the left, data table right."""
    kernels = channel.get("kernels") or []
    labels = [str(kernel["kernel"]) for kernel in kernels]
    values = [float(kernel["latency"]["mean_us"]) for kernel in kernels]
    labels, values = _group_small_slices(labels, values, top_k)
    if not labels or not values or sum(values) <= 0:
        return _missing_channel_figure(str(channel["channel"]), report["source"])

    total = sum(values)
    percentages = [100.0 * value / total for value in values]
    subtitle = (
        f"{report['source']} | context {channel['context_id']} | "
        "slice = mean kernel duration"
    )

    fig = plt.figure(figsize=(16, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(0.9, 1.1))
    pie_ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])

    wedges, _ = pie_ax.pie(
        values,
        colors=[_COLORS[index % len(_COLORS)] for index in range(len(values))],
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
    )
    small_labels: dict[int, list[tuple[float, float, float]]] = {-1: [], 1: []}
    for wedge, percentage in zip(wedges, percentages):
        angle = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
        x, y = math.cos(angle), math.sin(angle)
        if percentage >= 5.0:
            pie_ax.text(
                0.68 * x,
                0.68 * y,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        else:
            side = 1 if x >= 0 else -1
            small_labels[side].append((y, x, percentage))

    # Put small-slice percentages outside the pie with leader lines and enough
    # vertical separation to keep adjacent values readable.
    for side, entries in small_labels.items():
        entries.sort(key=lambda item: item[0])
        adjusted: list[tuple[float, float, float, float]] = []
        previous_y = -1.2
        for raw_y, x, percentage in entries:
            label_y = max(raw_y, previous_y + 0.13)
            adjusted.append((label_y, raw_y, x, percentage))
            previous_y = label_y
        if adjusted and adjusted[-1][0] > 1.05:
            shift = adjusted[-1][0] - 1.05
            adjusted = [
                (label_y - shift, raw_y, x, percentage)
                for label_y, raw_y, x, percentage in adjusted
            ]
        for label_y, raw_y, x, percentage in adjusted:
            pie_ax.annotate(
                f"{percentage:.1f}%",
                xy=(0.92 * x, 0.92 * raw_y),
                xytext=(1.12 * side, label_y),
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#555555",
                    "linewidth": 0.8,
                },
            )
    pie_ax.set_title("Average-time share", fontsize=12, fontweight="bold", pad=12)
    pie_ax.axis("equal")

    table_ax.axis("off")
    table_rows = [
        [
            "■",
            textwrap.fill(
                label,
                width=36,
                break_long_words=True,
                break_on_hyphens=False,
            ),
            f"{value:.3f}",
            f"{percentage:.1f}%",
        ]
        for label, value, percentage in zip(labels, values, percentages)
    ]
    table = table_ax.table(
        cellText=table_rows,
        colLabels=("", "Kernel name", "Avg time (µs)", "Percentage"),
        colWidths=(0.04, 0.56, 0.21, 0.19),
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)
    for column in range(4):
        header = table[(0, column)]
        header.set_facecolor("#1f4e78")
        header.set_text_props(color="white", fontweight="bold")
    for row in range(1, len(table_rows) + 1):
        for column in range(4):
            cell = table[(row, column)]
            cell.set_facecolor("#f2f6fa" if row % 2 else "white")
            cell.set_edgecolor("#c7d2dc")
        table[(row, 0)].get_text().set_color(
            _COLORS[(row - 1) % len(_COLORS)]
        )
        table[(row, 0)].get_text().set_fontsize(12)
        table[(row, 2)].get_text().set_ha("right")
        table[(row, 3)].get_text().set_ha("right")

    fig.suptitle(
        f"{channel['channel']} — kernel latency",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.94,
        subtitle,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    return fig


def _missing_channel_figure(channel: str, source: str) -> plt.Figure:
    """Create an explicit page when an expected channel has no kernels."""
    fig, ax = plt.subplots(figsize=(16, 8.5), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        f"No {channel} detected",
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color="#555555",
    )
    ax.text(
        0.5,
        0.45,
        "The trace contains no reportable kernels for this channel.",
        ha="center",
        va="center",
        fontsize=13,
        color="#777777",
    )
    fig.suptitle(source, fontsize=10, color="#555555")
    return fig


def _write_report(
    report: dict[str, Any],
    output_dir: Path,
    *,
    top_k: int,
    dpi: int,
) -> tuple[list[Path], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now().strftime("%Y%m%d")
    pdf_path = output_dir / f"{date_prefix}_nsys_kernel_latency_report.pdf"
    json_path = output_dir / f"{date_prefix}_nsys_kernel_latency_data.json"
    detected = {
        str(channel["channel"]): channel
        for channel in _ordered_channels(report.get("channels") or [])
    }
    missing = [
        channel for channel in _EXPECTED_CHANNELS if channel not in detected
    ]

    # Remove PNGs produced by older versions of this report from the same output
    # directory, including the retired channel-overview page.
    for pattern in (
        "??_*_kernel_latency.png",
        "??_*_module_latency.png",
        "00_channel_overview.png",
    ):
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()

    figures: list[tuple[str, plt.Figure]] = []
    page_channels = list(_EXPECTED_CHANNELS)
    page_channels.extend(
        channel for channel in _CHANNEL_ORDER
        if channel in detected and channel not in page_channels
    )
    for index, channel_name in enumerate(page_channels, 1):
        channel = detected.get(channel_name)
        figure = (
            _kernel_figure(report, channel, top_k)
            if channel is not None
            else _missing_channel_figure(channel_name, report["source"])
        )
        figures.append(
            (
                f"{index:02d}_{_safe_filename(channel_name)}_kernel_latency.png",
                figure,
            )
        )

    png_paths: list[Path] = []
    with PdfPages(pdf_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = f"Nsight kernel latency report: {report['source']}"
        metadata["Subject"] = (
            "Offline Aerial GPU Benchmark per-channel Nsight Systems analysis"
        )
        metadata["Creator"] = Path(__file__).name
        for filename, figure in figures:
            png_path = output_dir / filename
            figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            png_paths.append(png_path)

    report_data = {**report, "missing_channels": missing}
    json_path.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return [*png_paths, pdf_path, json_path], missing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate offline PNG pie charts and a combined PDF kernel-latency "
            "report from an Aerial GPU Benchmark .nsys-rep or exported "
            ".sqlite trace."
        )
    )
    parser.add_argument("input", type=Path, help="Input .nsys-rep or .sqlite file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for PNG files, PDF report, and parsed JSON data "
            "(default: a directory named after the input trace, next to it)."
        ),
    )
    parser.add_argument(
        "--nsys-bin",
        default="nsys",
        help="Nsight Systems executable used for export (default: nsys).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Show only the N largest kernel slices and combine the rest as Other. "
            "The default 0 shows every kernel, matching Grafana."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution in dots per inch (default: 180).",
    )
    return parser.parse_args()


def main() -> int:
    """Generate a local kernel-latency report from command-line arguments.

    Args:
        None.

    Returns:
        Zero on success, one when report generation fails, or two when command-line
        input validation fails.

    Raises:
        None. Expected parsing, file, and subprocess errors are reported to stderr.

    Examples:
        Run with the report written to ``run/`` next to the input trace::

            python3 nsys_gen_report.py run.nsys-rep
    """
    args = _parse_args()
    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2
    if args.top_k < 0:
        print("ERROR: --top-k must be zero or greater", file=sys.stderr)
        return 2
    if args.dpi <= 0:
        print("ERROR: --dpi must be greater than zero", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.input.parent / args.input.stem
    )
    try:
        report = nsys_parser.parse_nsys(args.input, nsys_bin=args.nsys_bin)
        if not report.get("channels"):
            raise RuntimeError("the trace contains no reportable CUDA channel contexts")
        outputs, missing_channels = _write_report(
            report,
            output_dir,
            top_k=args.top_k,
            dpi=args.dpi,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    png_count = sum(path.suffix.lower() == ".png" for path in outputs)
    print(
        f"[nsys_gen_report] {report['num_channels']} detected channels -> "
        f"{png_count} PNG files + PDF + JSON in {output_dir}"
    )
    for channel in missing_channels:
        print(f"  No {channel} detected")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
