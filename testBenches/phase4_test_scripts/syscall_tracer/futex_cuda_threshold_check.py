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

# Gate a PERF test run by comparing futex-attributed CUDA API counts against
# total CUDA API call counts from the LD_PRELOAD tracer log.
#
# Inputs:
#   --futex-summary <path>  per-thread futex summary written by
#                           futex_cuda_summary_multi_thread.py
#   --tracer-log <path>     cuda_api_tracer.log written by libcuda_api_tracer.so
#   --threshold <pct>       per-API and aggregate ratio threshold (default 50)
#   --min-calls <N>         minimum tracer-log total an API needs before it is
#                           subject to gating (default 10) -- avoids noisy
#                           single-call APIs failing the gate
#   -o, --output <path>     write a textual gate report
#
# Exit code:
#   0  every gated API and the aggregate are at or below threshold
#   1  any gated API or the aggregate exceeds threshold (or parse failure)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUMMARY_TABLE_MARKER = "Summary table (thread x CUDA API + CUDA + Non-CUDA + Total)"
RESERVED_COLUMNS = ("CUDA", "Non-CUDA", "Total")


def parse_futex_summary(path: Path) -> dict[str, int] | None:
    """Extract per-API futex counts from the TOTAL row of a futex/CUDA summary.

    Args:
        path: Path to the futex_cuda_summary.txt file produced by
            futex_cuda_summary_multi_thread.py.

    Returns:
        ``{api_name: futex_count}`` mapping, or ``None`` if the file cannot be
        read or the summary table cannot be parsed.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        sys.stderr.write(f"Could not read futex summary {path}: {e}\n")
        return None

    lines = text.splitlines()
    header_line: str | None = None
    header_idx: int | None = None
    in_section = False
    for i, line in enumerate(lines):
        if SUMMARY_TABLE_MARKER in line:
            in_section = True
            continue
        if in_section and line.strip():
            stripped = line.strip()
            if "Thread (name / TID)" in line or stripped.startswith("Thread"):
                header_line = line
                header_idx = i
                break
            if stripped.startswith("-"):
                break

    if header_line is None or header_idx is None:
        sys.stderr.write(f"No summary table header found in {path}\n")
        return None

    # Header: "Thread (name / TID)" + column names. Drop the four label tokens.
    tokens = header_line.split()
    if len(tokens) < 5:
        sys.stderr.write(f"Summary header too short in {path}\n")
        return None
    col_names = tokens[4:]

    total_vals: list[str] | None = None
    for line in lines[header_idx + 1:]:
        s = line.strip()
        if s.startswith("TOTAL"):
            total_vals = s.split()
            break

    if not total_vals or total_vals[0] != "TOTAL":
        sys.stderr.write(f"No TOTAL row found in {path}\n")
        return None

    values = total_vals[1:]
    if len(values) != len(col_names):
        sys.stderr.write(
            f"TOTAL row column count mismatch in {path} "
            f"(expected {len(col_names)}, got {len(values)})\n"
        )
        return None

    api_counts: dict[str, int] = {}
    for col, val_str in zip(col_names, values):
        if col in RESERVED_COLUMNS:
            continue
        try:
            api_counts[col] = int(val_str)
        except ValueError:
            sys.stderr.write(
                f"Invalid TOTAL value for column {col!r} in {path}: {val_str!r}\n"
            )
            return None
    return api_counts


def parse_tracer_log(path: Path) -> dict[str, int] | None:
    """Parse total CUDA API call counts from the LD_PRELOAD tracer log.

    Args:
        path: Path to cuda_api_tracer.log produced by libcuda_api_tracer.so.

    Returns:
        ``{api_name: total_count}`` mapping, or ``None`` if the file cannot
        be read, is empty, or contains a malformed (non-integer) count.
        Returning ``None`` causes main() to fail the gate so an empty or
        corrupt tracer log cannot silently produce an exit-0 PASS verdict.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        sys.stderr.write(f"Could not read tracer log {path}: {e}\n")
        return None

    counts: dict[str, int] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        api = parts[0]
        if api.upper() == "TOTAL":
            continue
        try:
            counts[api] = int(parts[1])
        except ValueError:
            sys.stderr.write(
                f"Invalid tracer count on line {lineno} in {path}: {line!r}\n"
            )
            return None
    if not counts:
        sys.stderr.write(f"No CUDA API counts found in tracer log {path}\n")
        return None
    return counts


def format_report(
    threshold: float,
    min_calls: int,
    per_api_rows: list[tuple[str, int, int, float, str]],
    skipped_low_volume: list[tuple[str, int, int]],
    missing_in_tracer: list[tuple[str, int]],
    aggregate: tuple[int, int, float, str] | None,
    overall_pass: bool,
) -> str:
    lines: list[str] = [
        "Futex/CUDA threshold gate report",
        "Generated by futex_cuda_threshold_check.py",
        "",
        f"Threshold:           {threshold:.2f}%",
        f"Min tracer calls:    {min_calls} (APIs with fewer total calls are not gated)",
        "",
    ]

    if per_api_rows:
        widths = {
            "api": max(len("CUDA API"), max(len(r[0]) for r in per_api_rows)),
            "futex": max(len("Futex"), max(len(str(r[1])) for r in per_api_rows)),
            "total": max(len("Total"), max(len(str(r[2])) for r in per_api_rows)),
            "ratio": max(len("Ratio (%)"), 9),
            "status": max(len("Status"), 6),
        }
        header = (
            f"{'CUDA API':<{widths['api']+2}}"
            f"{'Futex':<{widths['futex']+2}}"
            f"{'Total':<{widths['total']+2}}"
            f"{'Ratio (%)':<{widths['ratio']+2}}"
            f"{'Status':<{widths['status']}}"
        )
        lines.append("========== Per-API ratios (gated) ==========")
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        for api, futex, total, ratio, status in per_api_rows:
            lines.append(
                f"{api:<{widths['api']+2}}"
                f"{futex:<{widths['futex']+2}}"
                f"{total:<{widths['total']+2}}"
                f"{ratio:<{widths['ratio']+2}.2f}"
                f"{status:<{widths['status']}}"
            )
        lines.append("")
    else:
        lines.append("========== Per-API ratios (gated) ==========")
        lines.append("")
        lines.append("(no APIs satisfied the gating criteria)")
        lines.append("")

    if aggregate is not None:
        agg_futex, agg_total, agg_ratio, agg_status = aggregate
        lines.append("========== Aggregate ratio ==========")
        lines.append("")
        lines.append(f"Total futex (gated set): {agg_futex}")
        lines.append(f"Total calls (gated set): {agg_total}")
        lines.append(f"Aggregate ratio:         {agg_ratio:.2f}%  [{agg_status}]")
        lines.append("")
    else:
        lines.append("========== Aggregate ratio ==========")
        lines.append("")
        lines.append("(no gated APIs -- aggregate not computed)")
        lines.append("")

    if skipped_low_volume:
        lines.append(
            f"========== Skipped (tracer total < {min_calls}) =========="
        )
        lines.append("")
        for api, futex, total in skipped_low_volume:
            lines.append(f"  {api}: futex={futex}, total={total}")
        lines.append("")

    if missing_in_tracer:
        lines.append("========== Warning: in futex summary but missing from tracer log ==========")
        lines.append("")
        for api, futex in missing_in_tracer:
            lines.append(f"  {api}: futex={futex}  (LD_PRELOAD tracer does not shim this API; skipped)")
        lines.append("")

    lines.append("========== Verdict ==========")
    lines.append("")
    lines.append("PASS" if overall_pass else "FAIL")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate a PERF test run by per-API and aggregate futex/CUDA ratios."
        ),
    )
    parser.add_argument(
        "-f", "--futex-summary",
        type=Path, required=True,
        help="Path to futex_cuda_summary.txt produced by futex_cuda_summary_multi_thread.py",
    )
    parser.add_argument(
        "-t", "--tracer-log",
        type=Path, required=True,
        help="Path to cuda_api_tracer.log produced by libcuda_api_tracer.so",
    )
    parser.add_argument(
        "--threshold",
        type=float, default=50.0,
        help="Maximum acceptable futex/total ratio in percent (default: 50)",
    )
    parser.add_argument(
        "--min-calls",
        type=int, default=10,
        help="Minimum tracer-log total for an API to be gated (default: 10)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=None,
        help="Write the gate report to this file (in addition to stdout)",
    )
    args = parser.parse_args()

    futex_counts = parse_futex_summary(args.futex_summary)
    if futex_counts is None:
        return 1
    tracer_counts = parse_tracer_log(args.tracer_log)
    if tracer_counts is None:
        return 1

    per_api_rows: list[tuple[str, int, int, float, str]] = []
    skipped_low_volume: list[tuple[str, int, int]] = []
    missing_in_tracer: list[tuple[str, int]] = []

    agg_futex = 0
    agg_total = 0
    any_per_api_fail = False

    for api in sorted(tracer_counts.keys()):
        total = tracer_counts[api]
        futex = futex_counts.get(api, 0)
        if total < args.min_calls:
            if futex > 0 or total > 0:
                skipped_low_volume.append((api, futex, total))
            continue
        ratio = (100.0 * futex / total) if total > 0 else 0.0
        status = "PASS" if ratio <= args.threshold else "FAIL"
        if status == "FAIL":
            any_per_api_fail = True
        per_api_rows.append((api, futex, total, ratio, status))
        agg_futex += futex
        agg_total += total

    for api, futex in sorted(futex_counts.items()):
        if api in tracer_counts:
            continue
        if futex > 0:
            missing_in_tracer.append((api, futex))

    aggregate: tuple[int, int, float, str] | None = None
    aggregate_fail = False
    if agg_total > 0:
        agg_ratio = 100.0 * agg_futex / agg_total
        agg_status = "PASS" if agg_ratio <= args.threshold else "FAIL"
        aggregate_fail = agg_status == "FAIL"
        aggregate = (agg_futex, agg_total, agg_ratio, agg_status)

    overall_pass = not (any_per_api_fail or aggregate_fail)

    report = format_report(
        threshold=args.threshold,
        min_calls=args.min_calls,
        per_api_rows=per_api_rows,
        skipped_low_volume=skipped_low_volume,
        missing_in_tracer=missing_in_tracer,
        aggregate=aggregate,
        overall_pass=overall_pass,
    )
    sys.stdout.write(report)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        except OSError as e:
            sys.stderr.write(f"Could not write gate report {args.output}: {e}\n")
            return 1

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
