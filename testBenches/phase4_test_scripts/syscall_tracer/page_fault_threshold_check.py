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

# Gate a PERF test run on the total number of page faults observed during the
# trace window. A correctly pinned / memory-locked real-time PHY pipeline should
# take zero page faults in steady state, so the default threshold is 0: any page
# fault (user + kernel) fails the gate.
#
# Inputs:
#   -i, --input <path>      page_fault_summary.txt written by page_fault_summary.py
#   --threshold <N>         maximum acceptable total page-fault count
#                           (default: PAGE_FAULT_THRESHOLD_DEFAULT)
#   -o, --output <path>     write a textual gate report
#
# Exit code:
#   0  total page faults at or below threshold
#   1  total page faults exceed threshold (or the summary cannot be parsed)

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Default maximum acceptable page-fault count. Zero means "no page faults are
# tolerated": a locked-memory, pre-faulted real-time pipeline should not fault
# during the steady-state trace window, so any fault is a regression.
PAGE_FAULT_THRESHOLD_DEFAULT = 0

# Markers emitted by page_fault_summary.py.
SUMMARY_TABLE_MARKER = "Summary table (thread x page fault class)"
NO_EVENTS_MARKER = "(no page fault events found)"

# The summary table's per-row grand-total column; not a page-fault event class.
TOTAL_COLUMN = "Total"


def parse_page_fault_summary(path: Path) -> tuple[dict[str, int], int] | None:
    """Extract per-class and total page-fault counts from a page-fault summary.

    Reads the TOTAL row of the "Summary table (thread x page fault class)"
    section produced by page_fault_summary.py and returns the grand total along
    with the per-event-class breakdown (e.g. page_fault_user / page_fault_kernel
    on x86, or the combined page-faults column on ARM).

    Args:
        path: Path to the page_fault_summary.txt file.

    Returns:
        ``(per_class_counts, total)`` where ``total`` is the grand total page
        fault count, or ``None`` if the file cannot be read or the summary
        table cannot be parsed. A summary reporting no page-fault events is a
        valid result and returns ``({}, 0)`` -- it is not a parse failure.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        sys.stderr.write(f"Could not read page fault summary {path}: {e}\n")
        return None

    lines = text.splitlines()

    # A run with zero page faults emits the no-events marker and no table. That
    # is the PASS-by-default case, not a parse failure.
    if any(NO_EVENTS_MARKER in line for line in lines):
        return ({}, 0)

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

    per_class: dict[str, int] = {}
    grand_total: int | None = None
    for col, val_str in zip(col_names, values, strict=True):
        try:
            val = int(val_str)
        except ValueError:
            sys.stderr.write(
                f"Invalid TOTAL value for column {col!r} in {path}: {val_str!r}\n"
            )
            return None
        if col == TOTAL_COLUMN:
            grand_total = val
        else:
            per_class[col] = val

    # Prefer the table's own Total column; fall back to summing the event
    # classes if the layout ever omits it.
    total = grand_total if grand_total is not None else sum(per_class.values())
    return (per_class, total)


def format_report(
    threshold: int,
    per_class: dict[str, int],
    total: int,
    overall_pass: bool,
) -> str:
    lines: list[str] = [
        "Page fault threshold gate report",
        "Generated by page_fault_threshold_check.py",
        "",
        f"Threshold:           {threshold} page fault(s) (total user + kernel)",
        "",
        "========== Page fault counts ==========",
        "",
    ]

    if per_class:
        label_width = max(len(c) for c in per_class) + 1
        for col in sorted(per_class):
            lines.append(f"  {col + ':':<{label_width + 1}}{per_class[col]}")
    else:
        lines.append("  (no page fault events found)")
    lines.append(f"  {'Total:':<{(max((len(c) for c in per_class), default=5)) + 2}}{total}")
    lines.append("")

    lines.append("========== Verdict ==========")
    lines.append("")
    if overall_pass:
        lines.append("PASS")
    else:
        lines.append(f"FAIL (total {total} > threshold {threshold})")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate a PERF test run on the total page-fault count from a "
            "page_fault_summary.txt produced by page_fault_summary.py."
        ),
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_summary",
        type=Path, required=True,
        help="Path to page_fault_summary.txt produced by page_fault_summary.py",
    )
    parser.add_argument(
        "--threshold",
        type=int, default=PAGE_FAULT_THRESHOLD_DEFAULT,
        help=(
            "Maximum acceptable total page-fault count "
            f"(default: {PAGE_FAULT_THRESHOLD_DEFAULT})"
        ),
    )
    parser.add_argument(
        "-o", "--output",
        type=Path, default=None,
        help="Write the gate report to this file (in addition to stdout)",
    )
    args = parser.parse_args()

    if args.threshold < 0:
        sys.stderr.write(
            f"Invalid --threshold {args.threshold}: must be a non-negative integer\n"
        )
        return 1

    parsed = parse_page_fault_summary(args.input_summary)
    if parsed is None:
        return 1
    per_class, total = parsed

    overall_pass = total <= args.threshold

    report = format_report(
        threshold=args.threshold,
        per_class=per_class,
        total=total,
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
