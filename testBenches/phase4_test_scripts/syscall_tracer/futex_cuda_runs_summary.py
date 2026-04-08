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

# Collate per-run futex/CUDA summary files from subfolders into one run-level summary.
# Each subfolder of the input parent folder is treated as one test run; thread-level
# detail is collapsed (use the per-run summary file in each subfolder for thread-level analysis).
#
# Usage:
#   python3 futex_cuda_runs_summary.py -i <parent_folder> [--summary-name NAME] [-o OUTPUT.txt]
#   python3 futex_cuda_runs_summary.py -i <parent_folder> --include-tracer-log [--tracer-log-name NAME]
#
# Looks for summary files in each direct subfolder: first the given name (default:
# futex_cuda_summary_multi_thread.txt), then any file with "summary" in its name.
# Writes table 1 (run totals), table 2 (CUDA API breakdown by run). With --include-tracer-log,
# also looks for a CUDA API tracer log in each run folder (default: file with "cuda_api" in
# name, e.g. cuda_api_tracer.log) and adds table 3: per-API Total (tracer) vs Futex (summary) per run.

import argparse
import sys
from pathlib import Path


# Section title in the per-run summary file
SUMMARY_TABLE_MARKER = "Summary table (thread x CUDA API + CUDA + Non-CUDA + Total)"
# Default summary filename we look for in each run subfolder
DEFAULT_SUMMARY_FILENAME = "futex_cuda_summary_multi_thread.txt"
# Keyword to look for in tracer log filename when not explicitly given (e.g. cuda_api_tracer.log)
TRACER_LOG_KEYWORD = "cuda_api"
# Aggregated columns we expect in the TOTAL row (last three columns)
CUDA_COL = "CUDA"
NON_CUDA_COL = "Non-CUDA"
TOTAL_COL = "Total"
# Placeholder when an API is in summary but not in tracer log
NOT_COUNTED = "not counted"


def parse_summary_file(path: Path) -> dict[str, int] | None:
    """
    Parse a per-run summary file and return the TOTAL row as a dict: column_name -> value.
    Returns None if the file cannot be parsed or has no summary table.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        sys.stderr.write(f"Could not read {path}: {e}\n")
        return None

    lines = text.splitlines()
    # Find the summary table section and the header line
    header_line: str | None = None
    header_idx: int | None = None
    in_section = False
    for i, line in enumerate(lines):
        if SUMMARY_TABLE_MARKER in line:
            in_section = True
            continue
        if in_section and line.strip():
            if "Thread (name / TID)" in line or line.strip().startswith("Thread"):
                header_line = line
                header_idx = i
                break
            # If we hit a dash line or data before finding header, skip
            if line.strip().startswith("-"):
                break
        if in_section and line.strip().startswith("TOTAL"):
            # Header might be two lines back
            break

    if not header_line:
        sys.stderr.write(f"No summary table header found in {path}\n")
        return None

    # Header: "Thread (name / TID)" + spaces + column names (space-separated)
    # Split by whitespace; first 4 tokens are "Thread", "(name", "/", "TID)"
    tokens = header_line.split()
    if len(tokens) < 5:
        sys.stderr.write(f"Header too short in {path}\n")
        return None
    col_names = tokens[4:]  # skip "Thread (name / TID)"

    # Find the TOTAL row only after the summary table header (avoid matching TOTAL elsewhere in file)
    total_vals: list[str] | None = None
    search_start = (header_idx + 1) if header_idx is not None else 0
    for line in lines[search_start:]:
        s = line.strip()
        if s.startswith("TOTAL"):
            total_vals = s.split()
            break
    if not total_vals or total_vals[0] != "TOTAL":
        sys.stderr.write(f"No TOTAL row found in {path}\n")
        return None
    # total_vals[0] is "TOTAL", rest are values
    values = total_vals[1:]
    if len(values) != len(col_names):
        sys.stderr.write(f"TOTAL row column count mismatch in {path} (expected {len(col_names)}, got {len(values)})\n")
        return None

    result: dict[str, int] = {}
    for col, val_str in zip(col_names, values):
        try:
            result[col] = int(val_str)
        except ValueError:
            result[col] = 0
    return result


def discover_run_folders(parent: Path) -> list[Path]:
    """Return direct subdirectories of parent, sorted by name."""
    if not parent.is_dir():
        return []
    return sorted([p for p in parent.iterdir() if p.is_dir()])


def find_summary_in_folder(folder: Path, summary_name: str) -> Path | None:
    """
    Return path to summary file in folder. Tries summary_name first; if not found,
    looks for any file in the folder whose name contains 'summary' (case-insensitive).
    """
    p = folder / summary_name
    if p.is_file():
        return p
    # Fallback: any file with "summary" in the name
    candidates = sorted(
        f for f in folder.iterdir()
        if f.is_file() and "summary" in f.name.lower()
    )
    return candidates[0] if candidates else None


def find_tracer_log_in_folder(folder: Path, explicit_name: str | None) -> Path | None:
    """
    Return path to CUDA API tracer log in folder. If explicit_name is set, use that file;
    otherwise look for any file whose name contains TRACER_LOG_KEYWORD (e.g. cuda_api).
    """
    if explicit_name:
        p = folder / explicit_name
        return p if p.is_file() else None
    candidates = sorted(
        f for f in folder.iterdir()
        if f.is_file() and TRACER_LOG_KEYWORD in f.name.lower()
    )
    return candidates[0] if candidates else None


def parse_tracer_log(path: Path) -> dict[str, int] | None:
    """
    Parse a cuda_api_tracer-style log. Format: lines like "API_NAME COUNT", comment lines start with #.
    Returns dict api_name -> total count, or None on error.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        sys.stderr.write(f"Could not read tracer log {path}: {e}\n")
        return None
    result: dict[str, int] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        api_name = parts[0]
        if api_name.upper() == "TOTAL":
            continue
        try:
            count = int(parts[1])
        except ValueError:
            continue
        result[api_name] = count
    return result


def get_cuda_apis(runs_data: list[tuple[str, dict[str, int]]]) -> list[str]:
    """Sorted list of all CUDA API names appearing in any run (excludes CUDA, Non-CUDA, Total)."""
    apis: set[str] = set()
    for _run_name, data in runs_data:
        apis.update(
            k for k in data.keys()
            if k not in (CUDA_COL, NON_CUDA_COL, TOTAL_COL)
        )
    return sorted(apis)


def build_run_summary_table(
    runs_data: list[tuple[str, dict[str, int]]],
) -> tuple[list[str], list[list[str]]]:
    """
    Build first table: Run, Total futex, Total CUDA, Total Non-CUDA, % CUDA, % Non-CUDA.
    Returns (headers, rows).
    """
    if not runs_data:
        return (["Run"], [])

    headers = [
        "Run",
        "Total futex",
        "Total CUDA",
        "Total Non-CUDA",
        "% CUDA",
        "% Non-CUDA",
    ]
    rows: list[list[str]] = []
    for run_name, data in runs_data:
        total_futex = data.get(TOTAL_COL, 0)
        total_cuda = data.get(CUDA_COL, 0)
        total_non_cuda = data.get(NON_CUDA_COL, 0)
        pct_cuda = (100.0 * total_cuda / total_futex) if total_futex else 0.0
        pct_non_cuda = (100.0 * total_non_cuda / total_futex) if total_futex else 0.0
        rows.append([
            run_name,
            str(total_futex),
            str(total_cuda),
            str(total_non_cuda),
            f"{pct_cuda:.2f}",
            f"{pct_non_cuda:.2f}",
        ])
    return (headers, rows)


def build_api_breakdown_table(
    runs_data: list[tuple[str, dict[str, int]]],
) -> tuple[list[str], list[list[str]]]:
    """
    Build second table: one row per (run, CUDA API) with Run, CUDA API, Count, %.
    % is percentage of that run's total CUDA calls.
    """
    if not runs_data:
        return (["Run", "CUDA API", "Count", "%"], [])

    cuda_apis = get_cuda_apis(runs_data)
    headers = ["Run", "CUDA API", "Count", "%"]
    rows: list[list[str]] = []
    for run_name, data in runs_data:
        total_cuda = data.get(CUDA_COL, 0)
        for api in cuda_apis:
            count = data.get(api, 0)
            pct = (100.0 * count / total_cuda) if total_cuda else 0.0
            rows.append([run_name, api, str(count), f"{pct:.2f}"])
    return (headers, rows)


def build_tracer_comparison_table(
    runs_data: list[tuple[str, dict[str, int]]],
    runs_tracer_data: list[dict[str, int] | None],
) -> tuple[list[str], list[list[str]]]:
    """
    Build third table: one row per CUDA API, one column per run with two sub-columns
    (Total CUDA API calls from tracer log, CUDA API calls from futex/summary).
    If an API is in the summary but not in the tracer log for that run, use NOT_COUNTED for Total.
    """
    if not runs_data or len(runs_tracer_data) != len(runs_data):
        return (["CUDA API"], [])

    # Union of all APIs from summary and from tracer logs
    apis: set[str] = set(get_cuda_apis(runs_data))
    for tracer in runs_tracer_data:
        if tracer:
            apis.update(tracer.keys())
    api_list = sorted(apis)

    headers = ["CUDA API"]
    for run_name, _ in runs_data:
        headers.append(f"{run_name} (Total)")
        headers.append(f"{run_name} (Futex)")

    rows: list[list[str]] = []
    for api in api_list:
        row = [api]
        for (run_name, summary_dict), tracer_dict in zip(runs_data, runs_tracer_data):
            total_val: str
            if tracer_dict is not None and api in tracer_dict:
                total_val = str(tracer_dict[api])
            else:
                total_val = NOT_COUNTED
            futex_val = str(summary_dict.get(api, 0))
            row.append(total_val)
            row.append(futex_val)
        rows.append(row)

    return (headers, rows)


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Format a single table with aligned columns; returns list of lines."""
    if not rows:
        return []
    widths = [max(len(h), 8) for h in headers]
    for row in rows:
        for i, v in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(v))
    lines = []
    header_line = "".join(f"{h:<{widths[i]+1}}" for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    for row in rows:
        line = "".join(f"{row[i]:<{widths[i]+1}}" for i in range(len(headers)))
        lines.append(line)
    return lines


def write_summary_file(
    out_path: Path,
    runs_data: list[tuple[str, dict[str, int]]],
    runs_tracer_data: list[dict[str, int] | None] | None = None,
) -> None:
    """Write the collated run-level summary to out_path: table 1 = run totals, table 2 = CUDA API breakdown; optional table 3 = tracer vs futex per API per run."""
    if not runs_data:
        out_path.write_text(
            "No run data to summarize.\n",
            encoding="utf-8",
        )
        return

    lines = [
        "Futex/CUDA summary across test runs (thread-level detail collapsed)",
        "Generated by futex_cuda_runs_summary.py",
        "",
    ]

    # Table 1: Run-level totals
    h1, rows1 = build_run_summary_table(runs_data)
    lines.append("========== Run summary (Total futex, CUDA, Non-CUDA, %) ==========")
    lines.append("")
    lines.extend(_format_table(h1, rows1))
    lines.append("")

    # Table 2: CUDA API breakdown per run
    h2, rows2 = build_api_breakdown_table(runs_data)
    lines.append("========== CUDA API breakdown by run (Count and % of run's CUDA calls) ==========")
    lines.append("")
    lines.extend(_format_table(h2, rows2))
    lines.append("")

    # Table 3: Tracer log vs futex per API per run (only when --include-tracer-log was used)
    if runs_tracer_data is not None:
        h3, rows3 = build_tracer_comparison_table(runs_data, runs_tracer_data)
        lines.append("========== CUDA API: Total (tracer log) vs Futex (summary) by run ==========")
        lines.append("(Total = from cuda_api tracer log; Futex = from futex-attributed summary. 'not counted' = API not in tracer log.)")
        lines.append("")
        lines.extend(_format_table(h3, rows3))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collate per-run futex/CUDA summary files from subfolders into one run-level summary.",
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        dest="parent_folder",
        help="Parent folder containing one subfolder per test run",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output summary file (default: <parent>/futex_cuda_runs_summary.txt)",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default=DEFAULT_SUMMARY_FILENAME,
        help=f"Summary filename to look for first in each run subfolder (default: {DEFAULT_SUMMARY_FILENAME}); if not found, any file with 'summary' in the name is used",
    )
    parser.add_argument(
        "--include-tracer-log",
        action="store_true",
        help="Parse CUDA API tracer log in each run subfolder and add a third table (Total vs Futex per API per run). Log file: --tracer-log-name if set, else any file with 'cuda_api' in the name (e.g. cuda_api_tracer.log).",
    )
    parser.add_argument(
        "--tracer-log-name",
        type=str,
        default=None,
        help="Filename for the tracer log in each run subfolder (e.g. cuda_api_tracer.log). Only used with --include-tracer-log; if not set, a file with 'cuda_api' in the name is looked for.",
    )
    args = parser.parse_args()

    parent = args.parent_folder.resolve()
    if not parent.is_dir():
        print(f"Error: parent folder does not exist or is not a directory: {parent}", file=sys.stderr)
        return 1

    run_folders = discover_run_folders(parent)
    if not run_folders:
        print("Error: no subfolders found under", parent, file=sys.stderr)
        return 1

    runs_data: list[tuple[str, dict[str, int]]] = []
    runs_tracer_data: list[dict[str, int] | None] | None = None
    if args.include_tracer_log:
        runs_tracer_data = []

    for folder in run_folders:
        summary_path = find_summary_in_folder(folder, args.summary_name)
        if not summary_path:
            sys.stderr.write(f"Skipping {folder.name}: no summary file found (tried {args.summary_name}, then any file with 'summary' in name)\n")
            continue
        data = parse_summary_file(summary_path)
        if data is None:
            continue
        runs_data.append((folder.name, data))

        if args.include_tracer_log and runs_tracer_data is not None:
            tracer_path = find_tracer_log_in_folder(folder, args.tracer_log_name)
            if tracer_path:
                tracer_dict = parse_tracer_log(tracer_path)
                runs_tracer_data.append(tracer_dict)  # None if parse failed
            else:
                runs_tracer_data.append(None)

    if not runs_data:
        print("Error: no run summary files could be parsed.", file=sys.stderr)
        return 1

    if args.include_tracer_log and runs_tracer_data is not None and len(runs_tracer_data) != len(runs_data):
        runs_tracer_data = None  # should not happen; safety

    out_path = args.output
    if out_path is None:
        out_path = parent / "futex_cuda_runs_summary.txt"
    else:
        out_path = out_path.resolve()

    write_summary_file(out_path, runs_data, runs_tracer_data)
    print(f"Collated summary written to {out_path} ({len(runs_data)} run(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
