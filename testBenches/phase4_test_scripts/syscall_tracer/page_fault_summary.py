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

# Summarize per-thread (per-TID) user-space and kernel-space page fault counts
# from perf record data.
#
# Reads the same perf record files captured by perf_trace_workers.sh
# (which records exceptions:page_fault_user and exceptions:page_fault_kernel
# alongside futex syscalls in one stream); ignores other event types and any
# stack frames. Counts are attributed to the (comm, tid) on the event line.
#
# Usage:
#   python3 page_fault_summary.py -i <folder_path> [options]
#   python3 page_fault_summary.py <folder_path> [options]

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# Event classes we care about, in order of priority -- the first one whose tag
# substring appears on an event line wins. Order matters because longer/more
# specific tags must be checked before shorter ones: e.g. "page-faults" is
# generic and would also match a hypothetical "page-faults:something" line, so
# it is checked last.
#
# Per-arch coverage:
#   x86_64   -> "page_fault_user" + "page_fault_kernel" populated (split).
#   aarch64  -> "page-faults" populated (combined; kernel does not expose a
#               clean user/kernel split via tracepoints).
USER_EVENT_TAG = "page_fault_user"
KERNEL_EVENT_TAG = "page_fault_kernel"
COMBINED_EVENT_TAG = "page-faults"

USER_COL = "page_fault_user"
KERNEL_COL = "page_fault_kernel"
COMBINED_COL = "page-faults"
EVENT_COLS_ORDER = (USER_COL, KERNEL_COL, COMBINED_COL)

UNKNOWN_ATTRIBUTION = "[unknown]"

# Stack-frame line in `perf script -g --call-graph fp` output:
#   "  <hex addr>  <symbol>[+0x<off>] (<library/path>)"
STACK_LINE_RE = re.compile(r"^\s+([0-9a-fA-F]+)\s+(.+?)\s+\((.+)\)\s*$")
# Strip the "+0x<offset>" suffix from a resolved symbol.
SYMBOL_STRIP_OFFSET = re.compile(r"^(.+?)\+0x[0-9a-fA-F]*$")


def _strip_args(sym: str) -> str:
    """
    Remove the argument list from a demangled C++ symbol. Balanced-paren scan
    from the end so operator overloads like `Class::operator()(int)` keep the
    `operator()` part intact (the outer `(int)` is stripped). Returns the input
    unchanged if no balanced argument list is found.
    """
    depth = 0
    for i in range(len(sym) - 1, -1, -1):
        c = sym[i]
        if c == ')':
            depth += 1
        elif c == '(':
            depth -= 1
            if depth == 0:
                return sym[:i].rstrip()
    return sym


def _attribution_for_stack(stack_lines: list[str]) -> str:
    """
    Walk the stack from innermost (top) outwards and return the first resolved
    symbol, with `+0x...` and the argument list stripped. Falls back to
    UNKNOWN_ATTRIBUTION when every frame is `[unknown]` / unparseable.
    """
    for raw in stack_lines:
        m = STACK_LINE_RE.match(raw)
        if not m:
            continue
        sym_part = m.group(2).strip()
        if sym_part in ("[unknown]", "<unknown>"):
            continue
        # Strip "+0x<offset>" tail if present.
        m2 = SYMBOL_STRIP_OFFSET.match(sym_part)
        if m2:
            sym_part = m2.group(1).strip()
        sym_part = _strip_args(sym_part)
        if sym_part:
            return sym_part
    return UNKNOWN_ATTRIBUTION


def parse_perf_script_output(
    text: str, file_tid: str | None
) -> tuple[
    dict[str, dict[tuple[str, str], int]],
    dict[str, dict[tuple[str, str, str], int]],
]:
    """
    Parse 'perf script -g --call-graph fp' output. Each page-fault event is
    attributed to the (comm, tid) on its event line, plus the innermost
    resolved user-space symbol from the following stack frames.

    Returns:
        (totals, attribution)

        totals[event_class][(comm, tid)] -> count
        attribution[event_class][(comm, tid, function)] -> count

        event_class is one of USER_COL / KERNEL_COL / COMBINED_COL. Empty
        classes are present with empty inner dicts so callers do not need to
        .get() with defaults.
    """
    totals: dict[str, dict[tuple[str, str], int]] = {
        USER_COL: defaultdict(int),
        KERNEL_COL: defaultdict(int),
        COMBINED_COL: defaultdict(int),
    }
    attribution: dict[str, dict[tuple[str, str, str], int]] = {
        USER_COL: defaultdict(int),
        KERNEL_COL: defaultdict(int),
        COMBINED_COL: defaultdict(int),
    }

    current_event_class: str | None = None
    current_comm: str | None = None
    current_tid: str | None = None
    stack_lines: list[str] = []

    def flush() -> None:
        nonlocal current_event_class, current_comm, current_tid, stack_lines
        if current_event_class is None:
            stack_lines = []
            return
        comm = current_comm or "unknown"
        tid = current_tid or (file_tid or "unknown")
        totals[current_event_class][(comm, tid)] += 1
        func = _attribution_for_stack(stack_lines)
        attribution[current_event_class][(comm, tid, func)] += 1
        current_event_class = None
        current_comm = None
        current_tid = None
        stack_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        # Indented lines that follow an event header are stack frames. Tabs and
        # spaces both indicate indentation in perf script output.
        if line and (raw_line.startswith(" ") or raw_line.startswith("\t")):
            if current_event_class is not None:
                stack_lines.append(line)
            continue
        if not line:
            # Blank line terminates the current event's stack.
            flush()
            continue

        # Non-indented, non-empty -> potentially a new event header. Flush any
        # event in progress first so its accumulated stack is attributed.
        flush()

        # Check most specific tags first so a generic "page-faults" line is not
        # stolen by the user/kernel checks, and vice versa.
        if KERNEL_EVENT_TAG in line:
            event_class = KERNEL_COL
        elif USER_EVENT_TAG in line:
            event_class = USER_COL
        elif COMBINED_EVENT_TAG in line:
            event_class = COMBINED_COL
        else:
            continue

        # Event line format: "comm tid [cpu] timestamp: event: ..."
        # perf script may emit either "tid" or "pid/tid" in column 2 depending on
        # the --fields used when recording; rsplit on '/' is a no-op for the
        # plain-tid case and extracts the trailing TID for the combined case so
        # per-TID aggregation is not split across PID/TID pairs.
        parts = line.split()
        comm = "unknown"
        tid = file_tid or "unknown"
        if len(parts) >= 2:
            comm = parts[0]
            pid_tid = parts[1]
            tid = pid_tid.rsplit("/", 1)[-1] or (file_tid or "unknown")
        current_event_class = event_class
        current_comm = comm
        current_tid = tid
        stack_lines = []

    # Flush any tail event whose stack was never followed by a blank line.
    flush()

    return (
        {k: dict(v) for k, v in totals.items()},
        {k: dict(v) for k, v in attribution.items()},
    )


def discover_perf_data_files(folder: Path, recursive: bool = False) -> list[tuple[Path, str | None]]:
    """Discover perf record data files in a directory.

    Scans for files matching known perf data naming conventions (``perf.data``,
    ``perf.data.<tid>``, ``perf.<tid>``).  Only filenames containing "perf"
    are accepted for the numeric-suffix heuristic branches.

    Args:
        folder: Directory to search for perf data files.
        recursive: If True, recurse into subdirectories.

    Returns:
        List of ``(file_path, tid_from_filename)`` tuples.  *tid_from_filename*
        is ``None`` when the filename does not encode a TID.
    """
    results: list[tuple[Path, str | None]] = []
    if not folder.is_dir():
        return results
    it = folder.rglob("*") if recursive else folder.iterdir()
    for p in it:
        if not p.is_file():
            continue
        name = p.name
        tid_from_name: str | None = None
        if name == "perf.data":
            pass
        elif "perf" in name.lower() and name.endswith(".data"):
            pass
        elif ".data." in name:
            stem, suffix = name.rsplit(".data.", 1)
            if not suffix.isdigit() or "perf" not in stem.lower():
                continue
            tid_from_name = suffix
        elif "." in name:
            stem, suffix = name.rsplit(".", 1)
            if not suffix.isdigit() or "perf" not in stem.lower():
                continue
            tid_from_name = suffix
        else:
            continue
        results.append((p, tid_from_name))
    return results


def run_perf_script(perf_exe: str, data_path: Path, use_sudo: bool) -> str:
    """Run ``perf script -i <data_path>`` and return stdout.

    Args:
        perf_exe: Path to the perf executable.
        data_path: Path to the perf data file.
        use_sudo: If True, prefix the command with ``sudo``.

    Returns:
        Decoded stdout from perf script, or an empty string on error/timeout.
    """
    cmd = [perf_exe, "script", "--force", "-i", str(data_path)]
    if use_sudo:
        cmd = ["sudo"] + cmd
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0 and result.stderr:
            sys.stderr.write(f"perf script warning for {data_path}: {result.stderr}\n")
        return result.stdout or ""
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"perf script timed out for {data_path}\n")
        return ""
    except FileNotFoundError:
        sys.stderr.write(f"perf executable not found: {perf_exe}\n")
        return ""
    except OSError as e:
        sys.stderr.write(f"perf script failed for {data_path}: {e}\n")
        return ""


def merge_counts(
    dst: dict[str, dict[tuple, int]],
    src: dict[str, dict[tuple, int]],
) -> None:
    """
    Accumulate src into dst in place. Works for both:
      - totals:      event_class -> (comm, tid)       -> count
      - attribution: event_class -> (comm, tid, func) -> count
    Inner keys may be tuples of any arity; behavior is identical.
    """
    for event_class, src_inner in src.items():
        dst_inner = dst.setdefault(event_class, {})
        for k, v in src_inner.items():
            dst_inner[k] = dst_inner.get(k, 0) + v


def _active_columns(counts: dict[str, dict[tuple, int]]) -> list[str]:
    """Return event-class columns to render: those with any non-zero count."""
    return [
        col for col in EVENT_COLS_ORDER
        if any(v > 0 for v in counts.get(col, {}).values())
    ]


def _format_int_row(
    label: str,
    values: list[int],
    label_width: int,
    col_widths: list[int],
) -> str:
    return f"{label:<{label_width}}" + "".join(
        f"{str(v):<{w + 1}}" for v, w in zip(values, col_widths)
    )


def write_summary(
    out_path: Path,
    counts: dict[str, dict[tuple[str, str], int]],
    attribution: dict[str, dict[tuple[str, str, str], int]] | None = None,
) -> None:
    """
    Write per-thread page-fault table plus (when attribution is provided)
    per-thread and cross-thread attribution sections. Only event classes that
    were observed (non-zero anywhere) are included as columns, so x86 runs
    show user/kernel and ARM runs show the combined 'page-faults' column.
    """
    active_cols = _active_columns(counts)

    thread_keys: set[tuple[str, str]] = set()
    for col in active_cols:
        thread_keys.update(counts.get(col, {}).keys())
    sorted_threads = sorted(thread_keys, key=lambda k: (k[1], k[0]))

    lines: list[str] = [
        "Page fault counts per thread",
        "Generated by page_fault_summary.py",
        "",
    ]

    if not active_cols or not sorted_threads:
        lines.append("(no page fault events found)")
        lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    # ---- Per-thread totals (legacy section, unchanged) ---------------------
    for (comm, tid) in sorted_threads:
        lines.append(f"--- Thread: {comm} (TID {tid}) ---")
        thread_total = 0
        for col in active_cols:
            v = counts.get(col, {}).get((comm, tid), 0)
            thread_total += v
            lines.append(f"  {col + ':':<19}{v}")
        lines.append(f"  {'Total:':<19}{thread_total}")
        lines.append("")

    # ---- Summary table (legacy section, unchanged) -------------------------
    lines.append("========== Summary table (thread x page fault class) ==========")
    lines.append("")

    total_col = "Total"
    cols = list(active_cols) + [total_col]
    label_width = max(23, max(len(f"{c} / {t}") for c, t in sorted_threads) + 2)
    col_widths = [max(8, len(c) + 1) for c in cols]
    header = f"{'Thread (name / TID)':<{label_width}}" + "".join(
        f"{c:<{w + 1}}" for c, w in zip(cols, col_widths)
    )
    lines.append(header)
    lines.append("-" * len(header))

    col_totals = [0] * len(cols)
    for (comm, tid) in sorted_threads:
        row_vals: list[int] = []
        row_total = 0
        for col in active_cols:
            v = counts.get(col, {}).get((comm, tid), 0)
            row_vals.append(v)
            row_total += v
        row_vals.append(row_total)
        for i, v in enumerate(row_vals):
            col_totals[i] += v
        lines.append(_format_int_row(f"{comm} / {tid}", row_vals, label_width, col_widths))

    lines.append(_format_int_row("TOTAL", col_totals, label_width, col_widths))
    lines.append("")

    # ---- Attribution sections (new) ----------------------------------------
    if attribution is not None:
        _append_attribution_sections(lines, attribution, active_cols, sorted_threads)

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _append_attribution_sections(
    lines: list[str],
    attribution: dict[str, dict[tuple[str, str, str], int]],
    active_cols: list[str],
    sorted_threads: list[tuple[str, str]],
) -> None:
    """Append per-thread attribution sub-tables and a cross-thread roll-up."""
    # Build: per_thread_attr[(comm, tid)] -> {func -> {event_class -> count}}
    per_thread_attr: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    for event_class, inner in attribution.items():
        for (comm, tid, func), count in inner.items():
            funcs = per_thread_attr.setdefault((comm, tid), {})
            ec_counts = funcs.setdefault(func, {})
            ec_counts[event_class] = ec_counts.get(event_class, 0) + count

    if not per_thread_attr:
        return

    # Per-thread attribution sub-tables.
    lines.append("========== Page fault attribution per thread (innermost resolved symbol) ==========")
    lines.append("")
    for (comm, tid) in sorted_threads:
        funcs = per_thread_attr.get((comm, tid))
        if not funcs:
            continue
        # Rank rows by descending total within thread.
        rows: list[tuple[str, list[int], int]] = []
        for func, ec_counts in funcs.items():
            vals = [ec_counts.get(col, 0) for col in active_cols]
            total = sum(vals)
            rows.append((func, vals, total))
        rows.sort(key=lambda r: (-r[2], r[0]))

        func_width = max(len("Function"), max(len(r[0]) for r in rows))
        cols_local = list(active_cols) + ["Total"]
        col_widths_local = [max(8, len(c) + 1) for c in cols_local]
        header = f"--- Thread: {comm} (TID {tid}) ---"
        lines.append(header)
        col_header = f"  {'Function':<{func_width + 2}}" + "".join(
            f"{c:<{w + 1}}" for c, w in zip(cols_local, col_widths_local)
        )
        lines.append(col_header)
        lines.append("  " + "-" * (len(col_header) - 2))

        col_totals_local = [0] * len(cols_local)
        for func, vals, total in rows:
            row_vals = vals + [total]
            for i, v in enumerate(row_vals):
                col_totals_local[i] += v
            row = f"  {func:<{func_width + 2}}" + "".join(
                f"{str(v):<{w + 1}}" for v, w in zip(row_vals, col_widths_local)
            )
            lines.append(row)
        total_row = f"  {'TOTAL':<{func_width + 2}}" + "".join(
            f"{str(v):<{w + 1}}" for v, w in zip(col_totals_local, col_widths_local)
        )
        lines.append(total_row)
        lines.append("")

    # Cross-thread roll-up.
    lines.append("========== Attribution roll-up (thread x function x event class) ==========")
    lines.append("")

    all_rows: list[tuple[str, str, str, list[int], int]] = []  # (comm, tid, func, vals, total)
    for (comm, tid), funcs in per_thread_attr.items():
        for func, ec_counts in funcs.items():
            vals = [ec_counts.get(col, 0) for col in active_cols]
            total = sum(vals)
            all_rows.append((comm, tid, func, vals, total))
    # Group by thread (comm asc, tid asc), descending count within thread.
    all_rows.sort(key=lambda r: (r[0], r[1], -r[4], r[2]))

    # Defensive guard: every funcs dict could in principle be empty, in which
    # case the max() calls below would raise ValueError on the empty sequence.
    if not all_rows:
        lines.append("(no attribution data to roll up)")
        lines.append("")
        return
    thread_label_width = max(23, max(len(f"{c} / {t}") for c, t, *_ in all_rows) + 2)
    func_width = max(len("Function"), max(len(r[2]) for r in all_rows))
    cols_local = list(active_cols) + ["Total"]
    col_widths_local = [max(8, len(c) + 1) for c in cols_local]
    header = (
        f"{'Thread (name / TID)':<{thread_label_width}}"
        f"{'Function':<{func_width + 2}}"
        + "".join(f"{c:<{w + 1}}" for c, w in zip(cols_local, col_widths_local))
    )
    lines.append(header)
    lines.append("-" * len(header))

    grand_totals = [0] * len(cols_local)
    for comm, tid, func, vals, total in all_rows:
        row_vals = vals + [total]
        for i, v in enumerate(row_vals):
            grand_totals[i] += v
        lines.append(
            f"{comm + ' / ' + tid:<{thread_label_width}}"
            f"{func:<{func_width + 2}}"
            + "".join(f"{str(v):<{w + 1}}" for v, w in zip(row_vals, col_widths_local))
        )
    lines.append(
        f"{'TOTAL':<{thread_label_width}}"
        f"{'':<{func_width + 2}}"
        + "".join(f"{str(v):<{w + 1}}" for v, w in zip(grand_totals, col_widths_local))
    )
    lines.append("")


DEFAULT_PERF = "/usr/bin/perf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize per-TID page fault counts (user / kernel) from perf data.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=None,
        help="Path to folder containing perf record files",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_folder",
        type=Path,
        default=None,
        help="Path to folder containing perf record files (alternative to positional argument)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output summary file path (default: <folder>/page_fault_summary.txt)",
    )
    parser.add_argument(
        "--perf-exe",
        type=str,
        default=DEFAULT_PERF,
        help=f"Path to perf executable (default: {DEFAULT_PERF})",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Do not prefix `perf script` invocations with sudo",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders when discovering perf data files",
    )
    args = parser.parse_args()

    folder = args.input_folder if args.input_folder is not None else args.folder
    if folder is None:
        parser.error("input folder is required (positional or -i/--input)")
    folder = folder.resolve()
    if not folder.is_dir():
        print(f"Error: input folder does not exist or is not a directory: {folder}", file=sys.stderr)
        return 1

    perf_files = discover_perf_data_files(folder, recursive=args.recursive)
    if not perf_files:
        print(f"Warning: no perf data files found in {folder}", file=sys.stderr)

    use_sudo = (not args.no_sudo) and (os.geteuid() != 0)

    counts: dict[str, dict[tuple, int]] = {
        USER_COL: {},
        KERNEL_COL: {},
        COMBINED_COL: {},
    }
    attribution: dict[str, dict[tuple, int]] = {
        USER_COL: {},
        KERNEL_COL: {},
        COMBINED_COL: {},
    }

    for data_path, file_tid in perf_files:
        text = run_perf_script(args.perf_exe, data_path, use_sudo)
        if not text:
            continue
        file_totals, file_attribution = parse_perf_script_output(text, file_tid)
        merge_counts(counts, file_totals)
        merge_counts(attribution, file_attribution)

    out_path = args.output if args.output is not None else folder / "page_fault_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(out_path, counts, attribution)
    print(f"Page fault summary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
