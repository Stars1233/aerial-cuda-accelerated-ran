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

# Summarize futex call counts per CUDA API per thread from perf record data.
# MULTI-THREAD VERSION: each perf data file may contain events from multiple
# threads; each event is attributed to the (comm, tid) from its own event line.
#
# Usage:
#   python3 futex_cuda_summary_multi_thread.py -i <folder_path> [options]
#   python3 futex_cuda_summary_multi_thread.py <folder_path> [options]
#
# Walks the input folder for perf record files (or .txt with --text-only), runs
# "perf script -i <file>" on each (with sudo by default), parses sys_enter_futex
# events and their stack traces, attributes each futex to the (comm, tid) of that
# event and to the topmost CUDA API in the stack, and writes a combined per-thread summary.
# Multiple files in the folder are all processed and aggregated into one summary.

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# Stack line: optional spaces, hex addr, space, symbol (e.g. "cuEventRecord+0x30" or "[unknown]"), space, (path)
STACK_LINE_RE = re.compile(
    r"^\s+([0-9a-fA-F]+)\s+(.+?)\s+\((.+)\)\s*$"
)
# Symbol part: strip +0x... suffix to get API name
SYMBOL_STRIP_OFFSET = re.compile(r"^(.+?)\+0x[0-9a-fA-F]*$")

# Paths that indicate a CUDA API frame (driver, runtime, etc.)
CUDA_LIB_PATTERNS = ("libcuda.so", "libcudart.so", "libcublas", "libcusolver", "libcufft", "libcurand", "libcusparse", "libnvrtc")


def is_cuda_lib(path: str) -> bool:
    return any(p in path for p in CUDA_LIB_PATTERNS)


def symbol_name(raw: str) -> str:
    """Return symbol without +0x... offset."""
    m = SYMBOL_STRIP_OFFSET.match(raw.strip())
    return m.group(1).strip() if m else raw.strip()


def parse_perf_script_output(
    text: str, file_tid: str | None
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[tuple[str, str], int]]:
    """
    Parse 'perf script' output. Each futex event is attributed to the (comm, tid)
    from that event's line. Returns (per_thread_cuda_counts, per_thread_total_futex).
    file_tid: optional TID fallback when event line has no tid (e.g. from filename).
    """
    # Per-thread: (comm, tid) -> { api -> count }
    thread_cuda_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Per-thread: (comm, tid) -> total futex count
    thread_total_futex: dict[tuple[str, str], int] = defaultdict(int)
    current_event_line: str | None = None
    stack_lines: list[str] = []

    def flush_event():
        nonlocal current_event_line, stack_lines
        if not current_event_line:
            return
        # Parse event line: "comm tid [cpu] timestamp: event: ..."
        parts = current_event_line.split()
        comm = "unknown"
        tid = file_tid or "unknown"
        if len(parts) >= 2:
            comm = parts[0]
            tid = parts[1]
        key = (comm, tid)
        thread_total_futex[key] += 1
        if stack_lines:
            for line in stack_lines:
                m = STACK_LINE_RE.match(line)
                if not m:
                    continue
                _addr, sym_part, path = m.group(1), m.group(2), m.group(3)
                if sym_part.strip() == "[unknown]":
                    continue
                if is_cuda_lib(path):
                    api = symbol_name(sym_part)
                    if api:
                        thread_cuda_counts[key][api] += 1
                    break  # count only the topmost CUDA frame per futex
        current_event_line = None
        stack_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        # Check indentation first: indented lines are stack frames (e.g. __tracepoint_sys_enter_futex)
        # and must not be mistaken for a new event header.
        if current_event_line is not None and (line.startswith(" ") or line.startswith("\t")):
            stack_lines.append(line)
            continue
        if "sys_enter_futex" in line or "syscalls:sys_enter_futex" in line:
            flush_event()
            current_event_line = line.strip()
            continue
        if current_event_line is not None:
            flush_event()

    flush_event()
    # Convert to plain dicts
    per_thread_counts = {k: dict(v) for k, v in thread_cuda_counts.items()}
    per_thread_totals = dict(thread_total_futex)
    return (per_thread_counts, per_thread_totals)


def discover_perf_data_files(folder: Path, recursive: bool = False) -> list[tuple[Path, str | None]]:
    """Return list of (file_path, optional_tid_from_filename)."""
    results: list[tuple[Path, str | None]] = []
    if not folder.is_dir():
        return results
    if recursive:
        it = folder.rglob("*")
    else:
        it = folder.iterdir()
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
            suffix = name.split(".data.")[-1]
            if suffix.isdigit():
                tid_from_name = suffix
        elif "." in name:
            # Only treat as perf data if suffix looks like a TID (numeric); skip .py, .txt, etc.
            suffix = name.split(".")[-1]
            if not suffix.isdigit():
                continue
            tid_from_name = suffix
        else:
            continue
        results.append((p, tid_from_name))
    return results


def discover_text_files(folder: Path, recursive: bool = False) -> list[tuple[Path, str | None]]:
    """Return list of (file_path, optional_tid_from_filename) for perf script text files (.txt)."""
    results: list[tuple[Path, str | None]] = []
    if not folder.is_dir():
        return results
    if recursive:
        it = folder.rglob("*.txt")
    else:
        it = (p for p in folder.iterdir() if p.is_file() and p.suffix == ".txt")
    for p in it:
        if not p.is_file():
            continue
        name = p.name
        tid_from_name: str | None = None
        if ".data." in name:
            suffix = name.split(".data.")[-1].replace(".txt", "").split(".")[0]
            if suffix.isdigit():
                tid_from_name = suffix
        elif "_" in name:
            parts = name.replace(".txt", "").split("_")
            if parts[-1].isdigit():
                tid_from_name = parts[-1]
        results.append((p, tid_from_name))
    return results


def run_perf_script(perf_exe: str, data_path: Path, use_sudo: bool) -> str:
    """Run 'perf script -i <data_path>' and return stdout. Uses --force so perf reads files not owned by current user (e.g. when using sudo)."""
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


def build_global_api_set(per_thread: dict[tuple[str, str], dict[str, int]]) -> list[str]:
    """Sorted list of all CUDA APIs that appear in any thread."""
    apis: set[str] = set()
    for counts in per_thread.values():
        apis.update(counts.keys())
    return sorted(apis)


def write_summary(
    out_path: Path,
    per_thread: dict[tuple[str, str], dict[str, int]],
    total_futex_per_thread: dict[tuple[str, str], int],
) -> None:
    """Write per-thread table: Thread (name + TID) x CUDA API -> futex count."""
    apis = build_global_api_set(per_thread)
    lines: list[str] = [
        "Futex call counts per CUDA API, per thread (multi-thread: one file may contain many threads)",
        "Generated by futex_cuda_summary_multi_thread.py",
        "",
    ]
    thread_keys = sorted(per_thread.keys(), key=lambda k: (k[1], k[0]))

    for (comm, tid) in thread_keys:
        counts = per_thread[(comm, tid)]
        total_raw = total_futex_per_thread.get((comm, tid), 0)
        cuda_total = sum(counts.values())
        non_cuda = total_raw - cuda_total
        lines.append(f"--- Thread: {comm} (TID {tid}) ---")
        lines.append(f"  Total futex events: {total_raw}  (CUDA-attributed: {cuda_total}, Non-CUDA: {non_cuda})")
        if apis:
            col_width = max(len(a) for a in apis) + 2
            header = "  " + "".join(f"{a:<{col_width}}" for a in apis)
            lines.append(header)
            row = "  " + "".join(f"{counts.get(a, 0):<{col_width}}" for a in apis)
            lines.append(row)
        else:
            lines.append("  (no CUDA API frames found in stacks)")
        lines.append("")

    lines.append("========== Summary table (thread x CUDA API + CUDA + Non-CUDA + Total) ==========")
    lines.append("")
    if thread_keys:
        cuda_col = "CUDA"
        non_cuda_col = "Non-CUDA"
        total_col = "Total"
        all_cols = apis + [cuda_col, non_cuda_col, total_col]
        label_width = max(23, max(len(f"{c} / {t}") for c, t in thread_keys) + 2)
        header = f"{'Thread (name / TID)':<{label_width}}" + "".join(f"{c:<{max(8, len(c))+1}}" for c in all_cols)
        lines.append(header)
        lines.append("-" * len(header))
        # Accumulate column totals for the final row
        col_totals: list[int] = [0] * len(all_cols)
        for (comm, tid) in thread_keys:
            counts = per_thread[(comm, tid)]
            total_raw = total_futex_per_thread.get((comm, tid), 0)
            cuda_total = sum(counts.values())
            non_cuda = total_raw - cuda_total
            row_vals = [counts.get(a, 0) for a in apis] + [cuda_total, non_cuda, total_raw]
            for i, v in enumerate(row_vals):
                col_totals[i] += v
            row_vals_str = [str(v) for v in row_vals]
            row_label = f"{comm} / {tid}"
            row = f"{row_label:<{label_width}}" + "".join(f"{v:<{max(8, len(c))+1}}" for v, c in zip(row_vals_str, all_cols))
            lines.append(row)
        # Totals row
        total_row_vals = [str(t) for t in col_totals]
        total_row = f"{'TOTAL':<{label_width}}" + "".join(f"{v:<{max(8, len(c))+1}}" for v, c in zip(total_row_vals, all_cols))
        lines.append(total_row)
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


DEFAULT_PERF = "/usr/bin/perf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize futex call counts per CUDA API per thread (multi-thread per file)."
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
        type=Path,
        default=None,
        dest="input_folder",
        help="Path to folder containing perf record files (alternative to positional folder)",
    )
    parser.add_argument(
        "-p", "--perf",
        type=str,
        default=DEFAULT_PERF,
        dest="perf_executable",
        help=f"Path to perf executable (default: {DEFAULT_PERF})",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output summary file (default: <folder>/futex_cuda_summary_multi_thread.txt)",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Do not run 'perf script' with sudo",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for perf data files recursively under folder",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Treat files as already-converted perf script text (e.g. .txt); do not run perf script",
    )
    args = parser.parse_args()

    folder = args.input_folder or args.folder
    if folder is None:
        print("Error: input folder is required (use -i/--input or positional argument)", file=sys.stderr)
        return 1
    folder = Path(folder).resolve()
    if not folder.is_dir():
        print(f"Error: folder does not exist or is not a directory: {folder}", file=sys.stderr)
        return 1

    perf_exe: str = args.perf_executable or DEFAULT_PERF
    if not args.text_only:
        if not os.path.isabs(perf_exe) and "/" not in perf_exe:
            which = subprocess.run(["which", perf_exe], capture_output=True, text=True)
            if which.returncode == 0 and which.stdout.strip():
                perf_exe = which.stdout.strip()
        if not os.path.isfile(perf_exe) or not os.access(perf_exe, os.X_OK):
            print(f"Error: perf executable not found or not executable: {perf_exe}", file=sys.stderr)
            return 1

    out_path = args.output
    if out_path is None:
        out_path = folder / "futex_cuda_summary_multi_thread.txt"
    else:
        out_path = out_path.resolve()

    if args.text_only:
        files = discover_text_files(folder, recursive=args.recursive)
        if not files:
            print("No text files found under", folder, file=sys.stderr)
            return 1
    else:
        files = discover_perf_data_files(folder, recursive=args.recursive)
        if not files:
            print("No perf data files found under", folder, file=sys.stderr)
            return 1

    per_thread: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_futex_per_thread: dict[tuple[str, str], int] = defaultdict(int)

    print(f"Processing {len(files)} file(s) from {folder} ...")
    for data_path, file_tid in files:
        if args.text_only:
            try:
                stdout = data_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                sys.stderr.write(f"Could not read {data_path}: {e}\n")
                continue
        else:
            stdout = run_perf_script(perf_exe, data_path, use_sudo=not args.no_sudo)
        file_per_thread_counts, file_per_thread_totals = parse_perf_script_output(stdout or "", file_tid)
        for key, counts in file_per_thread_counts.items():
            for api, count in counts.items():
                per_thread[key][api] += count
        for key, total in file_per_thread_totals.items():
            total_futex_per_thread[key] += total
            per_thread[key]  # ensure thread appears even if no CUDA attribution

    per_thread = {k: dict(v) for k, v in per_thread.items()}
    write_summary(out_path, per_thread, dict(total_futex_per_thread))
    print(f"Summary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
