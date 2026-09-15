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

# Summarize per-syscall-type counts per DL/UL worker thread from a bpftrace map
# dump produced by bpftrace_trace_workers.sh (named-wildcard form).
#
# Expected input lines (bpftrace prints the @syscalls map on exit):
#   @syscalls[<comm>, tracepoint:syscalls:sys_enter_<name>]: <count>
# The raw_syscalls single-probe form (@syscalls[<comm>, <numeric id>]) is also
# accepted; the numeric id is used verbatim as the "syscall" label.
#
# Output: a table with one row per syscall type, one column per worker (comm),
# a Total column (sorted by Total descending), and a TOTAL row.
#
# Pass/fail: PASS if any syscalls were captured; FAIL if the capture is empty
# (no @syscalls entries). Exit code mirrors the verdict (0 = PASS, 1 = FAIL).
#
# Usage:
#   python3 bpftrace_syscall_summary.py -i <dump_file_or_folder> [-o <out>]

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# @syscalls[comm, probe]: count   (comm has no comma; probe runs up to "]:")
MAP_LINE_RE = re.compile(r"^@syscalls\[\s*([^,]+?)\s*,\s*(.+?)\s*\]:\s*(\d+)\s*$")
SYSCALL_PREFIX = "tracepoint:syscalls:sys_enter_"


def syscall_name(probe: str) -> str:
    """Return the bare syscall name from a bpftrace probe label.

    Args:
        probe: bpftrace probe string, e.g.
            ``tracepoint:syscalls:sys_enter_read`` or a numeric raw id.

    Returns:
        Syscall name with the ``tracepoint:syscalls:sys_enter_`` prefix removed,
        or ``probe`` unchanged when the prefix is absent.

    Raises:
        None.

    Examples:
        >>> syscall_name("tracepoint:syscalls:sys_enter_read")
        'read'
    """
    p = probe.strip()
    if p.startswith(SYSCALL_PREFIX):
        return p[len(SYSCALL_PREFIX):]
    return p


def parse_dump(
    text: str,
    counts: dict[str, dict[str, int]],
    workers: set[str],
) -> int:
    """Parse bpftrace @syscalls map lines into per-syscall, per-worker counts.

    Args:
        text: Raw bpftrace map dump text.
        counts: Mutable map ``syscall -> {worker_comm -> count}`` updated in place.
        workers: Mutable set of worker thread names (comm) updated in place.

    Returns:
        Number of ``@syscalls[...]`` map entries parsed from ``text``.

    Raises:
        ValueError: If a map line has a non-integer count field.

    Examples:
        >>> c: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        >>> w: set[str] = set()
        >>> parse_dump("@syscalls[DlPhyDriver0, tracepoint:syscalls:sys_enter_read]: 3\\n", c, w)
        1
    """
    n = 0
    for raw in text.splitlines():
        m = MAP_LINE_RE.match(raw.strip())
        if not m:
            continue
        comm, probe, cnt = m.group(1).strip(), m.group(2).strip(), int(m.group(3))
        counts[syscall_name(probe)][comm] += cnt
        workers.add(comm)
        n += 1
    return n


def discover_inputs(path: Path, recursive: bool) -> list[Path]:
    """Resolve a bpftrace dump file or directory to a list of input files.

    Args:
        path: A single dump file or a directory containing dump files.
        recursive: When ``path`` is a directory, search subdirectories recursively.

    Returns:
        Sorted list of files to parse; empty when ``path`` does not exist as a
        file or directory.

    Raises:
        OSError: Propagated from ``Path.is_file``, ``Path.is_dir``, ``iterdir``,
            or ``rglob`` on permission or I/O errors.

    Examples:
        >>> discover_inputs(Path("/tmp/bpftrace_syscalls.txt"), False)
        [PosixPath('/tmp/bpftrace_syscalls.txt')]
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        it = path.rglob("*") if recursive else path.iterdir()
        return sorted(p for p in it if p.is_file())
    return []


def write_summary(
    out_path: Path,
    counts: dict[str, dict[str, int]],
    workers: set[str],
) -> int:
    """Write the syscall-by-worker summary table and verdict to a file.

    Args:
        out_path: Destination path for the summary text file.
        counts: ``syscall -> {worker_comm -> count}`` aggregation.
        workers: Set of worker thread names (comm) used as table columns.

    Returns:
        Grand total syscall count across all workers and syscall types.

    Raises:
        OSError: If the parent directory cannot be created or the file cannot
            be written.

    Examples:
        >>> write_summary(Path("/tmp/out.txt"), {"read": {"DlPhyDriver0": 2}}, {"DlPhyDriver0"})
        2
    """
    worker_cols = sorted(workers)  # DlPhyDriver* sort before UlPhyDriver* naturally
    syscall_total = {s: sum(w.values()) for s, w in counts.items()}
    rows = sorted(counts.keys(), key=lambda s: (-syscall_total[s], s))
    grand_total = sum(syscall_total.values())

    # Column widths sized to content.
    syscall_w = max([len("Syscall")] + [len(s) for s in rows]) + 2
    col_totals = {w: sum(counts[s].get(w, 0) for s in rows) for w in worker_cols}

    def worker_col_width(w: str) -> int:
        vals = [counts[s].get(w, 0) for s in rows] + [col_totals[w]]
        return max([len(w)] + [len(str(v)) for v in vals]) + 2

    worker_w = {w: worker_col_width(w) for w in worker_cols}
    total_w = max([len("Total"), len(str(grand_total))] + [len(str(t)) for t in syscall_total.values()]) + 2

    lines = [
        "Syscall counts per DL/UL worker thread (bpftrace all-syscalls capture)",
        "Generated by bpftrace_syscall_summary.py",
        "",
    ]
    header = (
        f"{'Syscall':<{syscall_w}}"
        + "".join(f"{w:<{worker_w[w]}}" for w in worker_cols)
        + f"{'Total':<{total_w}}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in rows:
        row = f"{s:<{syscall_w}}"
        for w in worker_cols:
            row += f"{counts[s].get(w, 0):<{worker_w[w]}}"
        row += f"{syscall_total[s]:<{total_w}}"
        lines.append(row)
    lines.append("-" * len(header))
    total_row = (
        f"{'TOTAL':<{syscall_w}}"
        + "".join(f"{col_totals[w]:<{worker_w[w]}}" for w in worker_cols)
        + f"{grand_total:<{total_w}}"
    )
    lines.append(total_row)
    lines.append("")
    lines.append("========== Verdict ==========")
    lines.append("")
    lines.append("PASS" if grand_total > 0 else "FAIL (empty capture: no syscalls recorded)")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return grand_total


def main() -> int:
    """CLI entry point: aggregate bpftrace dumps and emit a PASS/FAIL verdict.

    Args:
        None. Reads ``-i/--input``, ``-o/--output``, and ``-r/--recursive`` from
        ``sys.argv`` via ``argparse``.

    Returns:
        ``0`` when at least one syscall was captured (PASS), ``1`` on empty
        capture (FAIL) or missing/invalid input.

    Raises:
        SystemExit: Not raised directly; callers typically use ``sys.exit(main())``.

    Examples:
        ``python3 bpftrace_syscall_summary.py -i /tmp/bpftrace_syscalls.txt``
    """
    parser = argparse.ArgumentParser(
        description="Summarize per-syscall-type counts per worker from a bpftrace map dump."
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="bpftrace map dump file, or a folder of dumps to aggregate",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output summary file (default: <input dir>/bpftrace_syscall_summary.txt)",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="When input is a folder, search it recursively",
    )
    args = parser.parse_args()

    in_path = args.input.resolve()
    if not in_path.exists():
        print(f"Error: input does not exist: {in_path}", file=sys.stderr)
        return 1

    files = discover_inputs(in_path, args.recursive)
    if not files:
        print(f"Error: no input files found under {in_path}", file=sys.stderr)
        return 1

    if args.output is not None:
        out_path = args.output.resolve()
    else:
        base = in_path if in_path.is_dir() else in_path.parent
        out_path = base / "bpftrace_syscall_summary.txt"

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    workers: set[str] = set()
    parsed = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            sys.stderr.write(f"Could not read {f}: {e}\n")
            continue
        parsed += parse_dump(text, counts, workers)

    grand_total = write_summary(out_path, counts, workers)
    verdict = "PASS" if grand_total > 0 else "FAIL"
    print(f"Parsed {parsed} map entries from {len(files)} file(s).")
    print(f"Summary written to {out_path}")
    print(f"Verdict: {verdict} (total syscalls captured: {grand_total})")
    return 0 if grand_total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
