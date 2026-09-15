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

# LDPC BLER waterfall tool. Drives cuphy_ex_ldpc_rm (the full PUSCH rate-matching
# decode path) to measure transport-block BLER vs SNR for a chosen decoder algo,
# parameterised by PUSCH config (--n-prb / --nl / --mcs / --mcs-table). Subcommands:
#   find-snr  : search for the SNR that hits a target TB BLER (--target-bler)
#   find-bler : measure TB BLER over a fixed SNR range        (--snr-range START,STEP,END)
# Also parses the binary's "elapsed time in usec" into a per-point latency.
# Usually run via ldpc_test.py (find-snr / find-bler); standalone example:
#   python3 ldpc_rm.py find-snr --n-prb 151 --nl 1 --mcs 27 --mcs-table 2 --algo 55 -n 10 --target-bler 0.10
import argparse
import csv
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass

from ldpc_bench_common import resolve_bin
from pathlib import Path
from typing import Optional


# Shannon SNR + 2.83 dB
TABLE1_TB_BLER10_SNR = [
    -4.70,
    -3.43,
    -2.42,
    -1.10,
    -0.03,
    1.09,
    2.05,
    2.99,
    3.83,
    4.61,
    4.62,
    5.34,
    6.33,
    7.25,
    8.23,
    9.17,
    9.77,
    9.75,
    10.34,
    11.38,
    12.37,
    13.32,
    14.28,
    15.27,
    16.25,
    17.17,
    18.10,
    18.77,
    19.46,
]

TABLE2_TB_BLER10_SNR = [
    -4.70,
    -2.42,
    -0.03,
    2.05,
    3.83,
    5.34,
    6.33,
    7.25,
    8.23,
    9.17,
    9.77,
    10.34,
    11.38,
    12.37,
    13.32,
    14.28,
    15.27,
    16.25,
    17.17,
    18.10,
    18.77,
    19.46,
    20.49,
    21.52,
    22.56,
    23.61,
    24.35,
    25.10,
]

SNR_ANCHORS_BY_MCS_TABLE = {
    1: TABLE1_TB_BLER10_SNR,
    2: TABLE2_TB_BLER10_SNR,
}

FALLBACK_SNR_BRACKET = (0.0, 30.0)
DEFAULT_RV = 0
DEFAULT_FPTYPE = "fp16"
DEFAULT_DECODE_RUNS = 1
DEFAULT_BRACKET_EXPAND_STEP_DB = 0.1
DEFAULT_MAX_BRACKET_EXPANDS = 20


class ArgumentDefaultsRawDescriptionFormatter(argparse.ArgumentDefaultsHelpFormatter,
                                              argparse.RawDescriptionHelpFormatter):
    pass



@dataclass(frozen=True)
class CaseSpec:
    algo: int
    max_iter: int
    fptype: str


@dataclass
class RunResult:
    n_prb: int
    nl: int
    mcs: int
    mcs_table: int
    rv: int
    target_bler: Optional[float]
    num_tbs: int
    case: CaseSpec
    snr: float
    status: int
    tb_bler: Optional[float]
    tb_err: Optional[int]
    tb_total: Optional[int]
    cb_bler: Optional[float]
    cb_err: Optional[int]
    cb_total: Optional[int]
    iter_stats: str
    latency_us: Optional[float]
    bg: Optional[int]
    z: Optional[int]
    p: Optional[int]
    log_path: Path
    command: list[str]


def parse_int_list(text: str, name: str, lo: int, hi: int) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a_text, b_text = item.split("-", 1)
            a = int(a_text)
            b = int(b_text)
            if b < a:
                raise argparse.ArgumentTypeError(f"{name} range must be ascending: {item}")
            values.extend(range(a, b + 1))
        else:
            values.append(int(item))
    values = sorted(set(values))
    for value in values:
        if value < lo or value > hi:
            raise argparse.ArgumentTypeError(f"{name} value {value} is outside [{lo}, {hi}]")
    if not values:
        raise argparse.ArgumentTypeError(f"{name} list is empty")
    return values


def parse_target_blers(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid --target-bler value: {item}") from exc
        if value < 0.01 or value >= 1.0:
            raise argparse.ArgumentTypeError("--target-bler values must be in [0.01, 1)")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--target-bler list is empty")
    return values


def parse_snr_range(items: list[str]) -> list[float]:
    text = " ".join(items).replace(",", " ")
    fields = [field for field in text.split() if field]
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("--snr-range format is START,STEP,END")
    try:
        start, step, end = (float(field) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--snr-range values must be numbers") from exc
    if step == 0.0:
        raise argparse.ArgumentTypeError("--snr-range STEP must be nonzero")
    if start < end and step < 0.0:
        raise argparse.ArgumentTypeError("--snr-range STEP must be positive for ascending ranges")
    if start > end and step > 0.0:
        raise argparse.ArgumentTypeError("--snr-range STEP must be negative for descending ranges")

    values: list[float] = []
    current = start
    epsilon = abs(step) * 1.0e-6
    if step > 0.0:
        while current <= end + epsilon:
            values.append(round(current, 6))
            current += step
    else:
        while current >= end - epsilon:
            values.append(round(current, 6))
            current += step
    if values and abs(values[-1] - end) <= epsilon:
        values[-1] = end
    return values


def parse_case(text: str, default_max_iter: int, default_fptype: str) -> CaseSpec:
    fields = text.split(":")
    if len(fields) > 3:
        raise argparse.ArgumentTypeError("--case format is ALGO[:MAX_ITER[:FPTYPE]]")
    algo = int(fields[0])
    max_iter = int(fields[1]) if len(fields) >= 2 and fields[1] else default_max_iter
    fptype = fields[2] if len(fields) >= 3 and fields[2] else default_fptype
    if max_iter <= 0:
        raise argparse.ArgumentTypeError("MAX_ITER must be positive")
    return CaseSpec(algo=algo, max_iter=max_iter, fptype=fptype)


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "cuPHY").is_dir() and ((path / ".git").exists() or (path / "build.aarch64").exists()):
            return path
    return start


def parse_fraction(pattern: str, output: str) -> tuple[Optional[int], Optional[int], Optional[float]]:
    match = re.search(pattern, output)
    if not match:
        return None, None, None
    return int(match.group(1)), int(match.group(2)), float(match.group(3))


def parse_int_label(label: str, output: str) -> Optional[int]:
    match = re.search(rf"^{re.escape(label)}\s*=\s*([0-9]+)\s*$", output, re.MULTILINE)
    return int(match.group(1)) if match else None


def snr_text(snr: float) -> str:
    return f"{snr:.4f}".rstrip("0").rstrip(".")


def value_tag(value: float) -> str:
    return snr_text(value).replace(".", "p").replace("-", "m")


def args_for_target(args: argparse.Namespace, target_bler: float) -> argparse.Namespace:
    target_args = argparse.Namespace(**vars(args))
    target_args.target_bler = target_bler
    target_args.num_tbs = math.ceil(args.min_tb_errors / target_bler)
    return target_args


def resolve_exe_path(args: argparse.Namespace) -> Path:
    exe = Path(args.exe)
    if exe.is_absolute():
        return exe
    return Path(args.repo_root) / exe


def build_cmd(args: argparse.Namespace, case: CaseSpec, mcs: int, snr: float) -> list[str]:
    cmd = [
        args.exe,
        "--n-prb", str(args.n_prb),
        "--nl", str(args.nl),
        "-m", str(mcs),
        "--mcs-table", str(args.mcs_table),
        "-S", snr_text(snr),
        "-w", str(args.num_tbs),
        "-n", str(case.max_iter),
        "-a", str(case.algo),
        "--fptype", case.fptype,
        "-r", str(DEFAULT_DECODE_RUNS),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.norm is not None:
        cmd.extend(["--norm", str(args.norm)])
    if args.clamp is not None:
        cmd.extend(["-C", str(args.clamp)])
    if args.skip_warmup:
        cmd.append("-k")
    if args.et:
        cmd.append("-x")
    if args.iter_count:
        cmd.append("-y")
    if getattr(args, "reuse_tb", False):
        cmd.append("--reuse-tb")
    return cmd


def parse_output(args: argparse.Namespace, case: CaseSpec, mcs: int, snr: float, status: int,
                 output: str, log_path: Path, cmd: list[str]) -> RunResult:
    tb_err, tb_total, tb_bler = parse_fraction(
        r"info-bit TB BLER\s*=\s*\(\s*([0-9]+)\s*/\s*([0-9]+)\s*\)\s*=\s*([0-9.eE+-]+)",
        output,
    )
    cb_err, cb_total, cb_bler = parse_fraction(
        r"CRC-based CB BLER\s*=\s*\(\s*([0-9]+)\s*/\s*([0-9]+)\s*\)\s*=\s*([0-9.eE+-]+)",
        output,
    )
    latency_us = None
    latency_match = re.search(r"Average .* elapsed time in usec =\s*([0-9.]+)", output)
    if latency_match:
        latency_us = float(latency_match.group(1))

    iter_stats = "NA"
    iter_match = re.search(r"Iteration count min/mean/max\s*=\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)", output)
    if iter_match:
        iter_stats = f"{iter_match.group(1)}/{iter_match.group(2)}/{iter_match.group(3)}"

    return RunResult(
        n_prb=args.n_prb,
        nl=args.nl,
        mcs=mcs,
        mcs_table=args.mcs_table,
        rv=args.rv,
        target_bler=args.target_bler,
        num_tbs=args.num_tbs,
        case=case,
        snr=snr,
        status=status,
        tb_bler=tb_bler,
        tb_err=tb_err,
        tb_total=tb_total,
        cb_bler=cb_bler,
        cb_err=cb_err,
        cb_total=cb_total,
        iter_stats=iter_stats,
        latency_us=latency_us,
        bg=parse_int_label("BG", output),
        z=parse_int_label("Z", output),
        p=parse_int_label("p", output),
        log_path=log_path,
        command=cmd,
    )


def run_once(args: argparse.Namespace, case: CaseSpec, mcs: int, snr: float, log_dir: Path) -> RunResult:
    # target_bler is None for find-bler (fixed SNR sweep, no target); drop it from the tag.
    tbler = "" if args.target_bler is None else f"tbler{value_tag(args.target_bler)}_"
    tag = (
        f"prb{args.n_prb}_nl{args.nl}_tbl{args.mcs_table}_rv{args.rv}_mcs{mcs}_"
        f"a{case.algo}_n{case.max_iter}_{case.fptype}_{tbler}"
        f"w{args.num_tbs}_snr{value_tag(snr)}"
    )
    log_path = log_dir / f"{tag}.log"
    cmd = build_cmd(args, case, mcs, snr)
    proc = subprocess.run(cmd,
                          cwd=getattr(args, "repo_root", None),
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log_path.write_text(proc.stdout)
    result = parse_output(args, case, mcs, snr, proc.returncode, proc.stdout, log_path, cmd)
    if result.tb_bler is None:
        tail = "\n".join(proc.stdout.splitlines()[-12:])
        print(f"  ERROR: no TB BLER parsed at SNR={snr_text(snr)} status={proc.returncode}", file=sys.stderr)
        if tail:
            print(tail, file=sys.stderr)
    return result


def result_distance(result: RunResult, target: float) -> float:
    if result.tb_bler is None:
        return float("inf")
    return abs(result.tb_bler - target)


def result_relative_distance(result: RunResult, target: float) -> float:
    return result_distance(result, target) / target


def bracketed(lo: RunResult, hi: RunResult, target: float) -> bool:
    if lo.tb_bler is None or hi.tb_bler is None:
        return False
    return lo.tb_bler >= target >= hi.tb_bler


def expand_bracket(args: argparse.Namespace, eval_snr, lo: RunResult, hi: RunResult) -> tuple[RunResult, RunResult]:
    expand_step = max(DEFAULT_BRACKET_EXPAND_STEP_DB, args.snr_window)
    for _ in range(DEFAULT_MAX_BRACKET_EXPANDS):
        if bracketed(lo, hi, args.target_bler):
            break
        if lo.tb_bler is None or hi.tb_bler is None:
            break
        if lo.tb_bler > args.target_bler and hi.tb_bler > args.target_bler:
            print(f"  expanding SNR bracket upward by {expand_step:.3f} dB")
            lo = hi
            hi = eval_snr(hi.snr + expand_step)
        elif lo.tb_bler < args.target_bler and hi.tb_bler < args.target_bler:
            print(f"  expanding SNR bracket downward by {expand_step:.3f} dB")
            hi = lo
            lo = eval_snr(lo.snr - expand_step)
        else:
            break
    return lo, hi


def initial_snr_bracket(args: argparse.Namespace, mcs: int) -> tuple[float, float, str]:
    table_anchors = SNR_ANCHORS_BY_MCS_TABLE.get(args.mcs_table)
    if table_anchors is not None and 0 <= mcs < len(table_anchors):
        anchor = table_anchors[mcs]
        return anchor - args.snr_window, anchor + args.snr_window, f"table{args.mcs_table}-anchor({anchor:.2f})"
    lo, hi = FALLBACK_SNR_BRACKET
    return lo, hi, "fallback"


def print_run(result: RunResult) -> None:
    tb = "NA" if result.tb_bler is None else f"{result.tb_bler:.5f}"
    frac = "NA" if result.tb_err is None else f"{result.tb_err}/{result.tb_total}"
    lat = "NA" if result.latency_us is None else f"{result.latency_us:.1f}"
    print(f"  {result.snr:7.3f}  {tb:>8}  {frac:>11}  {result.iter_stats:>10}  {lat:>10}  {result.status:>3}")


def search_one(args: argparse.Namespace, case: CaseSpec, mcs: int, log_dir: Path) -> RunResult:
    cache: dict[float, RunResult] = {}

    def eval_snr(snr: float) -> RunResult:
        key = round(snr, 6)
        if key not in cache:
            result = run_once(args, case, mcs, snr, log_dir)
            cache[key] = result
            print_run(result)
        return cache[key]

    snr_lo, snr_hi, snr_source = initial_snr_bracket(args, mcs)
    print(
        f"\nMCS={mcs} algo={case.algo} max_iter={case.max_iter} fptype={case.fptype} "
        f"target_tb_bler={args.target_bler:.6g} num_tbs={args.num_tbs}"
    )
    print(f"  initial SNR bracket: [{snr_lo:.3f}, {snr_hi:.3f}] from {snr_source}")
    print("      SNR   TB_BLER    TB errors        iter      lat_us  sts")
    lo = eval_snr(snr_lo)
    hi = eval_snr(snr_hi)
    lo, hi = expand_bracket(args, eval_snr, lo, hi)

    stop_reason = "not-bracketed"
    min_snr_tol = args.snr_tol if args.snr_tol > 0.0 else 1.0e-6
    while bracketed(lo, hi, args.target_bler):
        best = min(cache.values(), key=lambda r: result_relative_distance(r, args.target_bler))
        best_rel_dist = result_relative_distance(best, args.target_bler)
        if best_rel_dist <= args.bler_tol:
            stop_reason = f"bler-tol ({best_rel_dist:.6f} <= {args.bler_tol:.6f})"
            break
        bracket_width = hi.snr - lo.snr
        if bracket_width <= min_snr_tol:
            stop_reason = f"snr-tol ({bracket_width:.6f} <= {min_snr_tol:.6f})"
            break
        mid = 0.5 * (lo.snr + hi.snr)
        res = eval_snr(mid)
        if res.tb_bler is None:
            stop_reason = "parse-failed"
            break
        if res.tb_bler >= args.target_bler:
            lo = res
        else:
            hi = res

    if not bracketed(lo, hi, args.target_bler):
        stop_reason = "not-bracketed"
        print("  WARNING: target was not bracketed; using closest sampled point.", file=sys.stderr)

    best = min(cache.values(), key=lambda r: result_relative_distance(r, args.target_bler))
    best_bler = "NA" if best.tb_bler is None else f"{best.tb_bler:.6f}"
    print(f"  stop: {stop_reason}; best SNR={best.snr:.3f}, TB BLER={best_bler}")
    return best


def write_csv(path: Path, rows: list[RunResult]) -> None:
    fieldnames = [
        "n_prb", "nl", "mcs", "mcs_table", "rv", "algo", "max_iter", "fptype",
        "target_bler", "num_tbs", "snr", "tb_bler", "tb_err", "tb_total",
        "cb_bler", "cb_err", "cb_total", "bg", "z", "p", "iter_stats",
        "latency_us", "status", "log_path", "command",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "n_prb": r.n_prb,
                "nl": r.nl,
                "mcs": r.mcs,
                "mcs_table": r.mcs_table,
                "rv": r.rv,
                "algo": r.case.algo,
                "max_iter": r.case.max_iter,
                "fptype": r.case.fptype,
                "target_bler": "" if r.target_bler is None else f"{r.target_bler:.8g}",
                "num_tbs": r.num_tbs,
                "snr": f"{r.snr:.6f}",
                "tb_bler": "" if r.tb_bler is None else f"{r.tb_bler:.8g}",
                "tb_err": "" if r.tb_err is None else r.tb_err,
                "tb_total": "" if r.tb_total is None else r.tb_total,
                "cb_bler": "" if r.cb_bler is None else f"{r.cb_bler:.8g}",
                "cb_err": "" if r.cb_err is None else r.cb_err,
                "cb_total": "" if r.cb_total is None else r.cb_total,
                "bg": "" if r.bg is None else r.bg,
                "z": "" if r.z is None else r.z,
                "p": "" if r.p is None else r.p,
                "iter_stats": r.iter_stats,
                "latency_us": "" if r.latency_us is None else f"{r.latency_us:.3f}",
                "status": r.status,
                "log_path": str(r.log_path),
                "command": " ".join(shlex.quote(x) for x in r.command),
            })


def print_vectors(rows: list[RunResult]) -> None:
    groups: dict[tuple[int, int, str], list[RunResult]] = {}
    for row in rows:
        key = (row.case.algo, row.case.max_iter, row.case.fptype)
        groups.setdefault(key, []).append(row)

    print("\nVectors")
    for key in sorted(groups):
        group_rows = groups[key]
        snr_values = ", ".join(f"{r.snr:.3f}" for r in group_rows)
        mcs_values = ", ".join(str(r.mcs) for r in group_rows)
        tb_bler_values = ", ".join("nan" if r.tb_bler is None else f"{r.tb_bler:.6g}" for r in group_rows)
        target_values = ", ".join(f"{r.target_bler:.6g}" for r in group_rows)
        num_tbs_values = ", ".join(str(r.num_tbs) for r in group_rows)
        print(f"snr = [{snr_values}]")
        print(f"mcs = [{mcs_values}]")
        print(f"tb_bler = [{tb_bler_values}]")
        print(f"target_bler = [{target_values}]")
        print(f"num_tbs = [{num_tbs_values}]")


def print_summary(rows: list[RunResult]) -> None:
    print("\nSummary")
    print("nPRB nl MCS tbl rv   target       w  algo iter fptype     SNR   TB_BLER    TB errors  BG   Z   p    lat_us  log")
    print("---- -- --- --- -- -------- ------- ----- ---- ------- ------- --------- ----------- --- --- --- -------- ----------------")
    for r in rows:
        tb = "NA" if r.tb_bler is None else f"{r.tb_bler:.5f}"
        frac = "NA" if r.tb_err is None else f"{r.tb_err}/{r.tb_total}"
        lat = "NA" if r.latency_us is None else f"{r.latency_us:.1f}"
        bg = "NA" if r.bg is None else str(r.bg)
        z = "NA" if r.z is None else str(r.z)
        p = "NA" if r.p is None else str(r.p)
        print(
            f"{r.n_prb:4d} {r.nl:2d} {r.mcs:3d} {r.mcs_table:3d} {r.rv:2d} "
            f"{r.target_bler:8.4g} {r.num_tbs:7d} {r.case.algo:5d} {r.case.max_iter:4d} {r.case.fptype:<7s} "
            f"{r.snr:7.3f} {tb:>9} {frac:>11} {bg:>3} {z:>3} {p:>3} {lat:>8} {r.log_path}"
        )


FIND_SNR_EXAMPLES = """\
Examples:
  python3 ldpc_rm.py find-snr --n-prb 151 --nl 1 --mcs 27 --mcs-table 2 \\
    --algo 40 -n 10 --target-bler 0.10

  python3 ldpc_rm.py find-snr --n-prb 151 --nl 1 --mcs 27 --mcs-table 2 \\
    --algo 40 -n 10 --target-bler 0.2,0.1,0.01 --min-error-tb 20
"""


FIND_BLER_EXAMPLES = """\
Examples:
  python3 ldpc_rm.py find-bler --n-prb 151 --nl 1 --mcs 27 --mcs-table 2 \\
    --algo 56 -n 10 --snr-range 24.6, 0.2, 25
"""


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def add_allocation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-prb", type=positive_int, required=True, help="Number of allocated PRBs.")
    parser.add_argument("--nl", type=positive_int, required=True, help="Number of PUSCH layers.")
    parser.add_argument("--mcs", required=True,
                        help="MCS list or range, e.g. 27, 20,24,27 or 20-27.")
    parser.add_argument("--mcs-table", type=int, default=2, choices=[1, 2, 3])


def add_decoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--algo", default="40", help="LDPC decoder algorithm list or range.")
    parser.add_argument("-n", "--max-iter", type=positive_int, default=10)


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--et", action="store_true",
                        help="Enable CRC-based early termination in the decoder.")
    parser.add_argument("--iter-count", dest="iter_count", action="store_true",
                        help="Report per-CB iteration counts (min/mean/max). Implied by --et.")
    parser.add_argument("--reuse-tb", action="store_true", help="Reuse one generated TB/codeword for all trials.")
    parser.add_argument("--log-dir", default=f"/tmp/ldpc_rm_snr_sweep_logs_{os.getpid()}")
    parser.add_argument("-o", "--output", default="", help="Optional CSV output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PUSCH LDPC RM BLER utilities.",
        formatter_class=ArgumentDefaultsRawDescriptionFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    find_snr = subparsers.add_parser(
        "find-snr",
        description="Find SNR for target info-bit TB BLER.",
        epilog=FIND_SNR_EXAMPLES,
        formatter_class=ArgumentDefaultsRawDescriptionFormatter,
    )
    add_allocation_args(find_snr)
    add_decoder_args(find_snr)
    find_snr.add_argument("--target-bler", default="0.10", help="Target TB BLER value or comma-separated list.")
    find_snr.add_argument(
        "--min-tb-errors",
        "--min-error-tb",
        dest="min_tb_errors",
        type=positive_int,
        default=100,
        help="Minimum expected TB errors at each target BLER.",
    )
    find_snr.add_argument("--snr-window", type=float, default=0.1, help="Half-width around the 10%% BLER SNR anchor.")
    find_snr.add_argument("--snr-tol", type=nonnegative_float, default=0.001, help="Stop when SNR bracket is this narrow.")
    find_snr.add_argument(
        "--bler-tol",
        type=nonnegative_float,
        default=0.05,
        help="Stop when relative TB BLER error abs(measured-target)/target is this small.",
    )
    add_output_args(find_snr)
    find_snr.set_defaults(func=run_find_snr)

    find_bler = subparsers.add_parser(
        "find-bler",
        description="Measure info-bit TB BLER over a fixed SNR range.",
        epilog=FIND_BLER_EXAMPLES,
        formatter_class=ArgumentDefaultsRawDescriptionFormatter,
    )
    add_allocation_args(find_bler)
    add_decoder_args(find_bler)
    find_bler.add_argument(
        "--snr-range",
        nargs="+",
        required=True,
        help="Fixed SNR sweep as START,STEP,END, e.g. 24.6,0.2,25 or 24.6, 0.2, 25.",
    )
    find_bler.add_argument("--num-tbs", type=positive_int, default=1000, help="Transport blocks per SNR point.")
    add_output_args(find_bler)
    find_bler.set_defaults(func=run_find_bler)

    return parser


def make_common_args(ns: argparse.Namespace) -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    args = argparse.Namespace(**vars(ns))
    args.repo_root = str(repo_root)
    args.exe = resolve_bin(None, "cuphy_ex_ldpc_rm")
    args.rv = DEFAULT_RV
    args.fptype = DEFAULT_FPTYPE
    args.decode_runs = DEFAULT_DECODE_RUNS
    args.norm = None
    args.clamp = None
    args.et = bool(args.et)
    # ET reports its own iteration spread; asking for ET implies wanting the counts.
    args.iter_count = bool(args.iter_count or args.et)
    args.skip_warmup = True
    args.reuse_tb = bool(args.reuse_tb)
    args.mcs_values = parse_int_list(args.mcs, "MCS", 0, 31)
    args.cases = [CaseSpec(algo, args.max_iter, DEFAULT_FPTYPE) for algo in parse_int_list(args.algo, "algo", 0, 100000)]
    return args


def check_exe(args: argparse.Namespace) -> bool:
    exe = resolve_exe_path(args)
    if not exe.exists():
        print(f"ERROR: executable not found: {exe}", file=sys.stderr)
        return False
    return True


def dry_run_find_snr(args: argparse.Namespace) -> None:
    for target_bler in args.target_blers:
        target_args = args_for_target(args, target_bler)
        print(f"# target_tb_bler={target_bler:.6g} num_tbs={target_args.num_tbs}")
        for mcs in args.mcs_values:
            snr_lo, _, _ = initial_snr_bracket(target_args, mcs)
            for case in args.cases:
                cmd = build_cmd(target_args, case, mcs, snr_lo)
                print(" ".join(shlex.quote(x) for x in cmd))


def run_find_snr(ns: argparse.Namespace) -> int:
    start_time = time.monotonic()
    args = make_common_args(ns)
    args.target_blers = parse_target_blers(args.target_bler)
    args.snr_values = None

    if not check_exe(args):
        return 1
    if args.dry_run:
        dry_run_find_snr(args)
        print(f"\nTotal elapsed time: {format_elapsed(time.monotonic() - start_time)}")
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    target_desc = ", ".join(f"{target:.6g}" for target in args.target_blers)
    print(
        f"Config: n_prb={args.n_prb} nl={args.nl} mcs={args.mcs_values} "
        f"mcs_table={args.mcs_table} target_tb_bler=[{target_desc}] min_error_tb={args.min_tb_errors}"
    )
    for target_bler in args.target_blers:
        target_args = args_for_target(args, target_bler)
        print(
            f"Target {target_bler:.6g}: num_tbs={target_args.num_tbs}, "
            f"TB BLER granularity={1.0 / target_args.num_tbs:.6f} per TB error"
        )
    print(f"Logs: {log_dir}")

    rows: list[RunResult] = []
    for target_bler in args.target_blers:
        target_args = args_for_target(args, target_bler)
        for mcs in args.mcs_values:
            for case in args.cases:
                rows.append(search_one(target_args, case, mcs, log_dir))

    print_summary(rows)
    print_vectors(rows)
    if args.output:
        write_csv(Path(args.output), rows)
        print(f"\nWrote CSV: {args.output}")
    print(f"\nTotal elapsed time: {format_elapsed(time.monotonic() - start_time)}")
    return 0


def dry_run_find_bler(args: argparse.Namespace) -> None:
    print(f"# fixed_snr_sweep num_tbs={args.num_tbs}")
    for mcs in args.mcs_values:
        for case in args.cases:
            for snr in args.snr_values:
                cmd = build_cmd(args, case, mcs, snr)
                print(" ".join(shlex.quote(x) for x in cmd))


def print_bler_summary(rows: list[RunResult]) -> None:
    print("\nSummary")
    print("nPRB nl MCS tbl  algo iter     SNR   TB_BLER    TB errors  BG   Z   p    lat_us  log")
    print("---- -- --- --- ----- ---- ------- --------- ----------- --- --- --- -------- ----------------")
    for r in rows:
        tb = "NA" if r.tb_bler is None else f"{r.tb_bler:.5f}"
        frac = "NA" if r.tb_err is None else f"{r.tb_err}/{r.tb_total}"
        lat = "NA" if r.latency_us is None else f"{r.latency_us:.1f}"
        bg = "NA" if r.bg is None else str(r.bg)
        z = "NA" if r.z is None else str(r.z)
        p = "NA" if r.p is None else str(r.p)
        print(
            f"{r.n_prb:4d} {r.nl:2d} {r.mcs:3d} {r.mcs_table:3d} "
            f"{r.case.algo:5d} {r.case.max_iter:4d} {r.snr:7.3f} "
            f"{tb:>9} {frac:>11} {bg:>3} {z:>3} {p:>3} {lat:>8} {r.log_path}"
        )


def print_bler_vectors(rows: list[RunResult]) -> None:
    print("\nVectors")
    groups: dict[tuple[int, int], list[RunResult]] = {}
    for row in rows:
        groups.setdefault((row.case.algo, row.case.max_iter), []).append(row)
    for key in sorted(groups):
        group_rows = groups[key]
        snr_values = ", ".join(f"{r.snr:.3f}" for r in group_rows)
        mcs_values = ", ".join(str(r.mcs) for r in group_rows)
        tb_bler_values = ", ".join("nan" if r.tb_bler is None else f"{r.tb_bler:.6g}" for r in group_rows)
        print(f"snr = [{snr_values}]")
        print(f"mcs = [{mcs_values}]")
        print(f"tb_bler = [{tb_bler_values}]")


def run_find_bler(ns: argparse.Namespace) -> int:
    start_time = time.monotonic()
    args = make_common_args(ns)
    args.snr_values = parse_snr_range(args.snr_range)
    args.target_bler = None   # fixed SNR sweep: no target BLER

    if not check_exe(args):
        return 1
    if args.dry_run:
        dry_run_find_bler(args)
        print(f"\nTotal elapsed time: {format_elapsed(time.monotonic() - start_time)}")
        return 0

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    snr_desc = ", ".join(snr_text(snr) for snr in args.snr_values)
    print(
        f"Config: n_prb={args.n_prb} nl={args.nl} mcs={args.mcs_values} "
        f"mcs_table={args.mcs_table} snr=[{snr_desc}] num_tbs={args.num_tbs}"
    )
    print(f"Logs: {log_dir}")

    rows: list[RunResult] = []
    for mcs in args.mcs_values:
        for case in args.cases:
            print(f"\nMCS={mcs} algo={case.algo} max_iter={case.max_iter} num_tbs={args.num_tbs}")
            print("      SNR   TB_BLER    TB errors        iter      lat_us  sts")
            for snr in args.snr_values:
                result = run_once(args, case, mcs, snr, log_dir)
                print_run(result)
                rows.append(result)

    print_bler_summary(rows)
    print_bler_vectors(rows)
    if args.output:
        write_csv(Path(args.output), rows)
        print(f"\nWrote CSV: {args.output}")
    print(f"\nTotal elapsed time: {format_elapsed(time.monotonic() - start_time)}")
    return 0


def format_elapsed(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {secs:.1f} s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)} h {int(minutes)} min {secs:.1f} s"


def main() -> int:
    parser = build_parser()
    try:
        args = parser.parse_args()
        return args.func(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

