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

# Per-iteration APP (LLR) check sweep for the Z=384 BG1 LDPC decoders.
#
# Reads the sweep scope (which algos to cover, over which p) from
# ldpc2_decoder_bands.yaml, auto-discovers a test vector per parity-node count p
# by globbing testVectors/, runs cuphy_ex_ldpc_rm --llr_check for each (algo, p),
# and reports a table. Failures are cross-checked against each algo's
# known_llr_fail list, so a NEW regression is called out distinctly from an
# already-documented marginal.
#
# Usage:
#   python3 llr_check_sweep.py                    # full sweep, all listed algos
#   python3 llr_check_sweep.py --algo 40          # one algo
#   python3 llr_check_sweep.py --algo 35 40       # a subset
#
# Exit code: 0 if no NEW failures (known marginals allowed), 1 otherwise.

import argparse
import glob
import os
import re
import subprocess
import sys
import time

from ldpc_bench_common import resolve_bin

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# LDPC decode iterations for each --llr_check run. A harness run knob (how many
# iterations to compare), NOT a pass/fail gate -- the p99err/SNR gates live in the
# binary (ldpc_interm_check.hpp) and are left at the binary's own defaults.
MAX_ITER = 10

DEFAULT_YAML = os.path.join(HERE, "ldpc2_decoder_bands.yaml")
DEFAULT_TVDIR = os.path.join(REPO, "testVectors")


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def find_tv(tvdir: str, bg: int, z: int, p: int) -> str | None:
    """Return one TV path for this base graph / lifting size / parity count
    (sorted-first), or None."""
    hits = sorted(glob.glob(os.path.join(tvdir, f"*_bg{bg}_Z{z}_p{p}.h5")))
    return hits[0] if hits else None


# The binary echoes the config it actually decoded, e.g.
#   Running decoder: algo=40  BG=1  Zc=384  mb=21  C=2  maxItr=10  clamp=32.0
DECODED_CFG_RE = re.compile(r"BG=(\d+)\s+Zc=(\d+)\s+mb=(\d+)")


def run_one(binpath: str, tv: str | None, algo: int,
            bg: int, z: int, p: int) -> str:
    """Run a single --llr_check. Returns 'PASS' | 'FAIL' | 'ERR' | 'NO-TV'.
    Gates are omitted so the binary applies its own ldpc_interm_check.hpp defaults.
    Vectors are found by filename but decoded from their contents, so the two are
    cross-checked: a mislabelled file is an ERR."""
    if tv is None:
        return "NO-TV"
    cmd = [
        binpath, "-i", tv, "--llr_check", "-a", str(algo),
        "-n", str(MAX_ITER),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return "ERR"
    blob = res.stdout + res.stderr
    if "EXCEPTION" in blob:
        return "ERR"
    m = DECODED_CFG_RE.search(blob)
    if m and (int(m.group(1)), int(m.group(2)), int(m.group(3))) != (bg, z, p):
        print(f"    {os.path.basename(tv)}: name says BG{bg} Z{z} p{p}, "
              f"contents are BG{m.group(1)} Z{m.group(2)} p{m.group(3)}", file=sys.stderr)
        return "ERR"
    if "LLR_CHECK PASS" in blob:
        return "PASS"
    return "FAIL"


def sweep(doc: dict, binpath: str, tvdir: str, algos_filter: list[int] | None
          ) -> tuple[list[dict], dict[tuple[int, int], str], list[tuple[int, int]], float, tuple[int, int]]:
    block = doc["llr_check_sweep"]["z384"]
    bg, z = block["BG"], block["Z"]
    scope = block["algos"]
    if algos_filter:
        listed = [a["algo"] for a in scope]
        scope = [a for a in scope if a["algo"] in algos_filter]
        if not scope:
            raise SystemExit(
                f"--algo {algos_filter} matches nothing; listed algos are {listed}")

    results = {}          # (algo, p) -> status
    new_fail = []         # (algo, p) not in known_llr_fail
    t0 = time.time()
    for a in scope:
        algo = a["algo"]
        known = set(a.get("known_llr_fail", []))
        for p in range(a["p_min"], a["p_max"] + 1):
            tv = find_tv(tvdir, bg, z, p)
            st = run_one(binpath, tv, algo, bg, z, p)
            tag = ""
            if st == "FAIL":
                if p in known:
                    tag = " (known)"
                else:
                    tag = " (NEW!)"
                    new_fail.append((algo, p))
            results[(algo, p)] = st
            print(f"  a{algo} p{p}: {st}{tag}")
    elapsed = time.time() - t0
    return scope, results, new_fail, elapsed, (bg, z)


def print_summary(scope: list[dict], results: dict[tuple[int, int], str],
                  elapsed: float) -> None:
    print(f"\n=== SUMMARY ({elapsed:.1f}s) ===")
    for a in scope:
        algo = a["algo"]
        ps = range(a["p_min"], a["p_max"] + 1)
        npass = sum(1 for p in ps if results[(algo, p)] == "PASS")
        fails = [p for p in ps if results[(algo, p)] == "FAIL"]
        errs = [p for p in ps if results[(algo, p)] == "ERR"]
        notv = [p for p in ps if results[(algo, p)] == "NO-TV"]
        total = len(ps)
        line = f"  a{algo} p{a['p_min']}..{a['p_max']}: {npass}/{total} PASS"
        if fails:
            line += f"  FAIL={fails}"
        if errs:
            line += f"  ERR={errs}"
        if notv:
            line += f"  NO-TV={notv}"
        print(line)


EXAMPLES = """
Examples:
  # full per-iteration APP check, all listed Z=384 decoders vs MATLAB TVs:
  python3 llr_check_sweep.py
  # restrict to a subset:
  python3 llr_check_sweep.py --algo 35 40
"""


def main() -> int:
    """Run the per-iteration APP check sweep from command-line arguments.

    Reads the sweep scope from ``ldpc2_decoder_bands.yaml``, finds one test
    vector per parity-node count, runs ``cuphy_ex_ldpc_rm --llr_check`` for each
    (algo, p), and prints a per-algo summary.

    Args:
        None. All inputs are read from ``sys.argv`` through ``argparse``.

    Returns:
        Process exit code: ``0`` if no NEW failures (known marginals allowed),
        ``1`` if any vector was mislabelled or errored, if nothing was checked
        at all, or if a NEW failure appeared.

    Raises:
        SystemExit: On argument errors or ``--help`` (from ``argparse``), when
            ``--algo`` matches nothing in the YAML, or when the binary cannot be
            resolved (via ``resolve_bin``).
        FileNotFoundError: If ``--yaml`` does not exist.

    Examples:
        >>> import sys
        >>> sys.argv = ["llr_check_sweep.py", "--algo", "40"]
        >>> main()  # doctest: +SKIP
        0
    """
    ap = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLES,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--bin", default=None, help="path to cuphy_ex_ldpc_rm (else autodetect)")
    ap.add_argument("--testvectors-dir", default=DEFAULT_TVDIR)
    ap.add_argument("--algo", type=int, nargs="*", default=None,
                    help="restrict to these algos (default: all Z=384 algos)")
    args = ap.parse_args()

    doc = load_yaml(args.yaml)

    args.bin = resolve_bin(args.bin, "cuphy_ex_ldpc_rm")

    scope, results, new_fail, elapsed, (bg, z) = sweep(
        doc, args.bin, args.testvectors_dir, args.algo)
    print_summary(scope, results, elapsed)

    print(f"\nTotal test time: {elapsed:.1f}s")
    if all(st == "NO-TV" for st in results.values()):
        print(f"ERROR: no test vector matched *_bg{bg}_Z{z}_p<p>.h5 under "
              f"{args.testvectors_dir}; nothing was checked.", file=sys.stderr)
        return 1
    errs = sorted(k for k, st in results.items() if st == "ERR")
    if errs:
        print(f"ERRORS (decoder threw, or TV contents did not match its name): {errs}",
              file=sys.stderr)
        return 1
    if new_fail:
        print(f"NEW failures (not in known_llr_fail): {new_fail}")
        return 1
    print("No new failures (known marginals allowed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
