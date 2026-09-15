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

"""
LDPC decoder kernel R/S/L (REG/STACK/LOCAL) resource report.

For every accessory-capable decoder kernel it prints REG/STACK/LOCAL per GPU
arch, for the two variants that live in the SAME build:

    noET    the `_no_accessories` (ET-compiled-out / lean) kernel
    ET-on   the accessory kernel (ET compiled in)

So a single `.so` already shows the ET cost per kernel -- no GPU, fully offline
(cuobjdump reads every arch from the fatbin). For a before/after comparison,
either run this twice (build A, build B) and diff the two outputs, or pass
--premr to add symmetric PreMR-noET / PreMR-ET columns from a baseline build in
one shot (the git-worktree pattern for agents). A genuinely pre-ET baseline
simply shows `-` under PreMR-noET.

Accessory pairs are auto-discovered (every `*_no_accessories` symbol implies a
pair), so new kernels appear automatically.

This tool reports static resource usage only. To see which algo the
auto-dispatcher selects for a kernel, use cuphy_ldpc_find_dispatch /
plot_cuphy_ldpc_find_dispatch.py on the target GPU (dispatch is per-arch, so it
must run on the arch of interest) and cross-reference by kernel name.

Inputs may be a compiled `libcuphy_ldpc.so` (cuobjdump is run for you) or a saved
`cuobjdump --dump-resource-usage` text file. With no positional arg the first
`build*/cuPHY/src/cuphy/libcuphy_ldpc.so` under the repo is used.

Examples
--------
    # this build only (auto-find the .so): noET vs ET-on
    python ldpc_kernel_resource_report.py

    # restrict to one arch present in the fatbin
    python ldpc_kernel_resource_report.py --arches sm_120

    # one-shot before/after against a baseline build
    python ldpc_kernel_resource_report.py \\
        build.aarch64/cuPHY/src/cuphy/libcuphy_ldpc.so \\
        --premr <premr-build>/cuPHY/src/cuphy/libcuphy_ldpc.so
"""

import argparse
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SO_REL = "cuPHY/src/cuphy/libcuphy_ldpc.so"

RES_RE  = re.compile(r'(REG|STACK|LOCAL):(\d+)')
ARCH_RE = re.compile(r'arch = (sm_\d+)')
FUNC_RE = re.compile(r'\s*Function (\S+):')
SUFFIX  = "_no_accessories"


def default_so() -> Path | None:
    """First build*/.../libcuphy_ldpc.so under the repo, or None."""
    hits = sorted(REPO_ROOT.glob(f"build*/{SO_REL}"))
    return hits[0] if hits else None


def dump_text(path: Path) -> str:
    """Return cuobjdump resource text; run cuobjdump if given an ELF object."""
    with path.open("rb") as f:  # read only the magic, not the whole .so
        is_elf = f.read(4) == b"\x7fELF"
    if not is_elf:
        return path.read_text()
    try:
        return subprocess.run(["cuobjdump", "--dump-resource-usage", str(path)],
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL).stdout
    except FileNotFoundError:
        raise SystemExit("cuobjdump not found on PATH (needed to read a .so; pass a "
                         "saved 'cuobjdump --dump-resource-usage' text file instead).")


def parse(text: str) -> dict[tuple[str, str], str]:
    """(arch, kernel) -> 'REG/STACK/LOCAL'."""
    out: dict[tuple[str, str], str] = {}
    arch = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = ARCH_RE.match(line)
        if m:
            arch = m.group(1)
            continue
        m = FUNC_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        # Scan the following lines for the REG/STACK/LOCAL line rather than
        # assuming it is exactly i+1; stop at the next Function/arch section.
        for j in range(i + 1, min(i + 8, len(lines))):
            if FUNC_RE.match(lines[j]) or ARCH_RE.match(lines[j]):
                break
            v = dict(RES_RE.findall(lines[j]))
            if v:
                out[(arch, name)] = f"{v.get('REG','?')}/{v.get('STACK','?')}/{v.get('LOCAL','?')}"
                break
    return out


def main() -> None:
    """Parse arguments, read kernel resource usage, and print the per-arch table."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("so", type=Path, nargs="?", default=None,
                    help="this-build libcuphy_ldpc.so or cuobjdump text "
                         f"(default: first build*/{SO_REL}).")
    ap.add_argument("--premr", type=Path, default=None, help="baseline .so/text for the PreMR columns")
    ap.add_argument("--arches", nargs="+", default=["sm_90", "sm_120"])
    args = ap.parse_args()

    so = args.so or default_so()
    if so is None:
        raise SystemExit(f"No build*/{SO_REL} found under {REPO_ROOT}; pass one explicitly.")

    mr = parse(dump_text(so))
    pm = parse(dump_text(args.premr)) if args.premr else None

    # Auto-discover accessory pairs: any *_no_accessories symbol.
    bases = sorted({n[:-len(SUFFIX)] for (a, n) in mr if n.endswith(SUFFIX)})
    if not bases:
        raise SystemExit(f"No *{SUFFIX} kernels found in {so} -- is this libcuphy_ldpc.so, "
                         "and is cuobjdump emitting demangled ldpc2_ names?")

    # Column labels per arch: optional PreMR pair, then this build's pair.
    labels = (["PreMR-noET", "PreMR-ET"] if pm else []) + ["noET", "ET-on"]
    w = 10

    def cells(a: str, base: str) -> list[str]:
        """Return the R/S/L cells for one arch and kernel base, in column order."""
        vals = []
        if pm is not None:
            vals += [pm.get((a, base + SUFFIX), "-"), pm.get((a, base), "-")]
        vals += [mr.get((a, base + SUFFIX), "-"), mr.get((a, base), "-")]
        return vals

    hdr = f"{'kernel':46}"
    for a in args.arches:
        hdr += " | " + f"{a}: " + " ".join(f"{lab:>{w}}" for lab in labels)
    print(hdr)
    print("-" * len(hdr))
    for base in bases:
        short = base.replace("ldpc2_", "")
        row = f"{short:46}"
        for a in args.arches:
            row += " | " + " " * len(f"{a}: ") + " ".join(f"{c:>{w}}" for c in cells(a, base))
        print(row)
    print(f"\nLegend: REG/STACK/LOCAL bytes. noET={SUFFIX}; ET-on=accessory; "
          f"PreMR={'given' if pm else 'not given'}. {len(bases)} accessory kernels.")


if __name__ == "__main__":
    main()
