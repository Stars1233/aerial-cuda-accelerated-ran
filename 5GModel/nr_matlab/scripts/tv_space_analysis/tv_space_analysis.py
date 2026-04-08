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
Analyze test vector (TV) space usage from ls -alh output.
Parses TVnr_DLMIX/ULMIX and other .h5 files, aggregates by pattern and category.
Pattern ranges align with 5GModel/nr_matlab/test/genPerfPattern.m.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Pattern ranges (aligned with 5GModel/nr_matlab/test/genPerfPattern.m)
PATTERNS = [
    ("48", "Pattern 48", [(3720, 4039)], [(2080, 2143)]),
    ("49", "Pattern 49", [(4040, 4439)], [(2144, 2223)]),
    ("50", "Pattern 50", [(4440, 4759)], [(2224, 2287)]),
    ("51", "Pattern 51", [(4760, 5159)], [(2416, 2495)]),
    ("59", "Pattern 59, 59a, 59b, 59d", [(7072, 7471)], [(3040, 3119), (3200, 3279), (3760, 3839), (3520, 3599)]),
    ("59c", "Pattern 59c, 59e, 59f, 62c", [(9472, 10271)], [(4471, 4630), (4631, 4790), (3840, 3840), (4040, 4079)]),
    ("60", "Pattern 60, 60a-e, 63c", [(10452, 11251)], [(3600, 3759), (3280, 3359), (4911, 5070), (5071, 5230), (5231, 5390), (4220, 4299), (5391, 5430)]),
    ("61", "Pattern 61", [(7872, 8271)], [(3360, 3439)]),
    ("65", "Pattern 65a-d", [(11432, 12231)], [(5971, 6010), (6011, 6050), (6051, 6210), (6211, 6370)]),
    ("66", "Pattern 66a-d", [(11252, 11341)], [(833, 833), (5431, 5445), (5446, 5460), (5461, 5580), (5581, 5700)]),
    ("67", "Pattern 67, 67a-e", [(11342, 11431)], [(833, 833), (4791, 4910), (5701, 5715), (5716, 5730), (5731, 5850), (5851, 5970), (6996, 6996)]),
    ("69", "Pattern 69, 69a-e, 71", [(12232, 12441)], [(6621, 6695), (6696, 6770), (6771, 6845), (6371, 6445), (6846, 6920), (7260, 7364), (6446, 6520)]),
    ("73", "Pattern 73", [(12442, 12721)], [(6521, 6620)]),
    ("75", "Pattern 75", [(12722, 12931)], [(6921, 6995)]),
    ("77", "Pattern 77", [(12932, 13066)], [(7050, 7154)]),
    ("79", "Pattern 79, 79a, 79b", [(13067, 13216)], [(7155, 7259), (7953, 8057), (8058, 8162)]),
    ("81a", "Pattern 81a, 81b", [(13217, 13366)], [(7365, 7469), (7470, 7574)]),
    ("81c", "Pattern 81c, 81d", [(13217, 13321), (13337, 13381)], [(7365, 7469), (7470, 7574)]),
    ("83a", "Pattern 83a, 83b", [(13382, 13531)], [(7575, 7679), (7680, 7784)]),
    ("83c", "Pattern 83c, 83d", [(13382, 13486), (13502, 13546)], [(7575, 7679), (7680, 7784)]),
    ("85", "Pattern 85", [(13547, 13696)], [(7785, 7889)]),
    ("87", "Pattern 87", [(13067, 13171), (13187, 13216), (13697, 13711)], [(7155, 7259)]),
    ("89", "Pattern 89", [(13712, 13801)], [(7890, 7952)]),
    ("91", "Pattern 91", [(13802, 13951)], [(8163, 8267)]),
    ("101", "Pattern 101, 101a", [(7472, 7495)], [(4375, 4470), (8268, 8363)]),
    ("102", "Pattern 102, 102a", [(7496, 7519)], [(8364, 8459), (8460, 8555)]),
]


def parse_size(size_val: str, size_unit: str):
    """Convert size to MB. Supports K/M/G/T. Returns None on error."""
    try:
        val = float(size_val)
        if size_unit == "K":
            return val / 1024.0
        if size_unit == "M":
            return val
        if size_unit == "G":
            return val * 1024.0
        if size_unit == "T":
            return val * 1024.0 * 1024.0
        return None
    except (ValueError, TypeError):
        return None


def dedupe_lines_by_filename(lines: List[str]) -> List[str]:
    """
    Keep one line per filename (last occurrence wins) so repeated log appends
    do not double-count. Only dedupes lines that look like file entries (last token ends with .h5).
    """
    file_lines = {}
    other_lines = []
    for line in lines:
        parts = line.split()
        if not parts:
            other_lines.append(line)
            continue
        fn = parts[-1]
        if fn.endswith(".h5"):
            file_lines[fn] = line
        else:
            other_lines.append(line)
    return other_lines + list(file_lines.values())


def parse_log_lines(
    lines: List[str],
) -> Tuple[Dict, Dict, List, List, int]:
    """
    Parse ls -alh style lines. Returns:
    (all_files, other_tvnr, non_tvnr, parse_errors, skipped_lines)
    """
    all_files = {}
    other_tvnr = {}
    non_tvnr = []
    parse_errors = []
    skipped_lines = 0

    for line_num, line in enumerate(lines, 1):
        try:
            match_dlul = re.search(
                r"\s+(\d+\.?\d*)([KMGT])\s+(?:.*\s+)?(\S*TVnr_(DLMIX|ULMIX)_(\d+)_([^\s]+)\.h5)$",
                line,
            )
            if match_dlul:
                size_val, size_unit, _fn, testtype, tc_num, rest = match_dlul.groups()
                size_mb = parse_size(size_val, size_unit)
                if size_mb is None:
                    parse_errors.append(f"Line {line_num}: Invalid size unit '{size_unit}' or value '{size_val}'")
                    skipped_lines += 1
                    continue
                try:
                    tc_num = int(tc_num)
                except ValueError:
                    parse_errors.append(f"Line {line_num}: Invalid test case number '{tc_num}'")
                    skipped_lines += 1
                    continue
                file_type = "FAPI" if "FAPI" in rest else ("CUPHY" if "CUPHY" in rest else "OTHER")
                key = (testtype, tc_num, file_type)
                if key not in all_files:
                    all_files[key] = [0, 0]
                all_files[key][0] += 1
                all_files[key][1] += size_mb
                continue

            match_tvnr = re.search(
                r"\s+(\d+\.?\d*)([KMGT])\s+(?:.*\s+)?(\S*TVnr_(\d+)_([^\s]+)\.h5)$",
                line,
            )
            if match_tvnr:
                size_val, size_unit, _fn, tc_num, rest = match_tvnr.groups()
                size_mb = parse_size(size_val, size_unit)
                if size_mb is None:
                    parse_errors.append(f"Line {line_num}: Invalid size unit '{size_unit}' or value '{size_val}'")
                    skipped_lines += 1
                    continue
                try:
                    tc_num = int(tc_num)
                except ValueError:
                    parse_errors.append(f"Line {line_num}: Invalid test case number '{tc_num}'")
                    skipped_lines += 1
                    continue
                file_type = "FAPI" if "FAPI" in rest else ("CUPHY" if "CUPHY" in rest else "OTHER")
                if tc_num not in other_tvnr:
                    other_tvnr[tc_num] = {"FAPI": [0, 0], "CUPHY": [0, 0], "OTHER": [0, 0]}
                other_tvnr[tc_num][file_type][0] += 1
                other_tvnr[tc_num][file_type][1] += size_mb
                continue

            match_h5 = re.search(r"\s+(\d+\.?\d*)([KMGT])\s+(?:.*\s+)?(\S+\.h5)$", line)
            if match_h5:
                size_val, size_unit, filename = match_h5.groups()
                size_mb = parse_size(size_val, size_unit)
                if size_mb is None:
                    parse_errors.append(f"Line {line_num}: Invalid size unit '{size_unit}' or value '{size_val}'")
                    skipped_lines += 1
                    continue
                if "TVnr_" not in filename:
                    non_tvnr.append((filename, size_mb))
        except Exception as e:
            parse_errors.append(f"Line {line_num}: {type(e).__name__}: {e}")
            skipped_lines += 1

    return all_files, other_tvnr, non_tvnr, parse_errors, skipped_lines


def run_ls_alh(dir_path: Path) -> List[str]:
    """Run ls -alh on a directory and return output lines.

    Args:
        dir_path: Path to the directory to list.

    Returns:
        List of output lines from ls -alh (stdout split by newlines).

    Raises:
        RuntimeError: If the ls command fails (non-zero exit code).
    """
    result = subprocess.run(
        ["ls", "-alh", str(dir_path.resolve())],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ls -alh failed: {result.stderr or result.stdout}")
    return result.stdout.splitlines()


def count_in_ranges(ranges, testtype: str, file_type: str, all_files: Dict) -> Tuple[int, float]:
    count = 0
    size = 0.0
    for start, end in ranges:
        for tc in range(start, end + 1):
            key = (testtype, tc, file_type)
            if key in all_files:
                count += all_files[key][0]
                size += all_files[key][1]
    return count, size


def count_in_ranges_dedupe(ranges, testtype: str, file_type: str, counted_set: Set, all_files: Dict) -> Tuple[int, float]:
    count = 0
    size = 0.0
    for start, end in ranges:
        for tc in range(start, end + 1):
            key = (testtype, tc, file_type)
            if key in counted_set:
                continue
            if key in all_files:
                count += all_files[key][0]
                size += all_files[key][1]
                counted_set.add(key)
    return count, size


def run_report(all_files: Dict, other_tvnr: Dict, non_tvnr: List, parse_errors: List, skipped_lines: int) -> None:
    """Print the full table and summary to stdout."""
    accounted_tcs = set()
    counted_for_total = set()

    print("\n" + "=" * 190)
    print(f"{'Pattern / Category':<30} | {'DLMIX FAPI':^19} | {'DLMIX cuPHY':^19} | {'ULMIX FAPI':^19} | {'ULMIX cuPHY':^19} | {'Subtotal':^19} | {'Total':^15}")
    print(f"{'':30} | {'#TVs':>8} {'Size':>10} | {'#TVs':>8} {'Size':>10} | {'#TVs':>8} {'Size':>10} | {'#TVs':>8} {'Size':>10} | {'#TVs':>8} {'Size':>10} | {'Size':>15}")
    print("=" * 190)

    totals = {
        "dlmix_fapi": (0, 0.0),
        "dlmix_cuphy": (0, 0.0),
        "dlmix_other": (0, 0.0),
        "ulmix_fapi": (0, 0.0),
        "ulmix_cuphy": (0, 0.0),
        "ulmix_other": (0, 0.0),
    }

    for _pid, pattern_name, dlmix_ranges, ulmix_ranges in PATTERNS:
        d_fapi_c, d_fapi_s = count_in_ranges(dlmix_ranges, "DLMIX", "FAPI", all_files)
        d_cuphy_c, d_cuphy_s = count_in_ranges(dlmix_ranges, "DLMIX", "CUPHY", all_files)
        d_other_c, d_other_s = count_in_ranges(dlmix_ranges, "DLMIX", "OTHER", all_files)
        u_fapi_c, u_fapi_s = count_in_ranges(ulmix_ranges, "ULMIX", "FAPI", all_files)
        u_cuphy_c, u_cuphy_s = count_in_ranges(ulmix_ranges, "ULMIX", "CUPHY", all_files)
        u_other_c, u_other_s = count_in_ranges(ulmix_ranges, "ULMIX", "OTHER", all_files)

        for start, end in dlmix_ranges:
            for tc in range(start, end + 1):
                accounted_tcs.add(("DLMIX", tc))
        for start, end in ulmix_ranges:
            for tc in range(start, end + 1):
                accounted_tcs.add(("ULMIX", tc))

        st_count = d_fapi_c + d_cuphy_c + d_other_c + u_fapi_c + u_cuphy_c + u_other_c
        st_size = d_fapi_s + d_cuphy_s + d_other_s + u_fapi_s + u_cuphy_s + u_other_s

        d_fapi_str = f"{d_fapi_s / 1024:.1f}GB" if d_fapi_s > 0 else "-"
        d_cuphy_str = f"{d_cuphy_s / 1024:.1f}GB" if d_cuphy_s > 0 else "-"
        u_fapi_str = f"{u_fapi_s / 1024:.1f}GB" if u_fapi_s > 0 else "-"
        u_cuphy_str = f"{u_cuphy_s / 1024:.1f}GB" if u_cuphy_s > 0 else "-"
        st_str = f"{st_size / 1024:.1f}GB" if st_size > 0 else "-"
        total_str = f"{st_size / 1024:.2f} GB" if st_size > 0 else "-"

        print(f"{pattern_name:<30} | {d_fapi_c:>8} {d_fapi_str:>10} | {d_cuphy_c:>8} {d_cuphy_str:>10} | {u_fapi_c:>8} {u_fapi_str:>10} | {u_cuphy_c:>8} {u_cuphy_str:>10} | {st_count:>8} {st_str:>10} | {total_str:>15}")

        dc, ds = count_in_ranges_dedupe(dlmix_ranges, "DLMIX", "FAPI", counted_for_total, all_files)
        totals["dlmix_fapi"] = (totals["dlmix_fapi"][0] + dc, totals["dlmix_fapi"][1] + ds)
        dc, ds = count_in_ranges_dedupe(dlmix_ranges, "DLMIX", "CUPHY", counted_for_total, all_files)
        totals["dlmix_cuphy"] = (totals["dlmix_cuphy"][0] + dc, totals["dlmix_cuphy"][1] + ds)
        dc, ds = count_in_ranges_dedupe(dlmix_ranges, "DLMIX", "OTHER", counted_for_total, all_files)
        totals["dlmix_other"] = (totals["dlmix_other"][0] + dc, totals["dlmix_other"][1] + ds)
        dc, ds = count_in_ranges_dedupe(ulmix_ranges, "ULMIX", "FAPI", counted_for_total, all_files)
        totals["ulmix_fapi"] = (totals["ulmix_fapi"][0] + dc, totals["ulmix_fapi"][1] + ds)
        dc, ds = count_in_ranges_dedupe(ulmix_ranges, "ULMIX", "CUPHY", counted_for_total, all_files)
        totals["ulmix_cuphy"] = (totals["ulmix_cuphy"][0] + dc, totals["ulmix_cuphy"][1] + ds)
        dc, ds = count_in_ranges_dedupe(ulmix_ranges, "ULMIX", "OTHER", counted_for_total, all_files)
        totals["ulmix_other"] = (totals["ulmix_other"][0] + dc, totals["ulmix_other"][1] + ds)

    other_dlmix_fapi = (0, 0.0)
    other_dlmix_cuphy = (0, 0.0)
    other_dlmix_other = (0, 0.0)
    other_ulmix_fapi = (0, 0.0)
    other_ulmix_cuphy = (0, 0.0)
    other_ulmix_other = (0, 0.0)
    for (testtype, tc_num, file_type), (count, size_mb) in all_files.items():
        if (testtype, tc_num) not in accounted_tcs:
            if testtype == "DLMIX" and file_type == "FAPI":
                other_dlmix_fapi = (other_dlmix_fapi[0] + count, other_dlmix_fapi[1] + size_mb)
            elif testtype == "DLMIX" and file_type == "CUPHY":
                other_dlmix_cuphy = (other_dlmix_cuphy[0] + count, other_dlmix_cuphy[1] + size_mb)
            elif testtype == "DLMIX" and file_type == "OTHER":
                other_dlmix_other = (other_dlmix_other[0] + count, other_dlmix_other[1] + size_mb)
            elif testtype == "ULMIX" and file_type == "FAPI":
                other_ulmix_fapi = (other_ulmix_fapi[0] + count, other_ulmix_fapi[1] + size_mb)
            elif testtype == "ULMIX" and file_type == "CUPHY":
                other_ulmix_cuphy = (other_ulmix_cuphy[0] + count, other_ulmix_cuphy[1] + size_mb)
            elif testtype == "ULMIX" and file_type == "OTHER":
                other_ulmix_other = (other_ulmix_other[0] + count, other_ulmix_other[1] + size_mb)

    other_total_count = (
        other_dlmix_fapi[0] + other_dlmix_cuphy[0] + other_dlmix_other[0]
        + other_ulmix_fapi[0] + other_ulmix_cuphy[0] + other_ulmix_other[0]
    )
    other_total_size = (
        other_dlmix_fapi[1] + other_dlmix_cuphy[1] + other_dlmix_other[1]
        + other_ulmix_fapi[1] + other_ulmix_cuphy[1] + other_ulmix_other[1]
    )

    print("-" * 190)
    o_fapi_s = f"{other_dlmix_fapi[1] / 1024:.1f}GB" if other_dlmix_fapi[1] > 0 else "-"
    o_cuphy_s = f"{other_dlmix_cuphy[1] / 1024:.1f}GB" if other_dlmix_cuphy[1] > 0 else "-"
    o_ul_fapi_s = f"{other_ulmix_fapi[1] / 1024:.1f}GB" if other_ulmix_fapi[1] > 0 else "-"
    o_ul_cuphy_s = f"{other_ulmix_cuphy[1] / 1024:.1f}GB" if other_ulmix_cuphy[1] > 0 else "-"
    print(f"{'Other DLMIX/ULMIX (no pattern)':<30} | {other_dlmix_fapi[0]:>8} {o_fapi_s:>10} | {other_dlmix_cuphy[0]:>8} {o_cuphy_s:>10} | {other_ulmix_fapi[0]:>8} {o_ul_fapi_s:>10} | {other_ulmix_cuphy[0]:>8} {o_ul_cuphy_s:>10} | {other_total_count:>8} {other_total_size / 1024:.1f}GB   | {other_total_size / 1024:.2f} GB")

    dlmix_ulmix_total_count = (
        totals["dlmix_fapi"][0] + totals["dlmix_cuphy"][0] + totals["dlmix_other"][0]
        + totals["ulmix_fapi"][0] + totals["ulmix_cuphy"][0] + totals["ulmix_other"][0]
        + other_total_count
    )
    dlmix_ulmix_total_size = (
        totals["dlmix_fapi"][1] + totals["dlmix_cuphy"][1] + totals["dlmix_other"][1]
        + totals["ulmix_fapi"][1] + totals["ulmix_cuphy"][1] + totals["ulmix_other"][1]
        + other_total_size
    )

    print("─" * 190)
    st_d_fapi = (totals["dlmix_fapi"][1] + other_dlmix_fapi[1]) / 1024
    st_d_cuphy = (totals["dlmix_cuphy"][1] + other_dlmix_cuphy[1]) / 1024
    st_u_fapi = (totals["ulmix_fapi"][1] + other_ulmix_fapi[1]) / 1024
    st_u_cuphy = (totals["ulmix_cuphy"][1] + other_ulmix_cuphy[1]) / 1024
    print(f"{'SUBTOTAL: DLMIX/ULMIX Files':<30} | {totals['dlmix_fapi'][0] + other_dlmix_fapi[0]:>8} {st_d_fapi:.1f}GB     | {totals['dlmix_cuphy'][0] + other_dlmix_cuphy[0]:>8} {st_d_cuphy:.1f}GB     | {totals['ulmix_fapi'][0] + other_ulmix_fapi[0]:>8} {st_u_fapi:.1f}GB     | {totals['ulmix_cuphy'][0] + other_ulmix_cuphy[0]:>8} {st_u_cuphy:.1f}GB     | {dlmix_ulmix_total_count:>8} {dlmix_ulmix_total_size / 1024:.1f}GB     | {dlmix_ulmix_total_size / 1024:>12.2f} GB")

    print("─" * 190)
    other_tvnr_fapi_c = sum(d.get("FAPI", [0, 0])[0] for d in other_tvnr.values())
    other_tvnr_fapi_s = sum(d.get("FAPI", [0, 0])[1] for d in other_tvnr.values())
    other_tvnr_cuphy_c = sum(d.get("CUPHY", [0, 0])[0] for d in other_tvnr.values())
    other_tvnr_cuphy_s = sum(d.get("CUPHY", [0, 0])[1] for d in other_tvnr.values())
    other_tvnr_other_c = sum(d.get("OTHER", [0, 0])[0] for d in other_tvnr.values())
    other_tvnr_other_s = sum(d.get("OTHER", [0, 0])[1] for d in other_tvnr.values())
    other_tvnr_count = other_tvnr_fapi_c + other_tvnr_cuphy_c + other_tvnr_other_c
    other_tvnr_total = other_tvnr_fapi_s + other_tvnr_cuphy_s + other_tvnr_other_s

    ot_fapi_str = f"{other_tvnr_fapi_s / 1024:.1f}GB" if other_tvnr_fapi_s > 0 else "-"
    ot_cuphy_str = f"{other_tvnr_cuphy_s / 1024:.1f}GB" if other_tvnr_cuphy_s > 0 else "-"
    print(f"{'Other TVnr_ (not DLMIX/ULMIX)':<30} | {other_tvnr_fapi_c:>8} {ot_fapi_str:>10} | {other_tvnr_cuphy_c:>8} {ot_cuphy_str:>10} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {other_tvnr_count:>8} {other_tvnr_total / 1024:.1f}GB     | {other_tvnr_total / 1024:>12.2f} GB")

    non_tvnr_count = len(non_tvnr)
    non_tvnr_size = sum(s for _, s in non_tvnr)
    non_str = f"{non_tvnr_size / 1024:.1f}GB" if non_tvnr_size > 0 else "0.0GB"
    print(f"{'Non-TVnr_ .h5 files':<30} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {non_tvnr_count:>8} {non_str:>10} | {non_tvnr_size / 1024:>12.2f} GB")

    print("=" * 190)
    grand_count = dlmix_ulmix_total_count + other_tvnr_count + non_tvnr_count
    grand_size = dlmix_ulmix_total_size + other_tvnr_total + non_tvnr_size
    grand_str = f"{grand_size / 1024:.1f}GB"
    print(f"{'GRAND TOTAL (All .h5 files)':<30} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {grand_count:>8} {grand_str:>10} | {grand_size / 1024:>12.2f} GB")
    print("=" * 190)

    print("\nComplete Summary:")
    print("   " + "═" * 60)
    print(f"   {'Category':<35} {'Files':>10} {'Size':>15}")
    print("   " + "─" * 60)
    print(f"   {'DLMIX/ULMIX in Defined Patterns:':<35} {dlmix_ulmix_total_count - other_total_count:>10} {(dlmix_ulmix_total_size - other_total_size) / 1024:>13.2f} GB")
    print(f"   {'DLMIX/ULMIX not in Patterns:':<35} {other_total_count:>10} {other_total_size / 1024:>13.2f} GB")
    print(f"   {'Other TVnr_ files:':<35} {other_tvnr_count:>10} {other_tvnr_total / 1024:>13.2f} GB")
    print(f"   {'Non-TVnr_ .h5 files:':<35} {non_tvnr_count:>10} {non_tvnr_size / 1024:>13.2f} GB")
    print("   " + "─" * 60)
    print(f"   {'TOTAL:':<35} {grand_count:>10} {grand_size / 1024:>13.2f} GB")
    print("   " + "═" * 60)
    print()

    if skipped_lines or parse_errors:
        print("WARNING: Parsing Warnings and Errors:")
        print("   " + "─" * 60)
        print(f"   Skipped lines: {skipped_lines}")
        if parse_errors:
            print("\n   Error details (showing first 10):")
            for err in parse_errors[:10]:
                print(f"   • {err}")
            if len(parse_errors) > 10:
                print(f"   ... and {len(parse_errors) - 10} more errors")
        print("   " + "─" * 60)
        print()


def run_report_lp(all_files: Dict, lp_ids: List, parse_errors: List, skipped_lines: int) -> bool:
    """Print report for one or more LPs (patterns): size of all child TVs. Returns False if any LP unknown."""
    valid_ids = [p[0] for p in PATTERNS]
    patterns_by_id = {p[0]: p for p in PATTERNS}
    for lp_id in lp_ids:
        if lp_id not in patterns_by_id:
            print(f"Error: Unknown LP '{lp_id}'. Valid LP IDs: {', '.join(valid_ids)}", file=sys.stderr)
            return False

    rows = []
    # Use a shared set so overlapping LPs (e.g. 81a, 81c) are not double-counted in combined total
    combined_counted = set()
    combined_count = 0
    combined_size_mb = 0.0
    for lp_id in lp_ids:
        _pid, pattern_name, dlmix_ranges, ulmix_ranges = patterns_by_id[lp_id]
        d_fapi_c, d_fapi_s = count_in_ranges(dlmix_ranges, "DLMIX", "FAPI", all_files)
        d_cuphy_c, d_cuphy_s = count_in_ranges(dlmix_ranges, "DLMIX", "CUPHY", all_files)
        d_other_c, d_other_s = count_in_ranges(dlmix_ranges, "DLMIX", "OTHER", all_files)
        u_fapi_c, u_fapi_s = count_in_ranges(ulmix_ranges, "ULMIX", "FAPI", all_files)
        u_cuphy_c, u_cuphy_s = count_in_ranges(ulmix_ranges, "ULMIX", "CUPHY", all_files)
        u_other_c, u_other_s = count_in_ranges(ulmix_ranges, "ULMIX", "OTHER", all_files)
        total_count = d_fapi_c + d_cuphy_c + d_other_c + u_fapi_c + u_cuphy_c + u_other_c
        total_size_mb = d_fapi_s + d_cuphy_s + d_other_s + u_fapi_s + u_cuphy_s + u_other_s
        # Add only deduped counts to combined so overlapping LPs are not double-counted
        dc, ds = count_in_ranges_dedupe(dlmix_ranges, "DLMIX", "FAPI", combined_counted, all_files)
        combined_count += dc
        combined_size_mb += ds
        dc, ds = count_in_ranges_dedupe(dlmix_ranges, "DLMIX", "CUPHY", combined_counted, all_files)
        combined_count += dc
        combined_size_mb += ds
        dc, ds = count_in_ranges_dedupe(dlmix_ranges, "DLMIX", "OTHER", combined_counted, all_files)
        combined_count += dc
        combined_size_mb += ds
        dc, ds = count_in_ranges_dedupe(ulmix_ranges, "ULMIX", "FAPI", combined_counted, all_files)
        combined_count += dc
        combined_size_mb += ds
        dc, ds = count_in_ranges_dedupe(ulmix_ranges, "ULMIX", "CUPHY", combined_counted, all_files)
        combined_count += dc
        combined_size_mb += ds
        dc, ds = count_in_ranges_dedupe(ulmix_ranges, "ULMIX", "OTHER", combined_counted, all_files)
        combined_count += dc
        combined_size_mb += ds
        rows.append((lp_id, pattern_name, d_fapi_c, d_fapi_s, d_cuphy_c, d_cuphy_s, d_other_c, d_other_s,
                     u_fapi_c, u_fapi_s, u_cuphy_c, u_cuphy_s, u_other_c, u_other_s, total_count, total_size_mb))

    print("\n" + "=" * 190)
    print(f"{'Pattern / Category':<30} | {'DLMIX FAPI':^19} | {'DLMIX cuPHY':^19} | {'ULMIX FAPI':^19} | {'ULMIX cuPHY':^19} | {'Subtotal':^19} | {'Total':^15}")
    print(f"{'':30} | {'#TVs':>8} {'Size':>10} | {'#TVs':>8} {'Size':>10} | {'#TVs':>8} {'Size':>10} | {'#TVs':>8} {'Size':>10} | {'#TVs':>8} {'Size':>10} | {'Size':>15}")
    print("=" * 190)

    for (lp_id, pattern_name, d_fapi_c, d_fapi_s, d_cuphy_c, d_cuphy_s, _do_c, _do_s,
         u_fapi_c, u_fapi_s, u_cuphy_c, u_cuphy_s, _uo_c, _uo_s, total_count, total_size_mb) in rows:
        d_fapi_str = f"{d_fapi_s / 1024:.1f}GB" if d_fapi_s > 0 else "-"
        d_cuphy_str = f"{d_cuphy_s / 1024:.1f}GB" if d_cuphy_s > 0 else "-"
        u_fapi_str = f"{u_fapi_s / 1024:.1f}GB" if u_fapi_s > 0 else "-"
        u_cuphy_str = f"{u_cuphy_s / 1024:.1f}GB" if u_cuphy_s > 0 else "-"
        st_str = f"{total_size_mb / 1024:.1f}GB" if total_size_mb > 0 else "-"
        total_str = f"{total_size_mb / 1024:.2f} GB" if total_size_mb > 0 else "-"
        print(f"{pattern_name:<30} | {d_fapi_c:>8} {d_fapi_str:>10} | {d_cuphy_c:>8} {d_cuphy_str:>10} | {u_fapi_c:>8} {u_fapi_str:>10} | {u_cuphy_c:>8} {u_cuphy_str:>10} | {total_count:>8} {st_str:>10} | {total_str:>15}")

    if len(lp_ids) > 1:
        st_str = f"{combined_size_mb / 1024:.1f}GB" if combined_size_mb > 0 else "-"
        total_str = f"{combined_size_mb / 1024:.2f} GB" if combined_size_mb > 0 else "-"
        print("-" * 190)
        print(f"{'Total (selected LPs)':<30} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {combined_count:>8} {st_str:>10} | {total_str:>15}")
    print("=" * 190)

    for (lp_id, pattern_name, d_fapi_c, d_fapi_s, d_cuphy_c, d_cuphy_s, d_other_c, d_other_s,
         u_fapi_c, u_fapi_s, u_cuphy_c, u_cuphy_s, u_other_c, u_other_s, total_count, total_size_mb) in rows:
        print(f"\nLP {lp_id} (child TVs only):")
        print("   " + "═" * 50)
        print(f"   {'Files':>10}   {total_count}")
        print(f"   {'Size':>10}   {total_size_mb / 1024:.2f} GB")
        print("   " + "═" * 50)
        print("   DLMIX: FAPI {} ({}), CUPHY {} ({}), OTHER {} ({} MB)".format(
            d_fapi_c, f"{d_fapi_s / 1024:.2f} GB", d_cuphy_c, f"{d_cuphy_s / 1024:.2f} GB", d_other_c, f"{d_other_s:.1f}"))
        print("   ULMIX: FAPI {} ({}), CUPHY {} ({}), OTHER {} ({} MB)".format(
            u_fapi_c, f"{u_fapi_s / 1024:.2f} GB", u_cuphy_c, f"{u_cuphy_s / 1024:.2f} GB", u_other_c, f"{u_other_s:.1f}"))

    if len(lp_ids) > 1:
        print(f"\nCombined ({', '.join(lp_ids)}):")
        print("   " + "═" * 50)
        print(f"   {'Files':>10}   {combined_count}")
        print(f"   {'Size':>10}   {combined_size_mb / 1024:.2f} GB")
        print("   " + "═" * 50)
    print()

    if skipped_lines or parse_errors:
        print("WARNING: Parsing Warnings and Errors:")
        print("   " + "─" * 60)
        print(f"   Skipped lines: {skipped_lines}")
        if parse_errors:
            for err in parse_errors[:10]:
                print(f"   • {err}")
            if len(parse_errors) > 10:
                print(f"   ... and {len(parse_errors) - 10} more errors")
        print("   " + "─" * 60)
        print()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze test vector (TV) .h5 disk usage by pattern and category.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dir /path/to/TV/folder
  %(prog)s --dir /path/to/TV/folder --log my.log
  %(prog)s --dir /path/to/TV/folder --lp 81a
        """,
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="TV directory to analyze (runs ls -alh on it).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional: combine this log file with directory listing; updated with combined content.",
    )
    parser.add_argument(
        "--lp",
        type=str,
        default=None,
        metavar="ID[,ID,...]",
        help="Optional: analyze only these LP(s) and report size of their child TVs. Multiple IDs separated by commas, e.g. 81a,81c or 59c,102.",
    )
    args = parser.parse_args()

    dir_path = args.dir.resolve()
    if not dir_path.is_dir():
        print(f"Error: Directory '{dir_path}' not found.", file=sys.stderr)
        return 1

    lines = []
    if args.log is not None:
        log_path = args.log.resolve()
        if log_path.exists():
            lines = log_path.read_text().splitlines()
        else:
            log_path.touch()
            print(f"Info: Created log file: {log_path}")
        print("Combining log file and directory listing (both inputs will be analyzed)")
        try:
            ls_lines = run_ls_alh(dir_path)
        except RuntimeError as e:
            print(f"Failed to list ALH directory: {e}", file=sys.stderr)
            return 1
        lines = lines + [""] + ls_lines
        log_path.write_text("\n".join(lines) + "\n")
        print(f"Updated {log_path} with combined contents (original log + directory listing).")
        print()
    else:
        print(f"Analyzing directory: {dir_path}")
        print(f"Running: ls -alh '{dir_path}'")
        try:
            lines = run_ls_alh(dir_path)
        except RuntimeError as e:
            print(f"Failed to list ALH directory: {e}", file=sys.stderr)
            return 1
        print("Directory listing complete")
        print()

    # Dedupe by filename so repeated --log appends do not double-count
    lines = dedupe_lines_by_filename(lines)
    print(f"Analyzing {len(lines)} lines")
    print()

    all_files, other_tvnr, non_tvnr, parse_errors, skipped = parse_log_lines(lines)
    if args.lp:
        lp_list = [x.strip() for x in args.lp.split(",") if x.strip()]
        if not lp_list:
            print("Error: --lp requires at least one LP ID.", file=sys.stderr)
            return 1
        if not run_report_lp(all_files, lp_list, parse_errors, skipped):
            return 1
    else:
        run_report(all_files, other_tvnr, non_tvnr, parse_errors, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
