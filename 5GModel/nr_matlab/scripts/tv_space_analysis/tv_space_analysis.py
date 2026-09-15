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

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

PatternRow = Tuple[str, str, List[Tuple[int, int]], List[Tuple[int, int]]]
AnalysisGroup = Tuple[str, str, Tuple[str, ...]]
NRSIM_TV_MIN = 20000


def _range_entries_to_ranges(entries) -> List[Tuple[int, int]]:
    """Convert YAML scalar/list range entries into inclusive (start, end) tuples."""
    if entries is None:
        return []
    if not isinstance(entries, list):
        entries = [entries]

    ranges = []
    for entry in entries:
        if isinstance(entry, bool):
            raise ValueError("range entries must be numeric TVs, not booleans")
        if isinstance(entry, (int, float)):
            if int(entry) != entry:
                raise ValueError(f"single TV range entry must be integral: {entry}")
            start = int(entry)
            end = int(entry)
        elif isinstance(entry, (list, tuple)):
            if len(entry) == 1:
                start = int(entry[0])
                end = int(entry[0])
            elif len(entry) == 2:
                start = int(entry[0])
                end = int(entry[1])
            else:
                raise ValueError(f"range list entries must be [start, end], got {entry!r}")
        elif isinstance(entry, dict):
            entry_value = (
                entry.get("main")
                if "main" in entry
                else entry.get("additional")
                if "additional" in entry
                else entry.get("tv")
                if "tv" in entry
                else entry.get("range")
            )
            if entry_value is not None:
                if isinstance(entry_value, (list, tuple)):
                    if len(entry_value) == 1:
                        start = int(entry_value[0])
                        end = start
                    elif len(entry_value) == 2:
                        start = int(entry_value[0])
                        end = int(entry_value[1])
                    else:
                        raise ValueError(f"range mapping entries must use scalar or [start, end], got {entry!r}")
                else:
                    start = int(entry_value)
                    end = start
            elif "start_tv" in entry and "end_tv" in entry:
                start = int(entry["start_tv"])
                end = int(entry["end_tv"])
            else:
                raise ValueError(f"unsupported range mapping entry: {entry!r}")
        else:
            raise ValueError(f"unsupported range entry: {entry!r}")
        if end < start:
            raise ValueError(f"range end {end} is smaller than start {start}")
        ranges.append((start, end))
    return ranges


def _side_ranges(ranges_cfg: Dict, side: str) -> List[Tuple[int, int]]:
    if not ranges_cfg:
        return []
    if not isinstance(ranges_cfg, dict):
        raise ValueError("pattern ranges must be a mapping")
    side_cfg = ranges_cfg.get(side, ranges_cfg.get(side.lower()))
    if isinstance(side_cfg, dict):
        side_cfg = side_cfg.get("tvs", [])
    return _range_entries_to_ranges(side_cfg)


def _unique_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """De-dupe exact ranges while preserving the YAML/report order."""
    unique = []
    seen = set()
    for range_pair in ranges:
        if range_pair in seen:
            continue
        unique.append(range_pair)
        seen.add(range_pair)
    return unique


def _pattern_ranges(entry: Dict, lp_id: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    ranges_cfg = entry.get("ranges", {})
    try:
        dlmix_ranges = _side_ranges(ranges_cfg, "DLMIX")
        ulmix_ranges = _side_ranges(ranges_cfg, "ULMIX")
    except ValueError as exc:
        raise ValueError(f"Pattern '{lp_id}': invalid ranges: {exc}") from exc
    return dlmix_ranges, ulmix_ranges


def _load_analysis_groups(cfg: Dict) -> List[AnalysisGroup]:
    groups_cfg = cfg.get("analysis_groups") or []
    if not isinstance(groups_cfg, list):
        raise ValueError("Config 'analysis_groups' must be a list")

    groups = []
    for group_cfg in groups_cfg:
        if not isinstance(group_cfg, dict):
            raise ValueError("Each analysis group entry must be a mapping")
        group_id = str(group_cfg.get("id", "")).strip()
        if not group_id:
            raise ValueError("analysis group entry missing 'id'")
        label = str(group_cfg.get("label", f"Pattern {group_id}")).strip()
        pattern_ids_cfg = group_cfg.get("patterns")
        if pattern_ids_cfg is None:
            pattern_ids_cfg = [group_id]
        elif not isinstance(pattern_ids_cfg, list):
            pattern_ids_cfg = [pattern_ids_cfg]
        pattern_ids = tuple(str(pattern_id).strip() for pattern_id in pattern_ids_cfg)
        if not pattern_ids or any(not pattern_id for pattern_id in pattern_ids):
            raise ValueError(f"Analysis group '{group_id}' must list at least one pattern id")
        groups.append((group_id, label, pattern_ids))
    return groups


def _load_patterns_from_yaml(config_path: Path) -> Tuple[List[PatternRow], Dict[str, PatternRow]]:
    """Load report-grouped and individual pattern ranges from YAML."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config. Install with: pip install pyyaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a dict")

    analysis_groups = _load_analysis_groups(cfg)

    root_patterns = cfg.get("patterns", [])
    if not isinstance(root_patterns, list):
        raise ValueError("Config 'patterns' must be a list")
    patterns_cfg = list(root_patterns)
    pattern_files = cfg.get("pattern_files") or []
    if pattern_files:
        if not isinstance(pattern_files, list):
            raise ValueError("Config 'pattern_files' must be a list")
        for pattern_file in pattern_files:
            child_path = config_path.parent / str(pattern_file)
            with open(child_path, "r", encoding="utf-8") as f:
                child_cfg = yaml.safe_load(f)
            if not isinstance(child_cfg, dict) or not isinstance(child_cfg.get("patterns"), list):
                raise ValueError(f"Pattern file must have a 'patterns' list: {child_path}")
            patterns_cfg.extend(child_cfg["patterns"])
    patterns_cfg.sort(key=lambda entry: float(entry.get("pattern", float("inf"))))

    individual_patterns = {}
    genperf_ids = []
    for entry in patterns_cfg:
        if not isinstance(entry, dict):
            raise ValueError("Each pattern entry must be a mapping")
        lp_id = str(entry.get("id", "")).strip()
        if not lp_id:
            raise ValueError("pattern entry missing 'id'")
        dlmix_ranges, ulmix_ranges = _pattern_ranges(entry, lp_id)
        individual_patterns[lp_id] = (lp_id, f"Pattern {lp_id}", dlmix_ranges, ulmix_ranges)
        if entry.get("genperf"):
            genperf_ids.append(lp_id)

    grouped_patterns = []
    grouped_ids = set()
    for group_id, label, pattern_ids in analysis_groups:
        dlmix_ranges = []
        ulmix_ranges = []
        missing_ids = []
        for lp_id in pattern_ids:
            pattern = individual_patterns.get(lp_id)
            if pattern is None:
                missing_ids.append(lp_id)
                continue
            dlmix_ranges.extend(pattern[2])
            ulmix_ranges.extend(pattern[3])
        if missing_ids:
            missing = ", ".join(missing_ids)
            raise ValueError(f"Analysis pattern group '{label}' references missing pattern id(s): {missing}")
        grouped_patterns.append((group_id, label, _unique_ranges(dlmix_ranges), _unique_ranges(ulmix_ranges)))
        grouped_ids.update(pattern_ids)

    for lp_id in genperf_ids:
        if lp_id not in grouped_ids:
            grouped_patterns.append(individual_patterns[lp_id])

    return grouped_patterns, individual_patterns


def _find_config_path() -> Path:
    """Resolve path to perf_pattern/perf_pattern_helper.yaml in nr_matlab/test."""
    script_dir = Path(__file__).resolve().parent
    nr_matlab = script_dir.parent.parent
    return nr_matlab / "test" / "perf_pattern" / "perf_pattern_helper.yaml"


# Pattern ranges loaded lazily from 5GModel/nr_matlab/test/perf_pattern/perf_pattern_helper.yaml
PATTERNS = None
INDIVIDUAL_PATTERNS = None


def _ensure_patterns_loaded() -> None:
    """Load PATTERNS from YAML once. Idempotent after first successful load."""
    global PATTERNS, INDIVIDUAL_PATTERNS
    if PATTERNS is not None:
        return
    config_path = _find_config_path()
    if not config_path.exists():
        msg = (
            f"Pattern config not found: {config_path}. "
            "Run from repo or ensure 5GModel/nr_matlab/test/perf_pattern/perf_pattern_helper.yaml exists."
        )
        raise FileNotFoundError(msg)
    PATTERNS, INDIVIDUAL_PATTERNS = _load_patterns_from_yaml(config_path)


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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
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


def _add_count_size(total: Tuple[int, float], count: int, size_mb: float) -> Tuple[int, float]:
    return total[0] + count, total[1] + size_mb


def run_report(all_files: Dict, other_tvnr: Dict, non_tvnr: List, parse_errors: List, skipped_lines: int) -> None:
    """Print the full table and summary to stdout."""
    _ensure_patterns_loaded()
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
    nrsim_dlmix_fapi = (0, 0.0)
    nrsim_dlmix_cuphy = (0, 0.0)
    nrsim_dlmix_other = (0, 0.0)
    nrsim_ulmix_fapi = (0, 0.0)
    nrsim_ulmix_cuphy = (0, 0.0)
    nrsim_ulmix_other = (0, 0.0)
    for (testtype, tc_num, file_type), (count, size_mb) in all_files.items():
        if (testtype, tc_num) not in accounted_tcs:
            is_nrsim_tv = tc_num >= NRSIM_TV_MIN
            if testtype == "DLMIX" and file_type == "FAPI":
                if is_nrsim_tv:
                    nrsim_dlmix_fapi = _add_count_size(nrsim_dlmix_fapi, count, size_mb)
                else:
                    other_dlmix_fapi = _add_count_size(other_dlmix_fapi, count, size_mb)
            elif testtype == "DLMIX" and file_type == "CUPHY":
                if is_nrsim_tv:
                    nrsim_dlmix_cuphy = _add_count_size(nrsim_dlmix_cuphy, count, size_mb)
                else:
                    other_dlmix_cuphy = _add_count_size(other_dlmix_cuphy, count, size_mb)
            elif testtype == "DLMIX" and file_type == "OTHER":
                if is_nrsim_tv:
                    nrsim_dlmix_other = _add_count_size(nrsim_dlmix_other, count, size_mb)
                else:
                    other_dlmix_other = _add_count_size(other_dlmix_other, count, size_mb)
            elif testtype == "ULMIX" and file_type == "FAPI":
                if is_nrsim_tv:
                    nrsim_ulmix_fapi = _add_count_size(nrsim_ulmix_fapi, count, size_mb)
                else:
                    other_ulmix_fapi = _add_count_size(other_ulmix_fapi, count, size_mb)
            elif testtype == "ULMIX" and file_type == "CUPHY":
                if is_nrsim_tv:
                    nrsim_ulmix_cuphy = _add_count_size(nrsim_ulmix_cuphy, count, size_mb)
                else:
                    other_ulmix_cuphy = _add_count_size(other_ulmix_cuphy, count, size_mb)
            elif testtype == "ULMIX" and file_type == "OTHER":
                if is_nrsim_tv:
                    nrsim_ulmix_other = _add_count_size(nrsim_ulmix_other, count, size_mb)
                else:
                    other_ulmix_other = _add_count_size(other_ulmix_other, count, size_mb)

    other_total_count = (
        other_dlmix_fapi[0] + other_dlmix_cuphy[0] + other_dlmix_other[0]
        + other_ulmix_fapi[0] + other_ulmix_cuphy[0] + other_ulmix_other[0]
    )
    other_total_size = (
        other_dlmix_fapi[1] + other_dlmix_cuphy[1] + other_dlmix_other[1]
        + other_ulmix_fapi[1] + other_ulmix_cuphy[1] + other_ulmix_other[1]
    )
    nrsim_total_count = (
        nrsim_dlmix_fapi[0] + nrsim_dlmix_cuphy[0] + nrsim_dlmix_other[0]
        + nrsim_ulmix_fapi[0] + nrsim_ulmix_cuphy[0] + nrsim_ulmix_other[0]
    )
    nrsim_total_size = (
        nrsim_dlmix_fapi[1] + nrsim_dlmix_cuphy[1] + nrsim_dlmix_other[1]
        + nrsim_ulmix_fapi[1] + nrsim_ulmix_cuphy[1] + nrsim_ulmix_other[1]
    )

    print("-" * 190)
    ns_fapi_s = f"{nrsim_dlmix_fapi[1] / 1024:.1f}GB" if nrsim_dlmix_fapi[1] > 0 else "-"
    ns_cuphy_s = f"{nrsim_dlmix_cuphy[1] / 1024:.1f}GB" if nrsim_dlmix_cuphy[1] > 0 else "-"
    ns_ul_fapi_s = f"{nrsim_ulmix_fapi[1] / 1024:.1f}GB" if nrsim_ulmix_fapi[1] > 0 else "-"
    ns_ul_cuphy_s = f"{nrsim_ulmix_cuphy[1] / 1024:.1f}GB" if nrsim_ulmix_cuphy[1] > 0 else "-"
    ns_st_str = f"{nrsim_total_size / 1024:.1f}GB" if nrsim_total_size > 0 else "-"
    ns_total_str = f"{nrsim_total_size / 1024:.2f} GB" if nrsim_total_size > 0 else "-"
    print(f"{'nrSim DLMIX/ULMIX TV>=20000':<30} | {nrsim_dlmix_fapi[0]:>8} {ns_fapi_s:>10} | {nrsim_dlmix_cuphy[0]:>8} {ns_cuphy_s:>10} | {nrsim_ulmix_fapi[0]:>8} {ns_ul_fapi_s:>10} | {nrsim_ulmix_cuphy[0]:>8} {ns_ul_cuphy_s:>10} | {nrsim_total_count:>8} {ns_st_str:>10} | {ns_total_str:>15}")

    o_fapi_s = f"{other_dlmix_fapi[1] / 1024:.1f}GB" if other_dlmix_fapi[1] > 0 else "-"
    o_cuphy_s = f"{other_dlmix_cuphy[1] / 1024:.1f}GB" if other_dlmix_cuphy[1] > 0 else "-"
    o_ul_fapi_s = f"{other_ulmix_fapi[1] / 1024:.1f}GB" if other_ulmix_fapi[1] > 0 else "-"
    o_ul_cuphy_s = f"{other_ulmix_cuphy[1] / 1024:.1f}GB" if other_ulmix_cuphy[1] > 0 else "-"
    o_st_str = f"{other_total_size / 1024:.1f}GB" if other_total_size > 0 else "-"
    o_total_str = f"{other_total_size / 1024:.2f} GB" if other_total_size > 0 else "-"
    print(f"{'Other DLMIX/ULMIX TV<20000':<30} | {other_dlmix_fapi[0]:>8} {o_fapi_s:>10} | {other_dlmix_cuphy[0]:>8} {o_cuphy_s:>10} | {other_ulmix_fapi[0]:>8} {o_ul_fapi_s:>10} | {other_ulmix_cuphy[0]:>8} {o_ul_cuphy_s:>10} | {other_total_count:>8} {o_st_str:>10} | {o_total_str:>15}")

    dlmix_ulmix_total_count = (
        totals["dlmix_fapi"][0] + totals["dlmix_cuphy"][0] + totals["dlmix_other"][0]
        + totals["ulmix_fapi"][0] + totals["ulmix_cuphy"][0] + totals["ulmix_other"][0]
        + nrsim_total_count + other_total_count
    )
    dlmix_ulmix_total_size = (
        totals["dlmix_fapi"][1] + totals["dlmix_cuphy"][1] + totals["dlmix_other"][1]
        + totals["ulmix_fapi"][1] + totals["ulmix_cuphy"][1] + totals["ulmix_other"][1]
        + nrsim_total_size + other_total_size
    )

    print("─" * 190)
    st_d_fapi = (totals["dlmix_fapi"][1] + nrsim_dlmix_fapi[1] + other_dlmix_fapi[1]) / 1024
    st_d_cuphy = (totals["dlmix_cuphy"][1] + nrsim_dlmix_cuphy[1] + other_dlmix_cuphy[1]) / 1024
    st_u_fapi = (totals["ulmix_fapi"][1] + nrsim_ulmix_fapi[1] + other_ulmix_fapi[1]) / 1024
    st_u_cuphy = (totals["ulmix_cuphy"][1] + nrsim_ulmix_cuphy[1] + other_ulmix_cuphy[1]) / 1024
    st_d_fapi_str = f"{st_d_fapi:.1f}GB"
    st_d_cuphy_str = f"{st_d_cuphy:.1f}GB"
    st_u_fapi_str = f"{st_u_fapi:.1f}GB"
    st_u_cuphy_str = f"{st_u_cuphy:.1f}GB"
    st_all_str = f"{dlmix_ulmix_total_size / 1024:.1f}GB"
    st_total_str = f"{dlmix_ulmix_total_size / 1024:.2f} GB"
    print(f"{'SUBTOTAL: DLMIX/ULMIX Files':<30} | {totals['dlmix_fapi'][0] + nrsim_dlmix_fapi[0] + other_dlmix_fapi[0]:>8} {st_d_fapi_str:>10} | {totals['dlmix_cuphy'][0] + nrsim_dlmix_cuphy[0] + other_dlmix_cuphy[0]:>8} {st_d_cuphy_str:>10} | {totals['ulmix_fapi'][0] + nrsim_ulmix_fapi[0] + other_ulmix_fapi[0]:>8} {st_u_fapi_str:>10} | {totals['ulmix_cuphy'][0] + nrsim_ulmix_cuphy[0] + other_ulmix_cuphy[0]:>8} {st_u_cuphy_str:>10} | {dlmix_ulmix_total_count:>8} {st_all_str:>10} | {st_total_str:>15}")

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
    ot_st_str = f"{other_tvnr_total / 1024:.1f}GB" if other_tvnr_total > 0 else "-"
    ot_total_str = f"{other_tvnr_total / 1024:.2f} GB" if other_tvnr_total > 0 else "-"
    print(f"{'Other TVnr_ (not DLMIX/ULMIX)':<30} | {other_tvnr_fapi_c:>8} {ot_fapi_str:>10} | {other_tvnr_cuphy_c:>8} {ot_cuphy_str:>10} | {'-':>8} {'-':>10} | {'-':>8} {'-':>10} | {other_tvnr_count:>8} {ot_st_str:>10} | {ot_total_str:>15}")

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
    pattern_total_count = dlmix_ulmix_total_count - nrsim_total_count - other_total_count
    pattern_total_size = dlmix_ulmix_total_size - nrsim_total_size - other_total_size
    print(f"   {'DLMIX/ULMIX in Defined Patterns:':<35} {pattern_total_count:>10} {pattern_total_size / 1024:>13.2f} GB")
    print(f"   {f'nrSim DLMIX/ULMIX TV >= {NRSIM_TV_MIN}:':<35} {nrsim_total_count:>10} {nrsim_total_size / 1024:>13.2f} GB")
    print(f"   {f'Other DLMIX/ULMIX TV < {NRSIM_TV_MIN}:':<35} {other_total_count:>10} {other_total_size / 1024:>13.2f} GB")
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
    _ensure_patterns_loaded()
    valid_ids = [p[0] for p in PATTERNS]
    patterns_by_id = {p[0]: p for p in PATTERNS}
    for lp_id, pattern in INDIVIDUAL_PATTERNS.items():
        if lp_id not in patterns_by_id:
            valid_ids.append(lp_id)
            patterns_by_id[lp_id] = pattern
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

    _ensure_patterns_loaded()

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
