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

import argparse
import glob
import os
import re
import sys
from collections import OrderedDict

from cicd_threshold_generate import generate_thresholds
from aerial_postproc.logparse import parse_tc_info

MMIMO_PATTERNS = [
    "66a", "66b", "66c", "66d",
    "67a", "67b", "67c", "67d",
    "69", "69a", "69b", "69c",
    "79",
    "81a", "81b", "81c", "81d",
    "87", "89",
]


def find_matching_template(template_dir, pattern, contains_eh):
    """
    Find the matching generic perf_requirements template from the absolute/ directory.

    Selects based on mMIMO (determined by pattern) and EH status.
    """
    mmimo = pattern.lower() in MMIMO_PATTERNS
    tr_suffix = "64tr" if mmimo else "4tr"
    eh_suffix = "eh" if contains_eh else "noneh"
    filename = f"perf_requirements_{tr_suffix}_{eh_suffix}.csv"
    filepath = os.path.join(template_dir, filename)
    if os.path.exists(filepath):
        return filepath
    return None


def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, "perf_requirements", "absolute")
    output_dir = args.output_folder

    if not os.path.isdir(template_dir):
        print(f"ERROR: Template directory not found: {template_dir}")
        return 1

    os.makedirs(output_dir, exist_ok=True)

    cicd_folders = sorted(glob.glob(os.path.join(args.input_folder, "F08_*")))

    if not cicd_folders:
        print(f"No F08_* folders found in {args.input_folder}")
        return 1

    print(f"Found {len(cicd_folders)} test case folders")
    print(f"Template directory: {template_dir}")
    print(f"Output directory:   {output_dir}")

    groups = OrderedDict()
    for cicd_folder in cicd_folders:
        perf_csv = os.path.join(cicd_folder, "cuphy", "perf_results", "perf.csv")
        if not (os.path.isdir(cicd_folder) and os.path.isfile(perf_csv)):
            continue
        tc_name = os.path.basename(cicd_folder)
        base_name = re.sub(r'_RUN\d+$', '', tc_name)
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(perf_csv)

    print(f"Grouped into {len(groups)} unique test case(s)")
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for base_name, perf_csvs in groups.items():
        cell_count, pattern, bfp, modcomp, contains_eh, contains_swdisableeh, contains_gc = parse_tc_info(base_name)

        eh_str = "_EH" if contains_eh else ""
        gc_str = "_GC" if contains_gc else ""
        comp_str = "MODCOMP" if modcomp else f"BFP{bfp}"

        platform_str = ""
        if args.gh:
            platform_str = "gh_"
        elif args.gl4:
            platform_str = "gl4_"

        template_path = find_matching_template(template_dir, pattern, contains_eh)
        if template_path is None:
            print(f"SKIP: No matching template for {base_name}")
            skip_count += 1
            continue

        gating_filename = f"gating_perf_requirements_{platform_str}F08_{pattern}_{comp_str}{eh_str}{gc_str}_{cell_count:02d}C.csv"
        warning_filename = f"warning_perf_requirements_{platform_str}F08_{pattern}_{comp_str}{eh_str}{gc_str}_{cell_count:02d}C.csv"

        gating_path = os.path.join(output_dir, gating_filename)
        warning_path = os.path.join(output_dir, warning_filename)

        print(f"Processing {base_name} ({len(perf_csvs)} run(s))...")
        print(f"  Template: {os.path.basename(template_path)}")
        print(f"  Gating:   {gating_filename}")
        print(f"  Warning:  {warning_filename}")

        result = generate_thresholds(
            perf_csvs,
            template_path,
            gating_path,
            warning_path,
        )

        if result == 0:
            success_count += 1
        else:
            fail_count += 1
            print(f"  FAILED to generate thresholds for {base_name}")

        print()

    print("=" * 60)
    print(f"Summary: {success_count} succeeded, {skip_count} skipped (no template), {fail_count} failed")
    print("=" * 60)

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generate gating/warning perf_requirements for all test cases"
    )
    parser.add_argument(
        "input_folder", type=str,
        help="Input folder containing CICD run folders (expects F08_* pattern)"
    )
    parser.add_argument(
        "output_folder", type=str,
        help="Output folder for generated gating/warning perf_requirements files"
    )
    parser.add_argument(
        "-g", "--gh", action="store_true",
        help="Use GH platform naming prefix"
    )
    parser.add_argument(
        "-l", "--gl4", action="store_true",
        help="Use GL4 platform naming prefix"
    )
    args = parser.parse_args()

    sys.exit(main(args))
