#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# -*- coding: utf-8 -*-
"""
cuMAC 64T64R MU-MIMO channel-model Test Suite (GT-10416)

Generates and executes the standalone 64T64R MU-MIMO channel-model validation
sweep for the cuMAC multi-cell scheduler: it jointly exercises the GPU-based
3GPP 38.901 channel model, multi-cell interference, EESM PHY abstraction and
channel-estimation error modeling. Each combination runs the scheduler binary
(-t <slots> -l), then the cellStatAnalysis.py post-analysis; a case passes only
if both print their PASS line.

Usage:
  python3 cumac_64tr_test.py                    # Generate combinations only
  python3 cumac_64tr_test.py --execute          # Execute all combinations
  python3 cumac_64tr_test.py --execute 5        # Execute single combination (index 0005)
  python3 cumac_64tr_test.py --execute 10-20    # Execute range of combinations (0010-0020)
  python3 cumac_64tr_test.py --smoke --execute  # Execute the smoke subset
  python3 cumac_64tr_test.py --execute --log-dir /path/to/logs  # Custom log directory

Environment Variables:
  cuBB_SDK:           Path to cuBB SDK (default: /opt/nvidia/cuBB/)
  CUMAC_SIM_SLOTS:    Simulation slots for -t (default 1000)
  CUMAC_TEST_TIMEOUT: Per-combination timeout in seconds (default 3600)
  CUBB_BUILD_TIMEOUT: cuBB build timeout in seconds when build.<arch> is missing (default 7200)

Output Files:
- cumac_64tr_chanmodel_combinations.csv: Generated test combinations
- cumac_64tr_results.csv: Test execution results
- test_logs/ (or custom log directory): per-case folders with log, config.yaml,
  result H5 and the per-cell plot PNG, plus test_execution.log and
  test_summary_report.txt

Dependencies:
- System tools: yq (YAML manipulation), compute environment with the cuBB SDK
- Post-analysis: cellStatAnalysis.py (matplotlib; auto-installed if missing)
"""

import csv
import glob
import itertools
import platform
import shutil
import subprocess
import time
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Directory of this script - anchor the combination/result CSVs here so the same
# files are read/written regardless of the caller's working directory (the YAML
# slices and --execute <index> rely on a stable persisted index mapping).
SCRIPT_DIR = Path(__file__).resolve().parent


def load_combinations_from_csv(filename="cumac_64tr_chanmodel_combinations.csv"):
    """Load combinations from existing CSV file"""
    combinations = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            combinations.append(row)
    return combinations


def get_cubb_sdk_path():
    """Get cuBB SDK path from environment variable or use default"""
    cubb_sdk = os.environ.get('cuBB_SDK', '/opt/nvidia/cuBB/')
    if not cubb_sdk.endswith('/'):
        cubb_sdk += '/'
    return cubb_sdk


# Timeout (seconds) for the cuBB SDK build that runs when build.<arch> is
# missing; a full build is slow. Overridable via env CUBB_BUILD_TIMEOUT.
CUBB_BUILD_TIMEOUT = int(os.environ.get("CUBB_BUILD_TIMEOUT", "7200"))

# Timeout (seconds) for the one-off matplotlib install done when the container
# lacks it (cellStatAnalysis.py imports matplotlib at module load).
MATPLOTLIB_INSTALL_TIMEOUT = int(os.environ.get("MATPLOTLIB_INSTALL_TIMEOUT", "300"))


def get_cubb_build_dir():
    """Arch-specific cuBB build folder (build.x86_64 / build.aarch64).

    Mirrors the shell `build.$(uname -m)` layout via platform.machine().
    """
    return f"{get_cubb_sdk_path()}build.{platform.machine()}"


def ensure_cubb_build():
    """Build the cuBB SDK if the arch-specific build.<arch> folder is missing.

    The test executable lives under build.<arch>/; if that folder does not
    exist yet, run $cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh
    (from the SDK root) before executing any test.
    """
    build_dir = get_cubb_build_dir()
    if os.path.isdir(build_dir):
        return

    cubb_sdk = get_cubb_sdk_path()
    build_script = f"{cubb_sdk}testBenches/phase4_test_scripts/build_aerial_sdk.sh"
    if not os.path.exists(build_script):
        raise FileNotFoundError(f"cuBB build script not found: {build_script}")

    print(f"Build folder {build_dir} not found; building cuBB via {build_script} ...")
    logging.info("Building cuBB SDK (missing %s) via %s", build_dir, build_script)
    subprocess.run(["bash", build_script], cwd=cubb_sdk,
                   check=True, timeout=CUBB_BUILD_TIMEOUT)


def ensure_matplotlib() -> None:
    """Make matplotlib importable for the cellStatAnalysis.py post-analysis.

    The post-analysis is launched as ``python3 cellStatAnalysis.py`` and that
    script imports matplotlib at module load, so a container without it fails
    every case's post-analysis before any check runs. Probe the same ``python3``
    the post-analysis uses and install matplotlib once if it is missing. A
    failed install is logged (not raised) so the run surfaces the real
    post-analysis error per case instead of aborting the whole sweep.
    """
    probe = subprocess.run(["python3", "-c", "import matplotlib"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        return

    print("matplotlib not found for cellStatAnalysis.py post-analysis; installing ...")
    logging.info("Installing matplotlib for cellStatAnalysis.py post-analysis")
    try:
        subprocess.run(["python3", "-m", "pip", "install", "matplotlib"],
                       check=True, timeout=MATPLOTLIB_INSTALL_TIMEOUT,
                       capture_output=True, text=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logging.warning("matplotlib install failed (%s); post-analysis plots "
                        "may fail for every case", exc)


def check_test_result(stdout):
    """Check if the test output contains the PASS message"""
    return "Summary - cuMAC multi-cell MU-MIMO scheduler simulation test: PASS" in stdout


def setup_logging(log_dir="test_logs"):
    """Setup logging directory and configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'test_execution.log')),
            logging.StreamHandler()
        ]
    )

    return log_dir


def save_test_results(results, filename):
    """Save test results to CSV file"""
    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def generate_test_summary(results, log_dir="test_logs"):
    """Generate a comprehensive test summary report"""
    if not results:
        return

    summary_file = os.path.join(log_dir, "test_summary_report.txt")

    with open(summary_file, 'w') as f:
        f.write("cuMAC MIMO Test Execution Summary Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tests: {len(results)}\n")

        # Calculate statistics
        passed_tests = [r for r in results if r['test_passed']]
        failed_tests = [r for r in results if not r['test_passed']]
        total_time = sum(r['execution_time'] for r in results)

        f.write(f"Passed: {len(passed_tests)}\n")
        f.write(f"Failed: {len(failed_tests)}\n")
        f.write(f"Success Rate: {(len(passed_tests) / len(results) * 100):.1f}%\n")
        f.write(f"Total Execution Time: {total_time:.2f}s\n")
        f.write(f"Average Execution Time: {total_time / len(results):.2f}s\n")

        # Failed tests details
        if failed_tests:
            f.write("\nFailed Tests Details:\n")
            f.write("-" * 30 + "\n")
            for test in failed_tests:
                f.write(f"Index: {test.get('index', 'N/A')}\n")
                f.write(f"Parameters: {dict((k, v) for k, v in test.items() if k not in ['test_passed', 'execution_time', 'success', 'error_message', 'log_file'])}\n")
                f.write(f"Error: {test['error_message'][:200]}...\n" if len(test['error_message']) > 200 else f"Error: {test['error_message']}\n")
                f.write(f"Log File: {test.get('log_file', 'N/A')}\n")
                f.write("-" * 30 + "\n")

        # Performance analysis by cell count
        f.write("\nPerformance Analysis by Cell Count:\n")
        f.write("-" * 40 + "\n")
        cell_stats = {}
        for test in results:
            cell_count = test['nCell']
            if cell_count not in cell_stats:
                cell_stats[cell_count] = {'total': 0, 'passed': 0, 'total_time': 0}
            cell_stats[cell_count]['total'] += 1
            cell_stats[cell_count]['total_time'] += test['execution_time']
            if test['test_passed']:
                cell_stats[cell_count]['passed'] += 1

        for cell_count in sorted(cell_stats.keys()):
            stats = cell_stats[cell_count]
            success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            avg_time = stats['total_time'] / stats['total'] if stats['total'] > 0 else 0
            f.write(f"Cell Count {cell_count}: {stats['passed']}/{stats['total']} passed ({success_rate:.1f}%), "
                    f"Avg Time: {avg_time:.2f}s\n")

    print(f"Test summary report generated: {summary_file}")
    return summary_file


# =====================================================================
# GT-10416: cuMAC standalone 64T64R MU-MIMO channel-model validation
# ---------------------------------------------------------------------
# Jointly validates the GPU-based 3GPP 38.901 channel model, multi-cell
# inter-cell interference, EESM PHY abstraction and channel-estimation error
# modeling, integrated with the 64T64R MU-MIMO scheduler.
#
# =====================================================================

# Number of simulation slots (binary -t). Ticket default is 10000; uses
# 1000 for practical runtime. Overridable via env CUMAC_SIM_SLOTS.
CHANMODEL_SIM_SLOTS = int(os.environ.get("CUMAC_SIM_SLOTS", "1000"))

# Per-test wall-clock timeout (seconds). Large cell counts x 1000 slots can
# be slow; overridable via env CUMAC_TEST_TIMEOUT.
CHANMODEL_TEST_TIMEOUT = int(os.environ.get("CUMAC_TEST_TIMEOUT", "3600"))

# fading_type is fixed at 1 (statistic channel model) per the ticket.
CHANMODEL_FADING_TYPE = 1

# UE antenna panel (panel_1) layout keyed by nUeAnt, per Shaoran/West on the
# ticket:  4 ports -> [1,1,2,2,1];  2 ports -> [1,1,1,2,1]  (isotropic UE).
CHANMODEL_UE_ANT_SIZE = {
    4: [1, 1, 2, 2, 1],
    2: [1, 1, 1, 2, 1],
}

# Post-analysis QA tool (cellStatAnalysis.py). Shipped with the cuBB SDK at
# cuMAC/examples/multiCellMuMimoScheduler/ and used in place (the test no longer
# copies it into the container). Validates: (1) per-cell subplot sanity, (2) high TB-error ratio
# (smoothed TB-error > 0.2 on <= 10% of valid points), (3) scheduled-slot ratio
# >= 0.8 (via --apply-slot-ratio-threshold), and saves the per-cell plot PNG.
# Requires matplotlib inside the container.
CHANMODEL_SCRIPT_NAME = "cellStatAnalysis.py"

# Post-analysis pass indicator required by the ticket (the main-simulation PASS
# is checked via check_test_result).
CHANMODEL_POST_PASS = "Summary - cuMAC multi-cell MU-MIMO scheduler simulation post analysis test: PASS"

# Swept parameters (full coverage). fading_type is fixed at 1.
CHANMODEL_PARAMETERS = {
    "seed": [0, 100],
    "nCell": [1, 3, 6, 21],
    "nActiveUePerCell": [32, 64],
    "prbConfig": [[4, 68], [2, 68]],       # [nPrbPerGrp, nPrbGrp]
    "nUeAnt": [2, 4],
    "chanEstNmseDB": [-100.0, -15.0],
    "ut_drop_option": [2],
    "run_mode": [1],
}

# Smoke subset (--smoke): 68 combos = the small-cell full sweep below (1 & 3
# cells, 64 combos) PLUS four 6-cell sanity cases appended, so the smoke run
# exercises the same schema as the 128-combo full sweep at a fraction of the
# cost while still touching a multi-cell interference case. Both dicts reuse the
# full-sweep schema; generate_chanmodel_smoke_combinations() concatenates them.
CHANMODEL_SMOKE_PARAMETERS = {
    "seed": [0, 100],
    "nCell": [1, 3],
    "nActiveUePerCell": [32, 64],
    "prbConfig": [[4, 68], [2, 68]],       # [nPrbPerGrp, nPrbGrp]
    "nUeAnt": [2, 4],
    "chanEstNmseDB": [-100.0, -15.0],
    "ut_drop_option": [2],
    "run_mode": [1],
}

# Extra 6-cell sanity cases appended to the smoke subset (seed x NMSE).
CHANMODEL_SMOKE_EXTRA_PARAMETERS = {
    "seed": [0, 100],
    "nCell": [6],
    "nActiveUePerCell": [64],
    "prbConfig": [[4, 68]],       # [nPrbPerGrp, nPrbGrp]
    "nUeAnt": [4],
    "chanEstNmseDB": [-100.0, -15.0],
    "ut_drop_option": [2],
    "run_mode": [1],
}

# Scheduler parameters are not part of this sweep; they stay at config.yaml
# defaults and are re-pinned on every run (see update_chanmodel_config_file) so
# a value left in config.yaml by an earlier run can never leak in.

# Index order groups by (nCell, nActiveUePerCell, nPrbGrp, nUeAnt, run_mode) so
# the per-YAML slices are contiguous.
CHANMODEL_CSV = SCRIPT_DIR / "cumac_64tr_chanmodel_combinations.csv"
CHANMODEL_RESULTS_CSV = SCRIPT_DIR / "cumac_64tr_results.csv"

# The --smoke subset keeps its own combinations file so a smoke run never
# overwrites the full-sweep combinations (and vice versa); both modes write
# results to the same cumac_64tr_results.csv.
CHANMODEL_SMOKE_CSV = SCRIPT_DIR / "cumac_64tr_chanmodel_smoke_combinations.csv"
CHANMODEL_SMOKE_RESULTS_CSV = CHANMODEL_RESULTS_CSV


def generate_chanmodel_combinations(params=None):
    """Full-coverage (cartesian) combinations for the channel-model sweep.

    params defaults to CHANMODEL_PARAMETERS (the 128-combo full-coverage sweep).
    The --smoke subset is built by generate_chanmodel_smoke_combinations(), which
    calls this with the smoke param dicts. All dicts share the same schema.

    (nCell, nActiveUePerCell, prbConfig, nUeAnt, seed) are the outermost loops
    so each such group forms a contiguous index block -> one test YAML per group
    can run a clean 1-based index range (full sweep: 64 groups of 2, 0001-0128;
    naming <nCell>C_<nUe>UEperCell_<nPrbPerGrp>PrbPerGrp_<nUeAnt>UeAnt_RunMod1_Seed<seed>).
    Within each slice only chanEstNmseDB is swept.
    """
    params = CHANMODEL_PARAMETERS if params is None else params
    combos = []
    for nCell in params["nCell"]:
        for nUe in params["nActiveUePerCell"]:
            for prb in params["prbConfig"]:
                for nUeAnt in params["nUeAnt"]:
                    for seed in params["seed"]:
                        for nmse, utd, rmode in itertools.product(
                            params["chanEstNmseDB"],
                            params["ut_drop_option"],
                            params["run_mode"],
                        ):
                            combos.append({
                                "nCell": nCell,
                                "nActiveUePerCell": nUe,
                                "seed": seed,
                                "nPrbPerGrp": prb[0],
                                "nPrbGrp": prb[1],
                                "nUeAnt": nUeAnt,
                                "chanEstNmseDB": nmse,
                                "ut_drop_option": utd,
                                "run_mode": rmode,
                                "fading_type": CHANMODEL_FADING_TYPE,
                            })
    return combos


def generate_chanmodel_smoke_combinations():
    """Smoke combinations (--smoke): the small-cell full sweep (64) followed by
    four 6-cell sanity cases, for 68 combos total. The extra cases are appended
    so they land last in the CSV (indices 0065-0068).
    """
    return (generate_chanmodel_combinations(CHANMODEL_SMOKE_PARAMETERS)
            + generate_chanmodel_combinations(CHANMODEL_SMOKE_EXTRA_PARAMETERS))


def combo_signature(combo) -> tuple:
    """Normalized tuple of the sweep-defining fields for one combination.

    Used to compare a loaded CSV against the freshly generated sweep. CSV values
    are strings and generated values are typed, so both are coerced to a common
    form; 'index' is ignored. A missing field (e.g. an old-format CSV) raises
    KeyError, which the caller treats as a mismatch.
    """
    return (
        int(combo["nCell"]),
        int(combo["nActiveUePerCell"]),
        int(float(combo["seed"])),
        int(combo["nPrbPerGrp"]),
        int(combo["nPrbGrp"]),
        int(combo["nUeAnt"]),
        float(combo["chanEstNmseDB"]),
        int(combo["ut_drop_option"]),
        int(combo["run_mode"]),
        int(combo["fading_type"]),
    )


def save_chanmodel_to_csv(combinations, filename=CHANMODEL_CSV):
    """Save channel-model combinations preserving generation (grouped) order."""
    for i, combo in enumerate(combinations, 1):
        combo["index"] = f"{i:04d}"

    fieldnames = ["index"] + [k for k in combinations[0].keys() if k != "index"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for combo in combinations:
            writer.writerow(combo)

    print(f"Generated {len(combinations)} channel-model test cases saved to {filename}")

    # Print the per-group index ranges to make YAML wiring obvious.
    # nPrbGrp is fixed at 68, so nPrbPerGrp is the discriminating PRB
    # parameter; seed is part of the slice name.
    print("\nGroup index ranges (nCell x nActiveUePerCell x nPrbPerGrp x nUeAnt x seed):")
    groups = {}
    order = []
    for combo in combinations:
        key = (int(combo["nCell"]), int(combo["nActiveUePerCell"]),
               int(combo["nPrbPerGrp"]), int(combo["nUeAnt"]),
               int(combo["run_mode"]), int(float(combo["seed"])))
        idx = int(combo["index"])
        if key not in groups:
            order.append(key)
        lo, hi = groups.get(key, (idx, idx))
        groups[key] = (min(lo, idx), max(hi, idx))
    for key in order:
        lo, hi = groups[key]
        print(f"  {key[0]:>2}C_{key[1]:>2}UEperCell_{key[2]}PrbPerGrp_{key[3]}UeAnt_RunMod{key[4]}_Seed{key[5]}: "
              f"{lo:04d}-{hi:04d} ({hi - lo + 1} cases)")


def update_chanmodel_config_file(combo) -> None:
    """Update config.yaml for one channel-model combination via yq.

    Handles both top-level keys and nested channel_config.* keys, plus the
    coupled UE antenna panel (panel_1) layout derived from nUeAnt. Each edit is
    run as an argument list (no shell) so a config_path with spaces/special
    characters can never be mis-parsed.
    """
    cubb_sdk = get_cubb_sdk_path()
    config_path = f"{cubb_sdk}cuMAC/examples/multiCellMuMimoScheduler/config.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    n_ue_ant = int(combo["nUeAnt"])
    if n_ue_ant not in CHANMODEL_UE_ANT_SIZE:
        raise ValueError(f"Unsupported nUeAnt={n_ue_ant}; expected one of {list(CHANMODEL_UE_ANT_SIZE)}")
    ant_size = CHANMODEL_UE_ANT_SIZE[n_ue_ant]
    ant_size_yaml = "[" + ", ".join(str(x) for x in ant_size) + "]"

    # yq assignment expressions (passed as a single arg each; no shell).
    yq_exprs = [
        # deployment / traffic (top-level)
        f".seed = {combo['seed']}",
        # harqEnabled must be 0: the GT-10416 channel-model testbench has no
        # explicit HARQ handling (per ENG on the ticket); with the config
        # default of 1, scheduling stalls after the first few slots.
        ".harqEnabled = 0",
        ".targetMaxBlerMcs0 = 0.1",  # Can be disabled once MR!5468 is merged
        f".nCell = {combo['nCell']}",
        f".nActiveUePerCell = {combo['nActiveUePerCell']}",
        f".numUeForGrpPerCell = {combo['nActiveUePerCell']}",
        f".nMaxActUePerCell = {combo['nActiveUePerCell']}",
        f".nPrbPerGrp = {combo['nPrbPerGrp']}",
        f".nPrbGrp = {combo['nPrbGrp']}",
        f".nUeAnt = {n_ue_ant}",
        f".chanEstNmseDB = {combo['chanEstNmseDB']}",
        # scheduler parameters are not part of this sweep - re-pin them to the
        # config.yaml defaults on every run so a leftover value from an earlier
        # run can never leak in
        ".numUeSchdPerCellTTI = 16",
        ".nMaxLayerPerUeMuDl = 2",
        ".nMaxUegPerCellDl = 4",
        ".chanCorrThr = 0.7",
        # channel model (nested)
        f".channel_config.fading_type = {combo['fading_type']}",
        # isd / optional_pl_ind are not swept - re-pin to the config.yaml
        # defaults so a value written by an older sweep run can never leak in
        # (isd is ignored for UMa anyway).
        ".channel_config.system_level.isd = 1732.0",
        f".channel_config.system_level.ut_drop_option = {combo['ut_drop_option']}",
        ".channel_config.system_level.optional_pl_ind = 0",
        f".channel_config.simulation.run_mode = {combo['run_mode']}",
        # UE antenna panel (panel_1) coupled with nUeAnt
        f".channel_config.antenna_panels.panel_1.n_ant = {n_ue_ant}",
        f".channel_config.antenna_panels.panel_1.ant_size = {ant_size_yaml}",
    ]

    for expr in yq_exprs:
        try:
            subprocess.run(["yq", "-i", expr, config_path],
                           capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error updating config with '{expr}': {e}")
            print(f"Error output: {e.stderr}")
            raise


def run_chanmodel_test() -> tuple[bool, str, str]:
    """Run multiCellMuMimoScheduler for the channel-model sweep.

    Uses the ticket command:  -c config.yaml -t <slots> -l   (no sanitizer).
    Runs from the cuBB SDK root so relative paths (BLER LUT, H5 output)
    resolve the same way as a manual run. Returns (ok, stdout, stderr).

    Assumes the cuBB SDK is already built - execute_chanmodel_combinations()
    calls ensure_cubb_build() once up front, so a build failure aborts the run
    instead of being retried per combination.
    """
    cubb_sdk = get_cubb_sdk_path()
    test_executable = f"{get_cubb_build_dir()}/cuMAC/examples/multiCellMuMimoScheduler/multiCellMuMimoScheduler"
    config_path = f"{cubb_sdk}cuMAC/examples/multiCellMuMimoScheduler/config.yaml"

    if not os.path.exists(test_executable):
        raise FileNotFoundError(f"Test executable not found: {test_executable}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    test_cmd_args = [test_executable, "-c", config_path, "-t", str(CHANMODEL_SIM_SLOTS), "-l"]

    try:
        result = subprocess.run(test_cmd_args, cwd=cubb_sdk, capture_output=True,
                                text=True, timeout=CHANMODEL_TEST_TIMEOUT)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Test execution timed out after {CHANMODEL_TEST_TIMEOUT} seconds"


def chanmodel_combo_name(combo):
    """Full-combo name for the per-case folder: the leading part matches the
    owning test YAML (cuMAC_64TR_ChanModel_<slice>_Test), followed by the
    within-slice swept value (chanEstNmseDB), so the folder name alone fully
    identifies the configuration of a TC. int(float()) casts because
    CSV-loaded combos hold strings and nmse is stored as a float.
    """
    return (f"{int(combo['nCell'])}C_{int(combo['nActiveUePerCell'])}UEperCell_"
            f"{int(combo['nPrbPerGrp'])}PrbPerGrp_{int(combo['nUeAnt'])}UeAnt_"
            f"RunMod{int(combo['run_mode'])}_"
            f"Seed{int(float(combo['seed']))}_"
            f"Nmse{int(float(combo['chanEstNmseDB']))}")


def run_chanmodel_post_analysis(start_time, combo_index, case_dir):
    """Move this run's result H5 files into the per-case folder, tagged with
    the TC index, then post-analyze them.

    The binary always writes TV_cumac_result_64T64R_<cell>PC_DL.h5, so every
    combination would otherwise overwrite the previous one. Each freshly
    generated file is moved into case_dir (<log_dir>/TC<index>/) as
    TV_cumac_result_64T64R_<cell>PC_DL_TC<index>.h5 - co-located with the
    per-test log and the dumped config.yaml - before cellStatAnalysis.py
    runs on it (per GT-10416 / MR !5468):
        python3 cellStatAnalysis.py <h5> --apply-slot-ratio-threshold
                --save-plot --save-plot-dir <case_dir>
    The per-cell statistics plot lands in case_dir as <h5 stem>.png. Only
    files produced by the current run (mtime >= start_time) are used; moving
    them out of the SDK tree also prevents a later combination from
    re-picking a previous combo's TV.
    """
    cubb_sdk = get_cubb_sdk_path()
    post_analysis_script = f"{cubb_sdk}cuMAC/examples/multiCellMuMimoScheduler/{CHANMODEL_SCRIPT_NAME}"

    if not os.path.exists(post_analysis_script):
        return False, "", f"Post-analysis script not found: {post_analysis_script}"

    pattern = os.path.join(cubb_sdk, "**", "TV_cumac_result_64T64R_*PC_DL.h5")
    fresh = [f for f in glob.glob(pattern, recursive=True)
             if os.path.getmtime(f) >= start_time - 1]

    # Move into the per-case folder with the TC index so each combination's TV
    # is preserved, co-located with its log and config, for debugging.
    os.makedirs(case_dir, exist_ok=True)
    h5_files = []
    for f in sorted(set(fresh)):
        stem = os.path.splitext(os.path.basename(f))[0]   # TV_cumac_result_64T64R_<cell>PC_DL
        dest = os.path.join(case_dir, f"{stem}_TC{combo_index}.h5")
        try:
            if os.path.abspath(f) != os.path.abspath(dest):
                shutil.move(f, dest)
            h5_files.append(dest)
        except OSError:
            h5_files.append(f)                            # fall back to original location

    if not h5_files:
        return False, "", "No TV_cumac_result_64T64R_*PC_DL.h5 files found for post analysis"

    combined_out = []
    for h5 in h5_files:
        try:
            r = subprocess.run(["python3", post_analysis_script, h5,
                                "--apply-slot-ratio-threshold",
                                "--save-plot", "--save-plot-dir", case_dir],
                               capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return False, "\n".join(combined_out), f"Post analysis timed out for {h5}"
        combined_out.append(f"[{os.path.basename(h5)}]\n{r.stdout}\n{r.stderr}")
        # Require both the PASS line AND a clean exit, so a crash after the
        # PASS print (e.g. during plot saving) is not counted as a pass.
        if r.returncode != 0 or CHANMODEL_POST_PASS not in r.stdout:
            return False, "\n".join(combined_out), f"Post analysis did not PASS for {h5}"

    return True, "\n".join(combined_out), ""


def finalize_case_dir(case_dir, passed):
    """Rename the per-case folder to carry its verdict in the name.

    <log_dir>/TC<index>  ->  <log_dir>/TC<index>_PASS  or  TC<index>_FAIL
    so a directory listing immediately shows each combination's result. Any
    stale verdict folder from a previous rerun of the same index is removed
    first (latest verdict wins). Returns the final folder path (the original
    path if the rename fails).
    """
    final_dir = f"{case_dir}_{'PASS' if passed else 'FAIL'}"
    try:
        for stale in (f"{case_dir}_PASS", f"{case_dir}_FAIL"):
            if os.path.isdir(stale):
                shutil.rmtree(stale)
        os.rename(case_dir, final_dir)
        return final_dir
    except OSError as e:
        logging.warning(f"Could not rename case folder {case_dir}: {e}")
        return case_dir


def execute_chanmodel_combinations(combinations, results_file=CHANMODEL_RESULTS_CSV,
                                   start_index=None, end_index=None, log_dir="test_logs"):
    """Execute the channel-model sweep with per-combination logging.

    A combination passes only if BOTH the binary prints the main PASS line
    AND the post-analysis prints its PASS line on every generated H5.
    """
    log_dir = setup_logging(log_dir)

    # Build the cuBB SDK once up front if build.<arch> is missing. Doing it here
    # (not inside run_chanmodel_test) means a build failure aborts the whole run
    # rather than being retried - and timed out - for every combination.
    ensure_cubb_build()

    # Ensure matplotlib is available once up front so every case's post-analysis
    # (cellStatAnalysis.py) can run in containers that don't ship it.
    ensure_matplotlib()

    sorted_combinations = sorted(combinations, key=lambda x: int(x.get("index", "0")))

    if start_index is not None or end_index is not None:
        if start_index is None:
            start_index = 1
        if end_index is None:
            end_index = len(sorted_combinations)
        sorted_combinations = [c for c in sorted_combinations
                               if start_index <= int(c.get("index", "0")) <= end_index]
        print(f"Filtered to channel-model combinations with index range {start_index}-{end_index}")

    results = []
    total = len(sorted_combinations)
    print(f"Starting channel-model test execution for {total} combinations...")
    print("=" * 80)
    logging.info(f"Starting channel-model test execution for {total} combinations")

    for i, combo in enumerate(sorted_combinations, 1):
        combo_index = combo.get("index", f"{i:04d}")
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_id = f"{run_ts}_64TR_ChanModel_TC{combo_index}_test"

        print(f"\n[{i}/{total}] Testing channel-model combination (Index: {combo_index}):")
        print(f"  Cells: {combo['nCell']}, ActiveUE: {combo['nActiveUePerCell']}, seed: {combo['seed']}, "
              f"nUeAnt: {combo['nUeAnt']}, [nPrbPerGrp,nPrbGrp]: [{combo['nPrbPerGrp']},{combo['nPrbGrp']}], "
              f"chanEstNmseDB: {combo['chanEstNmseDB']}, "
              f"ut_drop_option: {combo['ut_drop_option']}, run_mode: {combo['run_mode']}")

        logging.info(f"Starting test {test_id}: {combo}")
        start_time = time.time()

        # Per-case folder: log, the exact config.yaml used, and result H5(s)
        # all live under <log_dir>/<timestamp>_TC<index>_<full-combo>/ for
        # self-contained debugging — the leading timestamp matches the log
        # file inside, then the TC index, then the combo (the slice part
        # matches the test YAML name, the swept values follow).
        # Latest run wins: remove folders from previous runs of this index
        # (in-progress or _PASS/_FAIL, legacy TC<idx>-first formats included)
        # so the listing stays one folder per index and H5s don't accumulate.
        for prev_dir in ([os.path.join(log_dir, f"TC{combo_index}")]
                         + glob.glob(os.path.join(log_dir, f"TC{combo_index}_*"))
                         + glob.glob(os.path.join(log_dir, f"*_TC{combo_index}_*"))):
            if os.path.isdir(prev_dir):
                shutil.rmtree(prev_dir, ignore_errors=True)
        case_dir = os.path.join(log_dir, f"{run_ts}_TC{combo_index}_{chanmodel_combo_name(combo)}")
        os.makedirs(case_dir, exist_ok=True)
        test_log_file = os.path.join(case_dir, f"{test_id}.log")

        try:
            logging.info(f"Updating config file for test {test_id}")
            update_chanmodel_config_file(combo)

            # Dump the fully-updated config.yaml for debugging/repro.
            config_path = f"{get_cubb_sdk_path()}cuMAC/examples/multiCellMuMimoScheduler/config.yaml"
            try:
                shutil.copy2(config_path, os.path.join(case_dir, "config.yaml"))
            except OSError as e:
                logging.warning(f"Could not dump config.yaml for {test_id}: {e}")

            logging.info(f"Running test executable for test {test_id}")
            success, stdout, stderr = run_chanmodel_test()

            main_passed = success and check_test_result(stdout)

            # Post-analysis only when the main simulation passed.
            if main_passed:
                post_passed, post_out, post_err = run_chanmodel_post_analysis(start_time, combo_index, case_dir)
            else:
                post_passed, post_out, post_err = False, "", "Skipped (main simulation did not PASS)"

            test_passed = bool(main_passed and post_passed)
            execution_time = time.time() - start_time

            with open(test_log_file, "w") as f:
                f.write(f"Test ID: {test_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Parameters: {combo}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Main PASS: {main_passed}   Post-analysis PASS: {post_passed}\n")
                f.write("=" * 80 + "\nSIMULATION STDOUT:\n")
                f.write(stdout)
                f.write("\n" + "=" * 80 + "\nSIMULATION STDERR:\n")
                f.write(stderr)
                f.write("\n" + "=" * 80 + "\nPOST-ANALYSIS OUTPUT:\n")
                f.write(post_out)
                if post_err:
                    f.write(f"\nPOST-ANALYSIS ERROR: {post_err}\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Return Code: {0 if test_passed else 1}\n")
                f.write(f"Execution Time: {execution_time:.2f}s\n")

            # Rename the case folder to TC<idx>_PASS / TC<idx>_FAIL.
            case_dir = finalize_case_dir(case_dir, test_passed)
            test_log_file = os.path.join(case_dir, os.path.basename(test_log_file))

            status = "PASS" if test_passed else "FAIL"
            print(f"  Result: {status} (main={main_passed}, post={post_passed}, Time: {execution_time:.2f}s)")
            print(f"  Case folder: {case_dir}")
            logging.info(f"Test {test_id} completed: {status} in {execution_time:.2f}s")

            if not test_passed:
                err_detail = stderr if not success else (post_err or "See log for details")
                print(f"  Error: {err_detail[:200]}")
                logging.error(f"Test {test_id} failed: {err_detail}")

            results.append({
                **combo,
                "main_passed": main_passed,
                "post_passed": post_passed,
                "test_passed": test_passed,
                "execution_time": round(execution_time, 2),
                "success": success,
                "error_message": (stderr if not success else post_err),
                "log_file": test_log_file,
            })

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  Result: FAIL (exception, Time: {execution_time:.2f}s)")
            print(f"  Error: {str(e)}")
            with open(test_log_file, "w") as f:
                f.write(f"Test ID: {test_id}\nParameters: {combo}\nERROR:\n{str(e)}\n")
            case_dir = finalize_case_dir(case_dir, False)
            test_log_file = os.path.join(case_dir, os.path.basename(test_log_file))
            logging.error(f"Test {test_id} encountered exception: {str(e)}")
            results.append({
                **combo,
                "main_passed": False,
                "post_passed": False,
                "test_passed": False,
                "execution_time": round(execution_time, 2),
                "success": False,
                "error_message": str(e),
                "log_file": test_log_file,
            })

        if i % 10 == 0:
            save_test_results(results, results_file)
            print(f"\nIntermediate results saved. Completed {i}/{total} tests.")

    save_test_results(results, results_file)
    generate_test_summary(results, log_dir)

    passed = sum(1 for r in results if r["test_passed"])
    print("\n" + "=" * 80)
    print("CHANNEL-MODEL TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    if results:
        print(f"Success rate: {(passed / len(results) * 100):.1f}%")
    print(f"Results saved to: {results_file}")


def main():
    """Main execution function"""

    # Parse command line arguments
    log_dir = "test_logs"
    start_index = None
    end_index = None
    execute_mode = False
    smoke = False

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--execute":
            execute_mode = True
            i += 1
            if i < len(sys.argv) and not sys.argv[i].startswith('--'):
                try:
                    if '-' in sys.argv[i]:
                        # Range format: start-end
                        start_str, end_str = sys.argv[i].split('-')
                        start_index = int(start_str)
                        end_index = int(end_str)
                    else:
                        # Single index
                        start_index = int(sys.argv[i])
                        end_index = start_index
                except ValueError:
                    print("Invalid index format. Use --execute [index] or --execute [start-end]")
                    print("Examples: --execute 5 or --execute 10-20")
                    sys.exit(1)
                i += 1
        elif sys.argv[i] == "--log-dir":
            i += 1
            if i < len(sys.argv):
                log_dir = sys.argv[i]
                i += 1
            else:
                print("Error: --log-dir requires a directory path")
                sys.exit(1)
        elif sys.argv[i] == "--smoke":
            smoke = True
            i += 1
        elif sys.argv[i] == "--help" or sys.argv[i] == "-h":
            print("Usage: python3 cumac_64tr_test.py [options]")
            print("Options:")
            print("  --execute [index|start-end]  Execute tests for specific combination(s)")
            print("  --log-dir <directory>        Specify log directory (default: test_logs)")
            print("  --smoke                      Channel-model smoke subset (68 combos; use with --execute)")
            print("  --help, -h                   Show this help message")
            print("")
            print("GT-10416 channel-model full coverage (128 combos):")
            print("validates the GPU 38.901 channel model, runs the binary with -t <slots> -l, then the")
            print("cellStatAnalysis.py post-analysis (--apply-slot-ratio-threshold --save-plot).")
            print("")
            print("Examples:")
            print("  python3 cumac_64tr_test.py                    # Generate channel-model combinations only")
            print("  python3 cumac_64tr_test.py --execute          # Execute all channel-model combinations")
            print("  python3 cumac_64tr_test.py --execute 1-8      # Execute channel-model index range 0001-0008")
            print("  python3 cumac_64tr_test.py --smoke --execute  # Execute the channel-model smoke subset")
            print("")
            print("Environment variables:")
            print("  CUMAC_SIM_SLOTS    Simulation slots for -t (default 1000; ticket default 10000)")
            print("  CUMAC_TEST_TIMEOUT Per-combination timeout in seconds (default 3600)")
            print("  CUBB_BUILD_TIMEOUT cuBB build timeout (s) when build.<arch> is missing (default 7200)")

            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            print("Use --help for usage information")
            sys.exit(1)

    # Display cuBB SDK path being used
    cubb_sdk = get_cubb_sdk_path()
    print(f"Using cuBB SDK path: {cubb_sdk}")

    # --smoke runs the 68-combo channel-model smoke subset (separate CSVs); the
    # default is the 128-combo full-coverage sweep. expected_combos is generated
    # up front (cheap, no I/O) and reused for the stale-CSV guard below.
    if smoke:
        active_csv = CHANMODEL_SMOKE_CSV
        active_results_csv = CHANMODEL_SMOKE_RESULTS_CSV
        expected_combos = generate_chanmodel_smoke_combinations()
        print("\nCHANNEL-MODEL SMOKE MODE (GT-10416): 68-combo channel-model subset")
    else:
        active_csv = CHANMODEL_CSV
        active_results_csv = CHANMODEL_RESULTS_CSV
        expected_combos = generate_chanmodel_combinations()
        print("\nCHANNEL-MODEL MODE (GT-10416): full coverage of the 38.901 channel-model sweep")
    print(f"  Simulation slots (-t): {CHANMODEL_SIM_SLOTS}   Per-test timeout: {CHANMODEL_TEST_TIMEOUT}s")

    expected_total = len(expected_combos)

    final_combinations = None
    if os.path.exists(active_csv):
        print(f"Loading existing channel-model combinations from {active_csv}...")
        final_combinations = load_combinations_from_csv(active_csv)
        # Stale guard: only reuse the CSV if it matches the current sweep exactly
        # - same combinations, same order. Comparing full normalized signatures
        # catches any parameter/ordering change (seeds, PRB, NMSE, ...) even when
        # the row count is unchanged, and an old-format CSV (missing columns)
        # falls through the KeyError/ValueError path to force regeneration.
        try:
            csv_matches = ([combo_signature(c) for c in final_combinations]
                           == [combo_signature(c) for c in expected_combos])
        except (KeyError, ValueError, TypeError):
            csv_matches = False
        if not csv_matches:
            print(f"Existing {active_csv} does not match the current sweep "
                  f"({expected_total} channel-model combos) - regenerating.")
            final_combinations = None
        else:
            print(f"Loaded {len(final_combinations)} existing combinations.")

    if final_combinations is None:
        final_combinations = expected_combos
        print(f"Total channel-model combinations generated: {len(final_combinations)}")
        save_chanmodel_to_csv(final_combinations, active_csv)

    if execute_mode:
        print("\n" + "=" * 80)
        print("CHANNEL-MODEL TEST EXECUTION MODE")
        print("=" * 80)
        if start_index is not None and end_index is not None:
            if start_index == end_index:
                print(f"Executing channel-model combination with index {start_index}")
            else:
                print(f"Executing channel-model combinations with index range {start_index}-{end_index}")
        else:
            print("Executing all channel-model combinations")
        execute_chanmodel_combinations(final_combinations, results_file=active_results_csv,
                                       start_index=start_index, end_index=end_index,
                                       log_dir=log_dir)
    else:
        print("\nTo execute channel-model tests, run:")
        print("  python3 cumac_64tr_test.py --execute            # all channel-model combinations")
        print("  python3 cumac_64tr_test.py --execute 1-8        # a single (nCell,UE) group range")

    print("Process completed successfully!")


if __name__ == "__main__":
    main()
