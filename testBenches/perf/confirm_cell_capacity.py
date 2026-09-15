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

"""Confirm cell capacity by testing one cell count at a time.

Algorithm:
  1. Run at --start N.
  2. If PASS  → try N+1, N+2, ... until FAIL (first fail is confirmed N+1)
  3. If FAIL  → try N-1, N-2, ... until PASS (that pass is confirmed capacity)
  4. After each run, rename the JSON to ..._<N>cells.json
  5. When done, combine all per-cell JSONs into one final JSON and run compare.py.

Usage (from SDK root):
  python3 testBenches/perf/confirm_cell_capacity.py \\
    --yaml 20260401_phase3CellCapacity_72eSplit/hwFEC_uciHW/cubb_gpu_test_config_perf103_72e.yaml \\
    --sm 8 50 72 68 8 \\
    --start 20 \\
    --freq 1610 --power 165 --slots 300 \\
    --output-dir results/phase3_72e_hwFEC_uciHW/1610MHz
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_file_name_str(sm_list: Sequence[str]) -> str:
    return "_".join(f"{int(s):03d}" for s in sm_list)


def _read_yaml_usecase(yaml_path: str) -> str:
    import yaml as _yaml
    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f) or {}
    return cfg.get("config", cfg).get("usecase", "F08")


def _run_measure(perf_dir: str, yaml_abs: str, sm_str: str, freq: int, power: int, N: int, slots: int,
                 max_mps_retries: int = 2, mps_retry_delay: int = 60) -> int:
    """Run measure.py for a single cell count N. Returns returncode.

    Retries automatically on CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED, which is
    a transient resource-exhaustion error from stale MPS connections.
    """
    import time as _time
    cmd = [
        sys.executable,
        os.path.join(perf_dir, "measure.py"),
        "--yaml", yaml_abs,
        "--target", *sm_str.split(),
        "--freq", str(freq),
        "--power", str(power),
        "--start", str(N),
        "--cap", str(N),
        "--slots", str(slots),
    ]

    for attempt in range(1 + max_mps_retries):
        if attempt > 0:
            print(f"\n  [retry {attempt}/{max_mps_retries}] Waiting {mps_retry_delay}s "
                  f"for MPS connections to clear...")
            _time.sleep(mps_retry_delay)

        print("\n" + "=" * 70)
        if attempt > 0:
            print(f"[run_measure] attempt {attempt+1}: " + " ".join(cmd))
        else:
            print("[run_measure] " + " ".join(cmd))
        print("=" * 70 + "\n")

        result = subprocess.run(cmd, capture_output=False, text=True, cwd=perf_dir)

        if result.returncode == 0:
            return 0

        # Check buffer file for MPS error signature to decide whether to retry
        # measure.py writes buffer-N.txt (not zero-padded)
        buf = os.path.join(perf_dir, f"buffer-{N}.txt")
        mps_error = False
        if os.path.isfile(buf):
            try:
                with open(buf) as f:
                    if "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED" in f.read():
                        mps_error = True
            except OSError:
                pass

        if not mps_error:
            print(f"  [run_measure] non-zero exit ({result.returncode}), not an MPS error — not retrying")
            return result.returncode

        print(f"  [run_measure] MPS error detected on attempt {attempt+1}")
        if attempt >= max_mps_retries:
            print(f"  [run_measure] exhausted {max_mps_retries} retries, giving up")
            return result.returncode

    return result.returncode  # unreachable


def _json_path(perf_dir: str, file_name_str: str, usecase: str) -> str:
    return os.path.join(perf_dir, f"{file_name_str}_sweep_graphs_avg_{usecase}.json")


def _cell_json_path(perf_dir: str, file_name_str: str, usecase: str, N: int) -> str:
    """Temporary per-cell JSON name."""
    return os.path.join(perf_dir, f"{file_name_str}_sweep_graphs_avg_{usecase}_{N}cells.json")


def _rename_to_cell_json(perf_dir: str, file_name_str: str, usecase: str, N: int) -> str:
    """Rename the latest measure.py output to the per-cell name. Returns new path."""
    src = _json_path(perf_dir, file_name_str, usecase)
    dst = _cell_json_path(perf_dir, file_name_str, usecase, N)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"  [rename] → {os.path.basename(dst)}")
    else:
        print(f"  [warn] expected JSON not found: {src}")
    return dst


def _check_pass(perf_dir: str, file_name_str: str, usecase: str, N: int) -> bool:
    """Check if cell count N passes (all ontimePercent == 1.0) in the per-cell JSON."""
    path = _cell_json_path(perf_dir, file_name_str, usecase, N)
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        data = json.load(f)
    # Try all plausible key formats: N+00, 0N+00 (zero-padded for N<10), N+0
    entry = None
    for key in dict.fromkeys([f"{N}+00", f"{N:02d}+00", f"{N}+0"]):
        entry = data.get(key)
        if entry is not None:
            break
    if entry is None:
        return False
    otp = entry.get("ontimePercent", {})
    vals = [v for v in otp.values() if v is not None]
    return bool(vals) and all(v == 1.0 for v in vals)


def _combine_cell_jsons(perf_dir: str, file_name_str: str, usecase: str, cell_list: Sequence[int]) -> Optional[str]:
    """Merge per-cell JSONs into one final JSON. Updates testConfig.start/cap."""
    combined: Dict[str, Any] = {}
    test_config: Optional[Dict[str, Any]] = None

    for N in sorted(cell_list):
        path = _cell_json_path(perf_dir, file_name_str, usecase, N)
        if not os.path.isfile(path):
            print(f"  [combine] missing {os.path.basename(path)}, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        if test_config is None:
            test_config = data.get("testConfig", {}).copy()
        # Merge cell entries
        for key, val in data.items():
            if key == "testConfig":
                continue
            combined[key] = val

    if test_config is None:
        print("  [combine] no data found, cannot combine")
        return None

    # Update start/cap to reflect the actual sweep range
    test_config["start"] = min(cell_list)
    test_config["cap"] = max(cell_list)
    combined["testConfig"] = test_config

    out_path = _json_path(perf_dir, file_name_str, usecase)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n  [combine] wrote {os.path.basename(out_path)} "
          f"(cells: {sorted(cell_list)})")
    return out_path


def _run_compare(perf_dir: str, file_name_str: str, usecase: str, cell_list: Sequence[int]) -> str:
    """Run compare.py with all tested cells."""
    sorted_cells = sorted(cell_list)
    cell_files = [os.path.basename(_cell_json_path(perf_dir, file_name_str, usecase, N)) for N in sorted_cells]
    cells_arg = " ".join(f"{N}+0" for N in sorted_cells)
    cmd = [
        sys.executable,
        os.path.join(perf_dir, "compare.py"),
        "--filenames",
        *cell_files,
        "--cells", *cells_arg.split(),
    ]
    print("\n" + "=" * 70)
    print("[compare] " + " ".join(cmd))
    print("=" * 70)
    now = datetime.datetime.now()
    result = subprocess.run(cmd, cwd=perf_dir)
    if result.returncode != 0:
        raise RuntimeError(f"compare.py failed with exit code {result.returncode}")
    plot_name = f"compare-{now.year}_{str(now.month).zfill(2)}_{str(now.day).zfill(2)}.png"
    plot_path = os.path.join(perf_dir, plot_name)
    if not os.path.isfile(plot_path):
        raise RuntimeError(f"compare.py completed but {plot_name} not found in {perf_dir}")
    return plot_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    """argparse type: int >= 1, else ArgumentTypeError."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"expected int, got {value!r}") from e
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {ivalue}")
    return ivalue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Confirm cell capacity by testing one cell count at a time."
    )
    parser.add_argument("--yaml", required=True,
                        help="Path to YAML config (relative to SDK root or absolute)")
    parser.add_argument("--sm", required=True, nargs="+",
                        help="SM allocation per sub-context")
    parser.add_argument("--start", required=True, type=_positive_int,
                        help="Starting cell count estimate N (>=1)")
    parser.add_argument("--freq", type=int, default=None,
                        help="GPU frequency MHz (overrides YAML)")
    parser.add_argument("--power", type=int, default=None,
                        help="GPU power limit W (overrides YAML)")
    parser.add_argument("--slots", type=int, default=300,
                        help="Slots per run (default: 300)")
    parser.add_argument("--max", type=_positive_int, default=None, dest="max_cells",
                        help="Maximum cell count to test when stepping up (>=1)")
    parser.add_argument("--output-dir", default=None,
                        help="Copy final JSON and plots to this directory")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip compare.py plot generation")
    args = parser.parse_args()

    # Resolve paths
    cubb_sdk = os.environ.get("cuBB_SDK", "/opt/nvidia/cuBB")
    yaml_abs = args.yaml if os.path.isabs(args.yaml) else os.path.join(cubb_sdk, args.yaml)
    perf_dir = os.path.join(cubb_sdk, "testBenches", "perf")

    # Read YAML defaults
    import yaml as _yaml
    with open(yaml_abs) as f:
        cfg = _yaml.safe_load(f) or {}
    yaml_cfg = cfg.get("config", cfg)
    freq: int = args.freq if args.freq is not None else yaml_cfg.get("freq", 1610)
    power: int = args.power if args.power is not None else yaml_cfg.get("power", 165)
    usecase = yaml_cfg.get("usecase", "F08")
    fix_ul: int = yaml_cfg.get("fix_ul_cell_count")
    fix_dl: int = yaml_cfg.get("fix_dl_cell_count")
    if fix_ul is None:
        fix_ul = -1
    if fix_dl is None:
        fix_dl = -1

    def _capacity_label(n: int) -> str:
        if fix_ul > 0 and fix_dl > 0:
            return f"{n} cells (DL {fix_dl}C fixed / UL {fix_ul}C fixed)"
        elif fix_ul > 0:
            return f"{n} cells (DL {n}C / UL {fix_ul}C fixed)"
        elif fix_dl > 0:
            return f"{n} cells (DL {fix_dl}C fixed / UL {n}C)"
        else:
            return f"{n} cells (DL {n}C / UL {n}C)"

    sm_str = " ".join(str(s) for s in args.sm)
    file_name_str = _derive_file_name_str([str(s) for s in args.sm])
    # measure.py (measure/TDD/run.py) appends "_gc" to the output prefix when the
    # YAML enables green contexts; mirror it or every JSON lookup below misses.
    if yaml_cfg.get("use_green_contexts"):
        file_name_str += "_gc"
    N = args.start
    if args.max_cells is not None and args.max_cells < N:
        parser.error("--max must be greater than or equal to --start")
    max_cells = args.max_cells if args.max_cells is not None else max(N, min(N * 3, 128))
    slots = args.slots

    print("=" * 70)
    print("Confirm Cell Capacity")
    print("=" * 70)
    print(f"  YAML        : {yaml_abs}")
    print(f"  SM          : {sm_str}")
    print(f"  Start       : {N}")
    print(f"  Max cells   : {max_cells}")
    print(f"  Freq / Power: {freq} MHz / {power} W")
    print(f"  Slots       : {slots}")
    print(f"  File prefix : {file_name_str}")
    print("=" * 70)

    tested_cells: list[int] = []
    confirmed_capacity: Optional[int] = None

    # --- Step 1: Test at N ---
    print(f"\n--- Testing {N} cells ---")
    rc = _run_measure(perf_dir, yaml_abs, sm_str, freq, power, N, slots)
    if rc != 0:
        raise RuntimeError(f"measure.py failed for {N} cells with exit code {rc}")
    _rename_to_cell_json(perf_dir, file_name_str, usecase, N)
    tested_cells.append(N)
    initial_pass = _check_pass(perf_dir, file_name_str, usecase, N)

    if initial_pass:
        print(f"  -> PASS at {N} cells")
        # Step up until FAIL
        current = N
        while current < max_cells:
            current += 1
            print(f"\n--- Testing {current} cells (stepping up) ---")
            rc = _run_measure(perf_dir, yaml_abs, sm_str, freq, power, current, slots)
            if rc != 0:
                raise RuntimeError(f"measure.py failed for {current} cells with exit code {rc}")
            _rename_to_cell_json(perf_dir, file_name_str, usecase, current)
            tested_cells.append(current)
            if _check_pass(perf_dir, file_name_str, usecase, current):
                print(f"  -> PASS at {current} cells, continuing up")
            else:
                print(f"  -> FAIL at {current} cells")
                confirmed_capacity = current - 1
                break
        if confirmed_capacity is None:
            print(f"  -> Reached max cell limit ({max_cells}) without failure")
            confirmed_capacity = current
    else:
        print(f"  -> FAIL at {N} cells")
        # Step down until PASS
        current = N
        while True:
            current -= 1
            if current < 1:
                print("  -> Cannot go below 1 cell, no capacity found")
                confirmed_capacity = 0
                break
            print(f"\n--- Testing {current} cells (stepping down) ---")
            rc = _run_measure(perf_dir, yaml_abs, sm_str, freq, power, current, slots)
            if rc != 0:
                raise RuntimeError(f"measure.py failed for {current} cells with exit code {rc}")
            _rename_to_cell_json(perf_dir, file_name_str, usecase, current)
            tested_cells.append(current)
            if _check_pass(perf_dir, file_name_str, usecase, current):
                print(f"  -> PASS at {current} cells")
                confirmed_capacity = current
                break
            else:
                print(f"  -> FAIL at {current} cells, stepping down")

    print(f"\n{'=' * 70}")
    print(f"  Confirmed cell capacity : {_capacity_label(confirmed_capacity)}")
    print(f"  Cells tested            : {sorted(tested_cells)}")
    print(f"{'=' * 70}")

    # --- Step 2: Combine per-cell JSONs ---
    final_json = _combine_cell_jsons(perf_dir, file_name_str, usecase, tested_cells)

    # --- Step 3: Compare plot ---
    compare_plot: Optional[str] = None
    if not args.no_compare and final_json:
        compare_plot = _run_compare(perf_dir, file_name_str, usecase, sorted(tested_cells))

    # --- Step 4: Copy to output dir ---
    if args.output_dir and final_json:
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy2(final_json, args.output_dir)
        for N_c in tested_cells:
            p = _cell_json_path(perf_dir, file_name_str, usecase, N_c)
            if os.path.isfile(p):
                shutil.copy2(p, args.output_dir)
        if compare_plot:
            shutil.copy2(compare_plot, args.output_dir)
        print(f"\n  Copied results to: {args.output_dir}")

    print(f"\nDone. Cell capacity = {confirmed_capacity} ({_capacity_label(confirmed_capacity)})")
    return confirmed_capacity


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result is not None else 1)
