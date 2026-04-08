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
cuMAC MIMO Test Suite

This script generates and executes comprehensive test combinations for cuMAC multi-cell MU-MIMO scheduler.

Features:
- Automatic dependency installation (allpairspy, yq)
- Optimized test combination generation using pairwise testing
- Detailed test execution logging
- Individual test log files for each combination
- Comprehensive summary reports
- Support for partial test execution (by index or range)

Usage:
  python3 cumac_64tr_test.py                    # Generate test combinations only
  python3 cumac_64tr_test.py --execute         # Execute all test combinations
  python3 cumac_64tr_test.py --execute 5       # Execute single combination (index 0005)
  python3 cumac_64tr_test.py --execute 10-20   # Execute range of combinations (0010-0020)
  python3 cumac_64tr_test.py --execute --log-dir /path/to/logs  # Execute with custom log directory

Environment Variables:
  cuBB_SDK: Path to cuBB SDK (default: /opt/nvidia/cuBB/)

Output Files:
- cumac_64tr_combinations.csv: Generated test combinations
- ccumac_64tr_test.py.csv: Test execution results
- test_logs/ (or custom log directory): Directory containing detailed logs
  - test_execution.log: Main execution log
  - test_XXXX_YYYYMMDD_HHMMSS.log: Individual test logs
  - test_summary_report.txt: Comprehensive summary report

Dependencies:
- Python packages: allpairspy, pyyaml
- System tools: yq (for YAML manipulation)
"""

import csv
import subprocess
import time
import os
import sys
import logging
from datetime import datetime
try:
    from allpairspy import AllPairs
    import yaml
    print("✓ All required Python dependencies are already available")
except ImportError as e:
    print(f"Failed to import allpairspy or pyyaml: {e}")
    print("This might be due to:")
    print("0. Python package not installed")
    print("1. Python path issues")
    print("2. Virtual environment not activated")
    print("3. Package installed in different Python environment")
    sys.exit(1)

# Updated parameter definitions based on requirements
PARAMETERS = {
    "nCell": [1, 3, 6, 8, 10, 12, 16, 20, 24],  # Number of cells in simulation
    "nActiveUePerCell": [16, 32, 64, 128],   # Active UEs per cell (also used for grouping and max active)
    "numUeSchdPerCellTTI": [6, 8, 16],          # UEs scheduled per Transmission Time Interval
    "nPrbGrp": [34, 68],                     # Physical Resource Block groups
    "harqEnabled": [0, 1],                   # HARQ feature flag (0=disabled, 1=enabled)
    "nMaxLayerPerUeMuDl": [2, 4],            # Max layers per UE for DL MU-MIMO
    "nMaxUegPerCellDl": [1, 2, 4],           # Max UE groups per cell in downlink
    "chanCorrThr": [0.05, 0.1, 0.2, 0.4, 0.7, 0.9]  # Channel correlation threshold
}

# Constraint validation function


def is_valid_combination(values) -> bool:
    """Check if parameter combination meets system constraints"""
    if len(values) < 8:
        return True  # Allow partial combinations during generation

    nCell, nActiveUePerCell, numUeSchdPerCellTTI, nPrbGrp, harqEnabled, nMaxLayerPerUeMuDl, nMaxUegPerCellDl, chanCorrThr = values

    # Scheduled UEs must be >= UE groups
    if numUeSchdPerCellTTI < nMaxUegPerCellDl:
        return False

    return True


def generate_test_combinations():
    """Generate optimized test combinations based on specific requirements"""
    all_combinations = []

    # Requirement 1: numUeSchdPerCellTTI = 6 full coverage on 1,3,6,8,10,16,20,24 cells
    print("Generating Requirement 1: numUeSchdPerCellTTI = 6 full coverage on 1,3,6,8,10,16,12,20,24 cells...")
    for nCell in [1, 3, 6, 8, 10, 12, 16, 20, 24]:
        for nActiveUePerCell in PARAMETERS["nActiveUePerCell"]:
            for harqEnabled in PARAMETERS["harqEnabled"]:
                for nMaxLayerPerUeMuDl in PARAMETERS["nMaxLayerPerUeMuDl"]:
                    for nMaxUegPerCellDl in PARAMETERS["nMaxUegPerCellDl"]:
                        for nPrbGrp in PARAMETERS["nPrbGrp"]:
                            for chanCorrThr in PARAMETERS["chanCorrThr"]:
                                # Skip invalid combinations
                                if 6 < nMaxUegPerCellDl:  # numUeSchdPerCellTTI is fixed at 6
                                    continue

                                combo = {
                                    "nCell": nCell,
                                    "nActiveUePerCell": nActiveUePerCell,
                                    "numUeSchdPerCellTTI": 6,
                                    "nPrbGrp": nPrbGrp,
                                    "harqEnabled": harqEnabled,
                                    "nMaxLayerPerUeMuDl": nMaxLayerPerUeMuDl,
                                    "nMaxUegPerCellDl": nMaxUegPerCellDl,
                                    "chanCorrThr": chanCorrThr
                                }
                                all_combinations.append(combo)

    # Requirement 2: numUeSchdPerCellTTI = 8, nActiveUePerCell=128, full coverage on 12, 24 cells
    print("Generating Requirement 2: numUeSchdPerCellTTI = 8, nActiveUePerCell=128 full coverage on 12, 24 cells...")
    for nCell in [12, 24]:
        for harqEnabled in PARAMETERS["harqEnabled"]:
            for nMaxLayerPerUeMuDl in PARAMETERS["nMaxLayerPerUeMuDl"]:
                for nMaxUegPerCellDl in PARAMETERS["nMaxUegPerCellDl"]:
                    for nPrbGrp in PARAMETERS["nPrbGrp"]:
                        for chanCorrThr in PARAMETERS["chanCorrThr"]:
                            # Skip invalid combinations
                            if 8 < nMaxUegPerCellDl:  # numUeSchdPerCellTTI is fixed at 8
                                continue

                            combo = {
                                "nCell": nCell,
                                "nActiveUePerCell": 128,
                                "numUeSchdPerCellTTI": 8,
                                "nPrbGrp": nPrbGrp,
                                "harqEnabled": harqEnabled,
                                "nMaxLayerPerUeMuDl": nMaxLayerPerUeMuDl,
                                "nMaxUegPerCellDl": nMaxUegPerCellDl,
                                "chanCorrThr": chanCorrThr
                            }
                            all_combinations.append(combo)

    # Requirement 3: numUeSchdPerCellTTI = 16, nActiveUePerCell=128, full coverage on 6, 24 cells
    print("Generating Requirement 3: numUeSchdPerCellTTI = 16, nActiveUePerCell=128 full coverage on 6, 24 cells...")
    for nCell in [6, 24]:
        for harqEnabled in PARAMETERS["harqEnabled"]:
            for nMaxLayerPerUeMuDl in PARAMETERS["nMaxLayerPerUeMuDl"]:
                for nMaxUegPerCellDl in PARAMETERS["nMaxUegPerCellDl"]:
                    for nPrbGrp in PARAMETERS["nPrbGrp"]:
                        for chanCorrThr in PARAMETERS["chanCorrThr"]:
                            # Skip invalid combinations
                            if 16 < nMaxUegPerCellDl:  # numUeSchdPerCellTTI is fixed at 16
                                continue

                            combo = {
                                "nCell": nCell,
                                "nActiveUePerCell": 128,
                                "numUeSchdPerCellTTI": 16,
                                "nPrbGrp": nPrbGrp,
                                "harqEnabled": harqEnabled,
                                "nMaxLayerPerUeMuDl": nMaxLayerPerUeMuDl,
                                "nMaxUegPerCellDl": nMaxUegPerCellDl,
                                "chanCorrThr": chanCorrThr
                            }
                            all_combinations.append(combo)

    # Requirement 4: nMaxUegPerCellDl = 4, nActiveUePerCell=128, nPrbGrp = 68, numUeSchdPerCellTTI = 16, Spot Check on 1,3,8,10,12,16,20 Cells
    print("Generating Requirement 4: Spot check combinations...")
    for nCell in [1, 3, 8, 10, 12, 16, 20]:
        for harqEnabled in PARAMETERS["harqEnabled"]:
            for nMaxLayerPerUeMuDl in PARAMETERS["nMaxLayerPerUeMuDl"]:
                for chanCorrThr in PARAMETERS["chanCorrThr"]:
                    combo = {
                        "nCell": nCell,
                        "nActiveUePerCell": 128,
                        "numUeSchdPerCellTTI": 16,
                        "nPrbGrp": 68,
                        "harqEnabled": harqEnabled,
                        "nMaxLayerPerUeMuDl": nMaxLayerPerUeMuDl,
                        "nMaxUegPerCellDl": 4,
                        "chanCorrThr": chanCorrThr
                    }
                    all_combinations.append(combo)

    # Add additional pairwise combinations for comprehensive coverage
    print("Generating additional pairwise combinations...")
    pairwise_combinations = []

    # Generate pairwise combinations with constraint validation
    for combo in AllPairs(
        [
            PARAMETERS["nCell"],
            PARAMETERS["nActiveUePerCell"],
            PARAMETERS["numUeSchdPerCellTTI"],
            PARAMETERS["nPrbGrp"],
            PARAMETERS["harqEnabled"],
            PARAMETERS["nMaxLayerPerUeMuDl"],
            PARAMETERS["nMaxUegPerCellDl"],
            PARAMETERS["chanCorrThr"]
        ],
        filter_func=is_valid_combination
    ):
        pairwise_combinations.append({
            "nCell": combo[0],
            "nActiveUePerCell": combo[1],
            "numUeSchdPerCellTTI": combo[2],
            "nPrbGrp": combo[3],
            "harqEnabled": combo[4],
            "nMaxLayerPerUeMuDl": combo[5],
            "nMaxUegPerCellDl": combo[6],
            "chanCorrThr": combo[7]
        })

    # Add unique pairwise combinations
    for combo in pairwise_combinations:
        if combo not in all_combinations:
            all_combinations.append(combo)

    # Remove duplicates while preserving order
    seen = set()
    unique_combinations = []
    for combo in all_combinations:
        combo_tuple = tuple(sorted(combo.items()))
        if combo_tuple not in seen:
            seen.add(combo_tuple)
            unique_combinations.append(combo)

    return unique_combinations


def expand_combinations(combinations):
    """Expand combinations to include the redundant parameters for compatibility"""
    expanded_combinations = []

    for combo in combinations:
        expanded_combo = {
            "nCell": combo["nCell"],
            "nActiveUePerCell": combo["nActiveUePerCell"],
            "numUeForGrpPerCell": combo["nActiveUePerCell"],  # Same as nActiveUePerCell
            "nMaxActUePerCell": combo["nActiveUePerCell"],    # Same as nActiveUePerCell
            "numUeSchdPerCellTTI": combo["numUeSchdPerCellTTI"],
            "nPrbGrp": combo["nPrbGrp"],
            "harqEnabled": combo["harqEnabled"],
            "nMaxLayerPerUeMuDl": combo["nMaxLayerPerUeMuDl"],
            "nMaxUegPerCellDl": combo["nMaxUegPerCellDl"],
            "chanCorrThr": combo["chanCorrThr"]
        }
        expanded_combinations.append(expanded_combo)

    return expanded_combinations


def save_to_csv(combinations, filename="cumac_64tr_combinations.csv"):
    """Save generated test combinations to CSV file with proper sorting"""
    # Sort combinations by nCell, nActiveUePerCell, numUeSchdPerCellTTI, nPrbGrp
    sorted_combinations = sorted(combinations, key=lambda x: (x['nCell'], x['nActiveUePerCell'], x['numUeSchdPerCellTTI'], x['nPrbGrp']))

    # Add index to each combination
    for i, combo in enumerate(sorted_combinations, 1):
        combo['index'] = f"{i:04d}"

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['index'] + list(sorted_combinations[0].keys())[:-1]  # Put index first, remove the duplicate index
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for combo in sorted_combinations:
            writer.writerow(combo)

    print(f"Generated {len(sorted_combinations)} test cases saved to {filename}")


def load_combinations_from_csv(filename="cumac_64tr_combinations.csv"):
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


def update_config_file(combo):
    """Update the config.yaml file with the given combination parameters"""
    cubb_sdk = get_cubb_sdk_path()
    config_path = f"{cubb_sdk}cuMAC/examples/multiCellMuMimoScheduler/config.yaml"

    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Update each parameter using yq
    yq_commands = [
        f"yq -i '.nCell = {combo['nCell']}' {config_path}",
        f"yq -i '.nActiveUePerCell = {combo['nActiveUePerCell']}' {config_path}",
        f"yq -i '.numUeForGrpPerCell = {combo['numUeForGrpPerCell']}' {config_path}",
        f"yq -i '.nMaxActUePerCell = {combo['nMaxActUePerCell']}' {config_path}",
        f"yq -i '.numUeSchdPerCellTTI = {combo['numUeSchdPerCellTTI']}' {config_path}",
        f"yq -i '.nPrbGrp = {combo['nPrbGrp']}' {config_path}",
        f"yq -i '.harqEnabled = {combo['harqEnabled']}' {config_path}",
        f"yq -i '.nMaxLayerPerUeMuDl = {combo['nMaxLayerPerUeMuDl']}' {config_path}",
        f"yq -i '.nMaxUegPerCellDl = {combo['nMaxUegPerCellDl']}' {config_path}",
        f"yq -i '.chanCorrThr = {combo['chanCorrThr']}' {config_path}"
    ]

    for cmd in yq_commands:
        try:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error updating config: {e}")
            print(f"Command: {cmd}")
            print(f"Error output: {e.stderr}")
            raise


def run_test():
    """Run the multiCellMuMimoScheduler test"""
    cubb_sdk = get_cubb_sdk_path()
    test_executable = f"{cubb_sdk}build/cuMAC/examples/multiCellMuMimoScheduler/multiCellMuMimoScheduler"
    config_path = f"{cubb_sdk}cuMAC/examples/multiCellMuMimoScheduler/config.yaml"

    # Check if test executable exists
    if not os.path.exists(test_executable):
        raise FileNotFoundError(f"Test executable not found: {test_executable}")

    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Run the test with config file
    test_cmd_args = ["compute-sanitizer", "--tool", "memcheck", test_executable, "-c", config_path]

    try:
        result = subprocess.run(test_cmd_args, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test execution timed out after 5 minutes"
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


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


def execute_test_combinations(combinations, results_file="cumac_64tr_results.csv", start_index=None, end_index=None, log_dir="test_logs"):
    """Execute tests for combinations and save results with detailed logging"""
    # Setup logging
    log_dir = setup_logging(log_dir)

    # Sort combinations by nCell, nActiveUePerCell, numUeSchdPerCellTTI, nPrbGrp
    sorted_combinations = sorted(combinations, key=lambda x: (x['nCell'], x['nActiveUePerCell'], x['numUeSchdPerCellTTI'], x['nPrbGrp']))

    # Filter by index range if specified
    if start_index is not None or end_index is not None:
        if start_index is None:
            start_index = 1
        if end_index is None:
            end_index = len(sorted_combinations)

        # Convert index strings to integers for comparison
        filtered_combinations = []
        for combo in sorted_combinations:
            combo_index = int(combo.get('index', '0'))
            if start_index <= combo_index <= end_index:
                filtered_combinations.append(combo)

        sorted_combinations = filtered_combinations
        print(f"Filtered to combinations with index range {start_index}-{end_index}")

    results = []
    total_combinations = len(sorted_combinations)

    print(f"Starting test execution for {total_combinations} combinations...")
    print("=" * 80)
    logging.info(f"Starting test execution for {total_combinations} combinations")

    for i, combo in enumerate(sorted_combinations, 1):
        combo_index = combo.get('index', f"{i:04d}")
        test_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_64TR_TC{combo_index}_test"

        print(f"\n[{i}/{total_combinations}] Testing combination (Index: {combo_index}):")
        print(f"  Cells: {combo['nCell']}, Active UEs: {combo['nActiveUePerCell']}, "
              f"Scheduled: {combo['numUeSchdPerCellTTI']}, HARQ: {combo['harqEnabled']}, "
              f"Layers: {combo['nMaxLayerPerUeMuDl']}, Groups: {combo['nMaxUegPerCellDl']}, "
              f"PRB: {combo['nPrbGrp']}, Channel Corr: {combo['chanCorrThr']}")

        # Log test start
        logging.info(f"Starting test {test_id}: {combo}")

        start_time = time.time()

        # Create individual test log file
        test_log_file = os.path.join(log_dir, f"{test_id}.log")

        try:
            # Update config file
            logging.info(f"Updating config file for test {test_id}")
            update_config_file(combo)

            # Run test
            logging.info(f"Running test executable for test {test_id}")
            success, stdout, stderr = run_test()

            # Save detailed test output to individual log file
            with open(test_log_file, 'w') as f:
                f.write(f"Test ID: {test_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Parameters: {combo}\n")
                f.write("=" * 80 + "\n")
                f.write("STDOUT:\n")
                f.write(stdout)
                f.write("\n" + "=" * 80 + "\n")
                f.write("STDERR:\n")
                f.write(stderr)
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Return Code: {0 if success else 1}\n")
                f.write(f"Execution Time: {time.time() - start_time:.2f}s\n")

            # Check result
            test_passed = check_test_result(stdout) if success else False

            execution_time = time.time() - start_time

            # Store result
            result = {
                **combo,
                'test_passed': test_passed,
                'execution_time': round(execution_time, 2),
                'success': success,
                'error_message': stderr if not success else "",
                'log_file': test_log_file
            }
            results.append(result)

            # Print result
            status = "PASS" if test_passed else "FAIL"
            print(f"  Result: {status} (Time: {execution_time:.2f}s)")
            print(f"  Log file: {test_log_file}")

            logging.info(f"Test {test_id} completed: {status} in {execution_time:.2f}s")

            if not success:
                print(f"  Error: {stderr[:200]}..." if len(stderr) > 200 else f"  Error: {stderr}")
                logging.error(f"Test {test_id} failed: {stderr}")

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  Result: ERROR (Time: {execution_time:.2f}s)")
            print(f"  Error: {str(e)}")

            # Save error log
            with open(test_log_file, 'w') as f:
                f.write(f"Test ID: {test_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Parameters: {combo}\n")
                f.write("=" * 80 + "\n")
                f.write("ERROR:\n")
                f.write(str(e))
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Execution Time: {execution_time:.2f}s\n")

            logging.error(f"Test {test_id} encountered exception: {str(e)}")

            result = {
                **combo,
                'test_passed': False,
                'execution_time': round(execution_time, 2),
                'success': False,
                'error_message': str(e),
                'log_file': test_log_file
            }
            results.append(result)

        # Save intermediate results every 10 tests
        if i % 10 == 0:
            save_test_results(results, results_file)
            print(f"\nIntermediate results saved. Completed {i}/{total_combinations} tests.")
            logging.info(f"Intermediate results saved. Completed {i}/{total_combinations} tests.")

    # Save final results
    save_test_results(results, results_file)

    # Generate summary report
    generate_test_summary(results, log_dir)

    # Print summary
    passed_tests = sum(1 for r in results if r['test_passed'])
    failed_tests = len(results) - passed_tests

    print("\n" + "=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests / len(results) * 100):.1f}%")
    print(f"Results saved to: {results_file}")
    print(f"Detailed logs saved to: {log_dir}")
    print(f"Summary report: {os.path.join(log_dir, 'test_summary_report.txt')}")

    logging.info(f"Test execution completed. Passed: {passed_tests}, Failed: {failed_tests}, Success rate: {(passed_tests / len(results) * 100):.1f}%")


def select_smoke_subset(expanded_combinations):
    """
    Return a small, deterministic subset of combinations for smoke testing.
    Matches exactly the rows you listed (by parameter values, not by old indices).
    """
    def pick(c, *, nCell, harqEnabled, nMaxLayerPerUeMuDl, chanCorrThr):
        return (
            c["nCell"] == nCell and
            c["nActiveUePerCell"] == 128 and
            c["numUeForGrpPerCell"] == 128 and
            c["nMaxActUePerCell"] == 128 and
            c["numUeSchdPerCellTTI"] == 16 and
            c["nPrbGrp"] == 68 and
            c["nMaxUegPerCellDl"] == 4 and
            c["harqEnabled"] == harqEnabled and
            c["nMaxLayerPerUeMuDl"] == nMaxLayerPerUeMuDl and
            c["chanCorrThr"] == float(chanCorrThr)
        )

    targets = [
        # nCell = 1
        dict(nCell=1,  harqEnabled=1, nMaxLayerPerUeMuDl=2, chanCorrThr=0.9),
        dict(nCell=1,  harqEnabled=0, nMaxLayerPerUeMuDl=4, chanCorrThr=0.9),

        # nCell = 16
        dict(nCell=16, harqEnabled=0, nMaxLayerPerUeMuDl=2, chanCorrThr=0.9),
        dict(nCell=16, harqEnabled=1, nMaxLayerPerUeMuDl=2, chanCorrThr=0.9),

        # nCell = 20
        dict(nCell=20, harqEnabled=1, nMaxLayerPerUeMuDl=2, chanCorrThr=0.9),
        dict(nCell=20, harqEnabled=1, nMaxLayerPerUeMuDl=4, chanCorrThr=0.9),

        # nCell = 24 (one harq=0 row + a sweep of chanCorrThr with harq=1, layer=4)
        dict(nCell=24, harqEnabled=0, nMaxLayerPerUeMuDl=2, chanCorrThr=0.9),
        dict(nCell=24, harqEnabled=1, nMaxLayerPerUeMuDl=4, chanCorrThr=0.05),
        dict(nCell=24, harqEnabled=1, nMaxLayerPerUeMuDl=4, chanCorrThr=0.1),
        dict(nCell=24, harqEnabled=1, nMaxLayerPerUeMuDl=4, chanCorrThr=0.2),
        dict(nCell=24, harqEnabled=1, nMaxLayerPerUeMuDl=4, chanCorrThr=0.4),
        dict(nCell=24, harqEnabled=1, nMaxLayerPerUeMuDl=4, chanCorrThr=0.7),
        dict(nCell=24, harqEnabled=1, nMaxLayerPerUeMuDl=4, chanCorrThr=0.9),
    ]

    # preserve order of 'targets'
    picked = []
    for t in targets:
        # find first matching combo (there should be exactly one)
        match = next((c for c in expanded_combinations if pick(c, **t)), None)
        if match is not None:
            picked.append(match)
        else:
            # If a row is missing due to earlier filters, surface it clearly:
            print(f"⚠️  Smoke target not found: {t}")
    return picked


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
            print("  --smoke                      Generate a tiny smoke subset (~13 test cases)")
            print("  --help, -h                   Show this help message")
            print("")
            print("Examples:")
            print("  python3 cumac_64tr_test.py                    # Generate test combinations only")
            print("  python3 cumac_64tr_test.py --execute         # Execute all combinations")
            print("  python3 cumac_64tr_test.py --execute 5       # Execute combination with index 0005")
            print("  python3 cumac_64tr_test.py --execute 10-20   # Execute combinations with index range 0010-0020")
            print("  python3 cumac_64tr_test.py --execute --log-dir /path/to/logs  # Execute with custom log directory")
            print("  python3 cumac_64tr_test.py --execute --log-dir /path/to/logs --smoke  # Execute with custom log directory and smoke mode (13 test cases)")

            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            print("Use --help for usage information")
            sys.exit(1)

    # Display cuBB SDK path being used
    cubb_sdk = get_cubb_sdk_path()
    print(f"Using cuBB SDK path: {cubb_sdk}")

    csv_filename = "cumac_64tr_combinations.csv"

    # Check if combinations file already exists
    if os.path.exists(csv_filename):
        print(f"Loading existing combinations from {csv_filename}...")
        final_combinations = load_combinations_from_csv(csv_filename)
        print(f"Loaded {len(final_combinations)} existing combinations.")
    else:
        print("Generating optimized test combinations based on requirements...")

        # Generate optimized combinations
        optimized_combinations = generate_test_combinations()

        # Expand to include redundant parameters for compatibility
        final_combinations = expand_combinations(optimized_combinations)

        if smoke: 
            print("\nSMOKE MODE: selecting a small, representative subset...")
            final_combinations = select_smoke_subset(final_combinations)

        # Display generation summary
        print(f"Total test combinations generated: {len(final_combinations)}")

        # Show coverage breakdown
        req1_combinations = [c for c in final_combinations if c['numUeSchdPerCellTTI'] == 6 and c['nCell'] in [1, 3, 6, 8, 10, 12, 16, 20, 24]]
        req2_combinations = [c for c in final_combinations if c['numUeSchdPerCellTTI'] == 8 and c['nActiveUePerCell'] == 128 and c['nCell'] in [12, 24]]
        req3_combinations = [c for c in final_combinations if c['numUeSchdPerCellTTI'] == 16 and c['nActiveUePerCell'] == 128 and c['nCell'] in [12, 24]]
        req4_combinations = [c for c in final_combinations if c['nMaxUegPerCellDl'] == 4 and c['nActiveUePerCell'] == 128 and c['nPrbGrp'] == 68 and c['numUeSchdPerCellTTI'] == 16 and c['nCell'] in [1, 3, 8, 10, 12, 16, 20]]

        print(f"\nCoverage Summary:")
        print(f"- Requirement 1 (numUeSchdPerCellTTI=6, cells 1,3,6,8,10,16,20,24): {len(req1_combinations)} combinations")
        print(f"- Requirement 2 (numUeSchdPerCellTTI=8, nActiveUePerCell=128, cells 12,24): {len(req2_combinations)} combinations")
        print(f"- Requirement 3 (numUeSchdPerCellTTI=16, nActiveUePerCell=128, cells 6,24): {len(req3_combinations)} combinations")
        print(f"- Requirement 4 (Spot check with nMaxUegPerCellDl=4, nActiveUePerCell=128, nPrbGrp=68, numUeSchdPerCellTTI=16): {len(req4_combinations)} combinations")

        # Save to CSV
        save_to_csv(final_combinations, csv_filename)

    # Check if user wants to execute tests
    if execute_mode:
        print("\n" + "=" * 80)
        print("TEST EXECUTION MODE")
        print("=" * 80)

        # Display cuBB SDK path being used
        cubb_sdk = get_cubb_sdk_path()
        print(f"Using cuBB SDK path: {cubb_sdk}")

        if start_index is not None and end_index is not None:
            if start_index == end_index:
                print(f"Executing combination with index {start_index}")
            else:
                print(f"Executing combinations with index range {start_index}-{end_index}")
        else:
            print("Executing all combinations")

        print("This will update the config.yaml file and run tests for each combination.")
        print("Results will be saved to ccumac_64tr_test.py.csv")
        print(f"Detailed logs will be saved to {log_dir}/ directory")

        # Execute tests
        execute_test_combinations(final_combinations, start_index=start_index, end_index=end_index, log_dir=log_dir)
    else:
        print("\nTo execute tests, run:")
        print("  python3 cumac_64tr_test.py --execute                      # Execute all combinations")
        print("  python3 cumac_64tr_test.py --execute 5                   # Execute combination with index 0005")
        print("  python3 cumac_64tr_test.py --execute 10-20               # Execute combinations with index range 0010-0020")
        print("  python3 cumac_64tr_test.py --execute --log-dir /path/to/logs  # Execute with custom log directory")
        print("\nEnvironment variables:")
        print("  cuBB_SDK: Path to cuBB SDK (default: /opt/nvidia/cuBB/)")

    print("Process completed successfully!")


if __name__ == "__main__":
    main()
