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

import subprocess
import fileinput
import os
import sys
import itertools
from datetime import datetime
import argparse
import csv
import glob
import re
import logging
import yaml
from logging.handlers import RotatingFileHandler
from pathlib import Path
from generate_tv import TVGenerator


class cuMACTest:
    def __init__(self, config):
        self.config = config
        self.cumac_folder = Path(f"{self.config['cubb_sdk']}/cuMAC")
        self.build_path = Path(self.config['cubb_sdk'])/"build"
        self.param_file = self.cumac_folder / "examples/parameters.h"
        self.param_file_bak = self.param_file.with_suffix(".h_bak")
        if self.config["option"] in ["f1", "f2", "f3", "f4"]:
            self._cpu_gpu_perf_gap_targets = {
                "cpuGpuPerfGapPerUeConst": "0.05",
                "cpuGpuPerfGapSumRConst": "0.05",
            }
            self.csv_filename = (
                self.cumac_folder / "scripts/cumac_tdl_tv_parameters.csv"
            )
            if self.config["option"] in ["f1", "f2"]:
                self.log_file = (
                    Path(config["log_folder"])
                    / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_tdl_"
                    f"{self.config['option']}_test_{self.config['tv_index']}_main.log"
                )
            elif self.config["option"] in ["f3", "f4"]:
                self.log_file = (
                    Path(config["log_folder"])
                    / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_cdl_"
                    f"{self.config['option']}_test_{self.config['tv_index']}_main.log"
                )
        elif self.config["option"] == "64tr":
            self.csv_filename = (
                self.cumac_folder / "scripts/cumac_mimo_tv_parameters.csv"
            )
            self.log_file = (
                Path(config["log_folder"])
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_"
                f"{self.config['option']}_test_{self.config['tv_index']}_main.log"
            )
        elif self.config["option"] in ["drl", "srs"]:
            self.log_file = (
                Path(config["log_folder"])
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_{self.config['option']}_{self.config['tv_index']}_test_main.log"
            )
        elif self.config["option"] == "pfmsort":
            self.csv_filename = (
                self.cumac_folder / "scripts/cumac_pfmsort_parameters.csv"
            )
            self.pfmsort_config_file = self.cumac_folder / "examples/pfmSort/config.yaml"
            self.pfmsort_config_file_bak = self.pfmsort_config_file.with_suffix(".yaml_bak")
            self.log_file = (
                Path(config["log_folder"])
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_pfmsort_{self.config['tv_index']}_test_main.log"
            )
        else:
            self.csv_filename = self.cumac_folder / "scripts/cumac_tv_parameters.csv"
            self.log_file = (
                Path(config["log_folder"])
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_"
                f"{self.config['option']}_{self.config['test']}_test_{self.config['tv_index']}_main.log"
            )
            self.tv_name = f"TV_cuMAC_{self.config['antenna']}T{self.config['antenna']}R_{self.config['direction']}_TC{self.config['tv_index']}"
            self.tv_file = os.path.join(
                f"{self.config['tv_folder']}/{self.config['antenna']}T{self.config['antenna']}R",
                f"{self.tv_name}.h5",
            )
        self.logger = self._setup_logger()
        print(f"Log file initialized at: {self.log_file}")
        #self.build_command = f"cd {self.cumac_folder} && {self.config['cmake']}"
        self.build_command = f"cd {self.config['cubb_sdk']} && {self.config['cmake']}"
        self.check_command = f"grep 'nBsAntConst\|nUeAntConst\|numSimChnRlz\|gpuDeviceIdx\|seedConst\|numCellConst\|numUePerCellConst\|numActiveUePerCellConst\|nPrbsPerGrpConst\|numUePerCellConst\|gpuAllocTypeConst\|cpuAllocTypeConst\|prdSchemeConst\|rxSchemeConst\|nPrbGrpsConst\|cpuGpuPerfGapPerUeConst\|cpuGpuPerfGapSumRConst' {self.param_file}"
        self.UL_SUPPORTED_INDICES = [
            "0004", "0008", "0012", "0016", "0020", "0024", "0028", "0032"]
        self.UE_500_INDICES = ["0005", "0006", "0007", "0008", "0013", "0014", "0015",
                               "0016", "0021", "0022", "0023", "0024", "0029", "0030", "0031", "0032"]

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        # Clear any existing handlers to prevent duplicates
        logger.handlers.clear()
        logger.setLevel(logging.INFO)

        handler = RotatingFileHandler(
            self.log_file, maxBytes=1000000, backupCount=5)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(handler)

        # Prevent logging from propagating to the root logger
        logger.propagate = False

        return logger

    def run_subprocess(self, command, log_file=None, shell=True):
        """Run a subprocess command and handle errors with real-time logging."""
        try:
            process = subprocess.Popen(
                command,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            stdout, stderr = [], []

            with open(log_file, "a") if log_file else open(os.devnull, "w") as f:
                for line in process.stdout:
                    print(line, end="")  # Print to console
                    f.write(line)  # Write to log file
                    stdout.append(line)
                for line in process.stderr:
                    print(line, end="")  # Print to console
                    f.write(line)  # Write to log file
                    stderr.append(line)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    command,
                    output="".join(stdout),
                    stderr="".join(stderr),
                )
            return "".join(stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {command}\n{e.stderr}")
            print(f"Return code: {e.returncode}")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 1

    def tdl_test_parameters_comb(self):
        if self.config["antenna"] == "4":
            nBsAntConst = [4]
            nUeAntConst = [4]
        elif self.config["antenna"] == "64":
            nBsAntConst = [64]
            nUeAntConst = [64]

        numSimChnRlz = [10, 500]
        gpuDeviceIdx = [self.config["gpu_id"]]
        numCellConst = [10, 20]
        numActiveUePerCellConst = [100, 500]
        gpuAllocTypeConst = [0, 1]
        cpuAllocTypeConst = [0, 1]

        all_combinations = list(
            itertools.product(
                nBsAntConst,
                nUeAntConst,
                numCellConst,
                numActiveUePerCellConst,
                numSimChnRlz,
                gpuAllocTypeConst,
                cpuAllocTypeConst,
                gpuDeviceIdx,
            )
        )

        with open(self.csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Index",
                    "nBsAntConst",
                    "nUeAntConst",
                    "numCellConst",
                    "numActiveUePerCellConst",
                    "numSimChnRlz",
                    "gpuAllocTypeConst",
                    "cpuAllocTypeConst",
                    "gpuDeviceIdx",
                ]
            )
            i = 1
            for combo in all_combinations:
                if combo[5] != combo[6]:
                    continue
                padded_index = f"{i:04}"
                csv_writer.writerow([padded_index] + list(combo))
                i += 1

        self.logger.info(
            f"The parameters and values mapping table has been written to {self.csv_filename}."
        )
        return all_combinations

    def mimo_test_parameters_comb(self):
        if self.config["antenna"] == "64":
            nBsAntConst = [64]
            nUeAntConst = [4]

        # numSimChnRlz = [10, 500]
        gpuDeviceIdx = [self.config["gpu_id"]]
        # numCellConst = [2]
        numCellConst = [3]
        numActiveUePerCellConst = [100]
        gpuAllocTypeConst = [1]
        nPrbsPerGrpConst = [2]
        nPrbGrpsConst = [136]
        # cpuAllocTypeConst = [0, 1]

        all_combinations = list(
            itertools.product(
                nBsAntConst,
                nUeAntConst,
                numCellConst,
                numActiveUePerCellConst,
                gpuAllocTypeConst,
                nPrbsPerGrpConst,
                nPrbGrpsConst,
                gpuDeviceIdx,
            )
        )

        with open(self.csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Index",
                    "nBsAntConst",
                    "nUeAntConst",
                    "numCellConst",
                    "numActiveUePerCellConst",
                    "gpuAllocTypeConst",
                    "nPrbsPerGrpConst",
                    "nPrbGrpsConst",
                    "gpuDeviceIdx",
                ]
            )
            i = 1
            for combo in all_combinations:
                padded_index = f"{i:04}"
                csv_writer.writerow([padded_index] + list(combo))
                i += 1

        self.logger.info(
            f"The parameters and values mapping table has been written to {self.csv_filename}."
        )
        return all_combinations

    def pfmsort_test_parameters_comb(self):
        """Generate parameter combinations for PFM sort test."""
        num_cells = [1, 8, 16, 32, 40]
        num_ue_per_cell = [16, 64, 128, 256, 512]
        num_dl_lc_per_ue = [2, 4]
        num_ul_lcg_per_ue = [2, 4]
        if self.config["smoke"]:
            seeds = [0, 1, 2]
        else:
            seeds = [0, 1, 2, 3, 4, 5]

        all_combinations = list(
            itertools.product(
                num_cells,
                num_ue_per_cell,
                num_dl_lc_per_ue,
                num_ul_lcg_per_ue,
                seeds,
            )
        )

        with open(self.csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Index",
                    "NUM_CELL",
                    "NUM_UE_PER_CELL",
                    "NUM_DL_LC_PER_UE",
                    "NUM_UL_LCG_PER_UE",
                    "SEED",
                ]
            )
            for i, combo in enumerate(all_combinations, start=1):
                padded_index = f"{i:04}"
                csv_writer.writerow([padded_index] + list(combo))

        self.logger.info(
            f"Generated {len(all_combinations)} parameter combinations written to {self.csv_filename}."
        )
        return all_combinations

    def update_pfmsort_config(self, params):
        """Update the PFM sort config.yaml file with test parameters."""
        self.logger.info(f"Updating PFM sort config file: {self.pfmsort_config_file}")

        try:
            # Read the existing config file if it exists
            if os.path.exists(self.pfmsort_config_file):
                with open(self.pfmsort_config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}

            # Update parameters
            config_data['NUM_CELL'] = int(params['NUM_CELL'])
            config_data['NUM_UE_PER_CELL'] = int(params['NUM_UE_PER_CELL'])
            config_data['NUM_DL_LC_PER_UE'] = int(params['NUM_DL_LC_PER_UE'])
            config_data['NUM_UL_LCG_PER_UE'] = int(params['NUM_UL_LCG_PER_UE'])
            config_data['SEED'] = int(params['SEED'])

            # Write updated config
            with open(self.pfmsort_config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

            self.logger.info(f"Updated config: NUM_CELL={params['NUM_CELL']}, "
                             f"NUM_UE_PER_CELL={params['NUM_UE_PER_CELL']}, "
                             f"NUM_DL_LC_PER_UE={params['NUM_DL_LC_PER_UE']}, "
                             f"NUM_UL_LCG_PER_UE={params['NUM_UL_LCG_PER_UE']}, "
                             f"SEED={params['SEED']}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update config file: {e}")
            return False

    def run_pfmsort_test(self):
        """Run a single PFM sort test with current configuration."""
        test_log = (
            Path(self.config["log_folder"])
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_pfmsort_"
            f"{self.config['tv_index']}_test.log"
        )

        cmd = f"cd {self.config['cubb_sdk']} && {self.build_path}/cuMAC/examples/pfmSort/pfmSortTest"

        self.logger.info(f"Starting PFM sort test, log stored in {test_log}")
        self.logger.info(f"Running command: {cmd}")

        test_cmd = f"{cmd} > {test_log} 2>&1"
        result = self.run_subprocess(test_cmd, self.log_file)

        # Check return code and log content
        if result == 1:
            self.logger.error(
                f"PFM sort test FAILED with non-zero exit code, please check {test_log}"
            )
            fail_log = test_log.with_suffix(".FAIL")
            test_log.rename(fail_log)
            return False
        else:
            # Check log content for pass/fail message
            try:
                with open(test_log, 'r') as f:
                    log_content = f.read()

                if "PFM sorting test passed" in log_content:
                    self.logger.info("PFM sorting test PASSED")
                    pass_log = test_log.with_suffix(".PASS")
                    test_log.rename(pass_log)
                    return True
                elif "PFM sorting test failed" in log_content:
                    self.logger.error("PFM sorting test FAILED - check log for details")
                    fail_log = test_log.with_suffix(".FAIL")
                    test_log.rename(fail_log)
                    return False
                else:
                    self.logger.warning("Could not determine test result from log")
                    unknown_log = test_log.with_suffix(".UNKNOWN")
                    test_log.rename(unknown_log)
                    return False
            except Exception as e:
                self.logger.error(f"Error reading test log: {e}")
                return False

    def run_pfmsort_test_single(self):
        """Run a single PFM sort test case based on tv_index."""
        if not os.path.exists(self.csv_filename):
            self.pfmsort_test_parameters_comb()

        with open(self.csv_filename, "r") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            params = next(
                (
                    row
                    for row in csv_reader
                    if row["Index"] == self.config["tv_index"]
                ),
                None,
            )

        if params:
            self.logger.info(
                f"Running PFM sort test case {self.config['tv_index']}"
            )
            self.logger.info(f"Parameters: {params}")

            # Build cuMAC only if build folder doesn't exist
            if not self.build_path.exists():
                self.logger.info("Build folder not found, building cuMAC...")
                if not self.build_cumac():
                    self.logger.error(
                        f"Failed to build cuMAC, please check the {self.log_file}"
                    )
                    return
            else:
                self.logger.info("Build folder exists, skipping build step")

            # Backup config file if not already backed up
            if not self.pfmsort_config_file_bak.exists() and os.path.exists(self.pfmsort_config_file):
                self.run_subprocess(
                    f"cp -f {self.pfmsort_config_file} {self.pfmsort_config_file_bak}"
                )

            # Update config file
            if self.update_pfmsort_config(params):
                # Run test
                self.run_pfmsort_test()
            else:
                self.logger.error("Failed to update PFM sort config")

            # Restore original config
            if self.pfmsort_config_file_bak.exists():
                self.run_subprocess(
                    f"cp {self.pfmsort_config_file_bak} {self.pfmsort_config_file}"
                )
        else:
            self.logger.error(
                f"Test case index {self.config['tv_index']} not found in {self.csv_filename}"
            )

    def run_pfmsort_test_all(self):
        """Run all PFM sort test cases."""
        if not os.path.exists(self.csv_filename):
            self.pfmsort_test_parameters_comb()

        # Build cuMAC only once if build folder doesn't exist
        if not self.build_path.exists():
            self.logger.info("Build folder not found, building cuMAC...")
            if not self.build_cumac():
                self.logger.error(
                    f"Failed to build cuMAC, please check the {self.log_file}"
                )
                return
        else:
            self.logger.info("Build folder exists, skipping build step")

        # Backup config file
        if not self.pfmsort_config_file_bak.exists() and os.path.exists(self.pfmsort_config_file):
            self.run_subprocess(
                f"cp -f {self.pfmsort_config_file} {self.pfmsort_config_file_bak}"
            )

        with open(self.csv_filename, "r") as csvfile:
            parameters_list = list(csv.DictReader(csvfile))

        total_tests = len(parameters_list)
        passed_tests = 0
        failed_tests = 0

        for params in parameters_list:
            self.config["tv_index"] = params["Index"]
            self.logger.info(
                f"Processing PFM sort test {self.config['tv_index']} of {total_tests}"
            )

            # Update config file and run test
            if self.update_pfmsort_config(params):
                # Run test
                if self.run_pfmsort_test():
                    passed_tests += 1
                else:
                    failed_tests += 1
            else:
                self.logger.error(f"Failed to update config for test {self.config['tv_index']}")
                failed_tests += 1

        # Restore original config
        if self.pfmsort_config_file_bak.exists():
            self.run_subprocess(
                f"cp {self.pfmsort_config_file_bak} {self.pfmsort_config_file}"
            )

        self.logger.info(
            f"PFM sort test suite completed: {passed_tests} passed, {failed_tests} unsuccessful out of {total_tests} total"
        )

        self.check_and_rename_log_file(
            self.log_file, "PFM Sort Tests", "fail|Fail|ERROR|FAIL"
        )

    def cumac_test_cmd(self):
        numSimChnRlz = self.get_parameter_value("numSimChnRlz")
        numCellConst = self.get_parameter_value("numCellConst")
        numActiveUePerCellConst = self.get_parameter_value("numActiveUePerCellConst")
        raw_num_cell = numCellConst
        raw_num_active_ue_per_cell = numActiveUePerCellConst
        try:
            numCellConst = int(numCellConst.split()[0]) if numCellConst else 0
            numActiveUePerCellConst = int(numActiveUePerCellConst.split()[0]) if numActiveUePerCellConst else 0
            numActiveLinks = numCellConst * (numCellConst * numActiveUePerCellConst)
        except (ValueError, TypeError, IndexError):
            self.logger.exception(
                "Failed to parse numCellConst/numActiveUePerCellConst for numActiveLinks; "
                "numCellConst=%r, numActiveUePerCellConst=%r. Using numActiveLinks=999999 to disable sanitizer.",
                raw_num_cell, raw_num_active_ue_per_cell
            )
            numActiveLinks = 999999  # disable sanitizer on parse error
        
        # Disable compute-sanitizer for numActiveLinks > 25600 (too slow / high memory)
        compute_sanitizer_cmd = (
            "compute-sanitizer --error-exitcode 0 --tool memcheck --leak-check full "
            if numSimChnRlz.strip() == "10" and numActiveLinks <= 25600  # 25600 links = 16 Cells, 100 UEs per cell
            else ""
        )

        d = "0" if self.config["direction"] == "UL" else "1"
        if self.config["option"] == "f1":
            f = "1"
        elif self.config["option"] == "f2":
            f = "2"
        elif self.config["option"] == "f3":
            f = "3"
        elif self.config["option"] == "f4":
            f = "4"

        if self.config["option"] in ["f1", "f2", "f3", "f4"]:
            test_log = (
                Path(self.config["log_folder"])
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_{self.config['antenna']}T{self.config['antenna']}R_{self.config['direction']}_{self.config['tv_index']}_{self.config['option']}_test.log"
            )
            cmd = f"{self.build_path}/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
            test_cmd = [
                "timeout -s 9 600",
                compute_sanitizer_cmd,
                str(cmd),
                "-d",
                d,
                "-b",
                "0",
                "-f",
                f,
                ">",
                str(test_log),
            ]
        elif self.config["option"] == "64tr":
            cmd = f"{self.build_path}/cuMAC/examples/multiCellMuMimoScheduler/multiCellMuMimoScheduler"
            test_log = (
                Path(self.config["log_folder"])
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_{self.config['antenna']}T{self.config['antenna']}R_{self.config['direction']}_{self.config['tv_index']}_test.log"
            )
            test_cmd = [
                str(cmd),
                "-d",
                d,
                "-a",
                "1",
                "-r",
                "1",
                "-i",
                self.config["tv_folder"],
                ">",
                str(test_log),
            ]
        elif self.config["option"] == "drl":
            cmd = f"{self.build_path}/cuMAC/examples/ml/drlMcsSelection/drlMcsSelection"
            test_log = (
                Path(self.config["log_folder"])
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_{self.config['option']}_test.log"
            )
            test_cmd = [
                str(cmd),
                "-i",
                self.config["tv_folder"],
                "-m",
                f"{self.config['cubb_sdk']}/cuMAC/examples/ml/trainedModels/model.onnx",
                "-g",
                self.config["gpu_id"],
                ">",
                str(test_log),
            ]
        elif self.config["option"] == "srs":
            cmd = f"{self.build_path}/cuMAC/examples/multiCellSrsScheduler/multiCellSrsScheduler"
            test_log = (
                Path(self.config["log_folder"])
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cuMAC_{self.config['option']}_{self.config['tv_index']}_test.log"
            )
            test_cmd = [
                f"cd {self.build_path} && {compute_sanitizer_cmd}",
                str(cmd),
                f"{self.cumac_folder}/examples/multiCellSrsScheduler/srs_scheduler_testing_config.yaml",
                ">",
                str(test_log),
            ]
        self.logger.info(f"Starting tests, log stored in {test_log}")
        result = self.run_subprocess(" ".join(test_cmd), self.log_file)

        if result == 1:
            self.logger.error(
                f"TEST FAIL,the exit code of Command:{test_cmd} is {result}, not 0, please check {test_log}"
            )
            fail_log = test_log.with_suffix(".FAIL")
            test_log.rename(fail_log)
        else:
            if self.config["option"] == "64tr":
                with open(test_log, 'r') as f:
                    log_content = f.read()
                    Checkpoint = f"Summary - cuMAC multi-cell MU-MIMO scheduler solution check: PASS"
                    if Checkpoint in log_content:
                        self.logger.info(Checkpoint)
                    else:
                        self.logger.warning(
                            f"TEST FAIL: Missing expected message in log: {Checkpoint}")
                        log = test_log.with_suffix(".FAIL")
                        test_log.rename(log)
            elif self.config["option"] in ["drl", "srs"]:
                log = test_log.with_suffix(".PASS")
                self.logger.info(f"TEST PASS")
                test_log.rename(log)
            elif self.config["option"] in ["f1", "f2", "f3", "f4"]:
                log = test_log.with_suffix(".PASS")
                self.logger.info(f"Execution PASS")
                self.check_and_rename_log_file(
                    test_log, f"tdl {self.config['option']} test log", "returned an error")


    # apply bigger cpu/gpu perf gap patch to examples/parameters.h when option is f1..f4 and allow_bigger_cpu_gpu_gap is true
    def _maybe_apply_bigger_gap_patch(self):
        """
        If --allow-bigger-cpu-gpu-gap is enabled AND option is f1..f4,
        change cpuGpuPerfGapPerUeConst and cpuGpuPerfGapSumRConst to 0.05 in file examples/parameters.h
        in lines of the form:  #define NAME VALUE 
        """
        if self.config.get("option") not in ["f1", "f2", "f3", "f4"]:
            return
        if not self.config.get("allow_bigger_cpu_gpu_gap"):
            return
        self.logger.info("Applying bigger CPU/GPU perf-gap patch in examples/parameters.h")
        try:
            text = self.param_file.read_text()
            original_text = text
            for name, new_val in self._cpu_gpu_perf_gap_targets.items():
                define_re = re.compile(
                    rf'(^\s*#\s*define\s+{name}\s+)([0-9]*\.?[0-9]+)\b',
                    re.IGNORECASE | re.MULTILINE
                )
                text, n = define_re.subn(rf'\g<1>{new_val}', text)
                self.logger.info(f"{name}: updated {n} #define occurrence(s)")
            if text != original_text:
                # Ensure backup before first modification
                if not self.param_file_bak.exists():
                    self.run_subprocess(f"cp -f {self.param_file} {self.param_file_bak}")
                self.param_file.write_text(text)
                self.run_subprocess(self.check_command, self.log_file)
            else:
                self.logger.warning("Bigger-gap patch made no changes (no matching #define lines found).")
        except Exception as e:
            self.logger.error(f"Error applying bigger CPU/GPU perf-gap patch in examples/parameters.h: {e}")


    def run_tdl_test_all(self):
        if not self.param_file_bak.exists():
            self.run_subprocess(
                f"cp -f {self.param_file} {self.param_file_bak}")

        if not self.csv_filename.exists():
            self.tdl_test_parameters_comb()

        with open(self.csv_filename, "r") as csvfile:
            parameters_list = list(csv.DictReader(csvfile))

        for index, params in enumerate(parameters_list, start=1):
            if params["gpuAllocTypeConst"] != params["cpuAllocTypeConst"]:
                continue

            self.config["tv_index"] = f"{index:04}"
            self.logger.info(
                f"Processing cuMAC_TDL_Test_{self.config['tv_index']}")

            param_combin = self.update_parameters(params)
            self._maybe_apply_bigger_gap_patch()
            if self.build_cumac():
                self.config["tv_index"] = param_combin
                self.cumac_test_cmd()
                self.backup_parameters()

        self.restore_original_parameters()
        self.check_and_rename_log_file(self.log_file, "TDL Tests", "Fail|Err")

    def run_cumac_test(self):
        if self.config["option"] != "drl" and self.config["option"] != "srs":
            if self.config["option"] in ["f1", "f2", "f3", "f4"]:
                test_param = self.tdl_test_parameters_comb()
            elif self.config["option"] == "64tr":
                test_param = self.mimo_test_parameters_comb()

            if not self.param_file_bak.exists():
                self.run_subprocess(
                    f"cp -f {self.param_file} {self.param_file_bak}")

            if not self.csv_filename.exists():
                test_param

            with open(self.csv_filename, "r") as csvfile:
                csv_reader = csv.DictReader(csvfile)
                params = next(
                    (
                        row
                        for row in csv_reader
                        if row["Index"] == self.config["tv_index"]
                    ),
                    None,
                )

            if params:
                self.logger.info(
                    f"Updating {self.config['tv_index']} parameters ......"
                )
                self.logger.info("Parameters have been updated.")
                param_combin = self.update_parameters(params)
                print(f"Test case {self.config['tv_index']} : {param_combin}")

                self._maybe_apply_bigger_gap_patch()
                if self.build_cumac():
                    if self.config["option"] in ["f1", "f2"]:
                        self.config["tv_index"] = f"{param_combin}_TDL"
                    elif self.config["option"] in ["f3", "f4"]:
                        self.config["tv_index"] = f"{param_combin}_CDL"
                    else:
                        self.config["tv_index"] = (
                            f"{param_combin}_{self.config['option']}"
                        )
                    self.cumac_test_cmd()
                    self.backup_parameters()
                else:
                    self.logger.error(
                        f"Failed to build cuMAC, please check the {self.log_file}"
                    )
            else:
                self.logger.error(
                    f"TV index {self.config['tv_index']} not found in {self.csv_filename}"
                )

            self.restore_original_parameters()
        else:
            cmd = f"python3 update_parameter.py -f {self.param_file} -p gpuDeviceIdx -v {self.config['gpu_id']}"
            self.run_subprocess(cmd, self.log_file)
            if self.build_cumac():
                self.cumac_test_cmd()
            else:
                self.logger.error(
                    f"Failed to build cuMAC, please check the {self.log_file}"
                )

    def single_tti_test_perTV(self):
        if self.config["direction"] == "DL":
            dir_command = "-d 1"
        elif self.config["direction"] == "UL":
            dir_command = "-d 0"
        else:
            self.logger.error(
                "Invalid direction specified, only supports 'DL' or 'UL'."
            )
            return

        if self.config["test"] == "UE_selection":
            test_command = "-m 1000"
        elif self.config["test"] == "PRG_allocation":
            test_command = "-m 0100"
        elif self.config["test"] == "Layer_selection":
            test_command = "-m 0010"
        elif self.config["test"] == "MCS_selection":
            test_command = "-m 0001"
        elif self.config["test"] == "Scheduler_pipeline":
            test_command = "-m 1111"
        else:
            self.logger.error(
                "Invalid test specified, only supports 'UE_selection','PRG_allocation','Layer_selection','MCS_selection','Scheduler_pipeline'."
            )
            return

        # For 4t4r_fast_fading, run test for each fading type
        if self.config["option"] in ["4t4r_tdl", "4t4r_cdl"]:
            if self.config["option"] == "4t4r_tdl":
                fading_types = ["f1", "f2"]
            elif self.config["option"] == "4t4r_cdl":
                fading_types = ["f3", "f4"]
            original_tv_index = self.config["tv_index"]
            for fading in fading_types:
                self.config["tv_index"] = f"{original_tv_index}_{fading}"
                self.tv_name = f"TV_cuMAC_{self.config['antenna']}T{self.config['antenna']}R_{self.config['direction']}_TC{self.config['tv_index']}"
                self.tv_file = os.path.join(
                    f"{self.config['tv_folder']}",
                    f"{self.tv_name}.h5",
                )

                # Check if we should skip this TV based on conditions
                print(f"self.tv_name: {self.tv_name}")
                if self.should_skip_tv():
                    continue

                self._run_single_tti_test(dir_command, test_command)
        else:
            # For non-fast-fading tests, check skip conditions before running
            if not self.should_skip_tv():
                self._run_single_tti_test(dir_command, test_command)
            else:
                self.logger.info("Skipping TV based on conditions")

    def _run_single_tti_test(self, dir_command, test_command):
        """Helper method to run a single TTI test with given parameters"""
        self.logger.info(
            f"Start to run Single-TTI test for {self.tv_file}......"
        )
        self.logger.info(f"Test: {self.config['test']}")
        self.logger.info(f"Test command: {test_command}")
        self.logger.info(f"Direction: {self.config['direction']}")
        self.logger.info(f"TV file: {self.tv_file}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(
            f"Check if the {self.tv_file} exists, if no, then generate it ......"
        )
        if not os.path.exists(self.tv_file):
            self.logger.info(
                f"Not found the {self.tv_file}, start to generate {self.tv_file} ......"
            )
            print(
                f"run_single_tti_test: self.config['tv_index']: {self.config['tv_index']}")
            self.generate_tv_file()
            if not os.path.exists(self.tv_file):
                self.logger.error(
                    f"FAILED to generate {self.tv_file}, please check the log: {self.log_file}."
                )
                return
        self.logger.info(f"Found {self.tv_file}, start to run testing ......")
        run_test = f"cd {self.build_path}/cuMAC/examples/tvLoadingTest/ && ./tvLoadingTest -i {self.tv_file} -g 2 {dir_command} {test_command}"
        self.run_subprocess(run_test, self.log_file)

    def single_tti_test_all(self):
        for filename in os.listdir(f"{self.config['tv_folder']}"):
            self.logger.info(
                f"Start to run single tti test for all TV under {self.config['tv_folder']}"
            )
            if filename.endswith(".h5") and "NoPrecoding_MMSE-IRC_UL" not in filename:
                self.logger.info("TV file: %s", filename)
                # self.tv_name = filename
                self.tv_file = os.path.join(
                    f"{self.config['tv_folder']}/", f"{filename}"
                )
                self.logger.info("self tv_file: %s", self.tv_file)
                self.single_tti_test_perTV()

    def continuous_time_test_perTV(self):

        self.logger.info(
            f"Start to run Continuous-Time test for {self.tv_file}......")
        self.logger.info(f"Direction: {self.config['direction']}")
        self.logger.info(f"TV file: {self.tv_file}")
        self.logger.info(f"Log file: {self.log_file}")

        tv_log = self.find_tv_logs()
        self.logger.info(f"TV log: {tv_log}")
        if not tv_log:
            self.logger.info(
                "TV log not found in %s. Generating %s...",
                self.config["log_folder"],
                self.tv_file,
            )
            self.logger.info(
                f"Clean up the tv file before re-gernerate it ....")
            rm_tv = f"rm -rf {self.tv_file}"
            self.run_subprocess(rm_tv, None)
            self.generate_tv_file()
            tv_log = self.find_tv_logs()
        self.logger.info("Found %s. Starting log analysis...", tv_log)
        self.analyze_tv_logs(tv_log)

    def generate_tv_file(self):
        """Generate TV file using current configuration."""
        if os.path.exists(self.tv_file):
            os.remove(self.tv_file)
        # Create a new TVGenerator instance with current tv_index
        tv_generator = self._create_tv_generator()
        tv_generator.generate()

    def find_tv_logs(self):
        return glob.glob(
            os.path.join(self.config["log_folder"],
                         f"generate_{self.tv_name}.log*")
        )

    def analyze_tv_logs(self, tv_log):
        for log in tv_log:
            self.check_and_rename_log_file(
                log, "Continuous-Time Test", "FAIL|ERR")

    def update_parameters(self, params):
        param_combin = ""
        for param, value in params.items():
            if param != "Index" and value != self.config["tv_index"]:
                command = f"python3 update_parameter.py -f {self.param_file} -p {param} -v {value}"
                self.run_subprocess(command, self.log_file)
                if param == "numCellConst":
                    param_combin += f"{value}C"
                elif param == "numActiveUePerCellConst":
                    param_combin += f"_{value}UEPerCell"
                elif param == "numSimChnRlz":
                    param_combin += f"_{value}TTI"
                elif param == "gpuAllocTypeConst":
                    param_combin += f"_gpuAllocType{value}"
                elif param == "cpuAllocTypeConst":
                    param_combin += f"_cpuAllocType{value}"
                self.run_subprocess(command, self.log_file)
        self.run_subprocess(self.check_command, self.log_file)
        return param_combin

    def get_parameter_value(self, param: str) -> str:
        """Get the value of a preprocessor macro from the parameters header.

        Runs grep/awk via run_subprocess to find the #define line for the given
        macro in self.param_file; command output is appended to self.log_file.
        If the subprocess fails or returns non-string (e.g. 1), returns "".

        Args:
            param: Macro name to look up (e.g. "numCellConst"). Lookup is done
                in self.param_file.

        Returns:
            The stripped macro value (third token of the matching #define line),
            or "" if no match, run_subprocess fails (returns 1), or result is not
            a string.

        Raises:
            None. run_subprocess catches subprocess.CalledProcessError and
            other Exception and returns 1 instead of raising.

        Examples:
            >>> self.get_parameter_value("numCellConst")
            '16'
            >>> self.get_parameter_value("nonexistent")
            ''
        """
        # Match only the line that defines this macro (#define param value), not lines that use it
        result = self.run_subprocess(
            f"grep -E '^\\s*#\\s*define\\s+{param}\\s+' {self.param_file} | awk '{{print $3}}'",
            self.log_file,
        )
        return result.strip() if isinstance(result, str) else ""

    def _is_ul_supported_index(self):
        """Check if the current index supports UL direction."""
        # Extract the base index number from tv_index (e.g., '0032' from '0032_f1')
        base_index = self.config["tv_index"].split('_')[0]
        return base_index in self.UL_SUPPORTED_INDICES

    def _is_500_ue_index(self):
        """Check if the current index has 500 UEs configuration."""
        ue_index = self.config["tv_index"].split('_')[0]
        return ue_index in self.UE_500_INDICES

    def _is_unsupported_fading_type(self):
        """Check if the current fading type is unsupported for 500 UEs."""
        print(f"self.tv_name: {self.tv_name}")
        return "f2" in self.tv_name or "f4" in self.tv_name

    def should_skip_tv(self):
        """Determine if the current TV should be skipped based on combined conditions."""
        if self._is_500_ue_index():
            # For 500 UE cases, only skip f2 and f4 fading types
            if self._is_unsupported_fading_type():
                self.logger.info(
                    f"Skip this test: 4T4R fast fading test does not support 500 UEs per cell for {self.tv_name}")
                return True
        elif self.config["direction"].strip() == "UL":
            # For non-500 UE cases with UL direction
            print(f"self.config['tv_index']: {self.config['tv_index']}")
            if not self._is_ul_supported_index():
                self.logger.info(
                    f"Skip this test: UL does not support GPU allocation type 0 and no precoding (prdSchemeConst=0) for {self.tv_name}")
                return True

        return False

    def build_cumac(self):
        self.logger.info(f"Start to build cuMAC ......")
        build_result = self.run_subprocess(self.build_command, self.log_file)
        if build_result == 0:
            self.logger.info(
                f"Built cuMAC successfully, ready to generate TV_cuMAC_{self.config['antenna']}T{self.config['antenna']}R_{self.config['direction']}_TC{self.config['tv_index']}."
            )
            return True
        # return False
        return True

    def backup_parameters(self):
        new_filename = (
            f"{self.cumac_folder}/examples/parameters_{self.config['tv_index']}.h"
        )
        if not os.path.exists(new_filename):
            self.run_subprocess(
                f"cp -f {self.param_file} {new_filename}", self.log_file
            )
            self.run_subprocess(
                f"mv {new_filename} {self.config['log_folder']}", self.log_file
            )
        self.run_subprocess(
            f"cp {self.param_file_bak} {self.param_file}", self.log_file
        )

    def restore_original_parameters(self):
        self.run_subprocess(
            f"cp {self.param_file_bak} {self.param_file}", self.log_file
        )

    def check_csv_does_not_contain(self, file_path, content):
        with open(file_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                for cell in row:
                    if content in str(cell):
                        return False
        return True

    def check_and_rename_log_file(self, log_file, test_type, key_word):
        log_folder = os.path.dirname(log_file)
        log_file_str = str(log_file)
        if not os.path.isfile(log_file):
            self.logger.error("Log file %s does not exist.", log_file)
            return

        # Check if the file name already ends with .FAIL or .PASS
        if log_file_str.endswith(".FAIL") or log_file_str.endswith(".PASS"):
            self.logger.info(
                "Log file %s already processed. Skipping rename.", log_file_str
            )
            return

        try:
            with open(log_file, "r") as file:
                log_content = file.read()
                self.logger.info("Checking the log file %s ......", log_file)

                # Case-insensitive search for failure patterns
                failure_patterns = re.compile(r"{}".format(key_word))
                if failure_patterns.search(log_content):
                    new_status = "FAIL"
                    self.logger.error(
                        "%s FAILED, %s contains %s message.",
                        test_type,
                        log_file,
                        key_word,
                    )
                else:
                    new_status = "PASS"
                    self.logger.info("%s PASS.", test_type)

                new_log_file = f"{log_file}.{new_status}"
                # new_log_file_path = os.path.join(log_folder, new_log_file)

                # Rename the file
                os.rename(log_file, new_log_file)
                # os.rename(log_file, new_log_file_path)
                self.logger.info("Renamed %s to %s", log_file, new_log_file)

        except Exception as e:
            self.logger.error(
                "An error occurred while processing %s: %s", log_file, str(e)
            )

    def create_logger(
        logger_name,
        log_file,
        log_file_max_size=10,
        log_file_max_backups=5,
        log_level=logging.INFO,
    ):
        # Create log directory if it doesn't exist
        log_folder = os.path.dirname(log_file)
        os.makedirs(log_folder, exist_ok=True)

        # Create logger object
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        # Create file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_file_max_size * 1024 * 1024,
            backupCount=log_file_max_backups,
        )
        file_handler.setLevel(log_level)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Set log format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def test_main(self):
        """Main function to run testing."""
        Path(self.config["log_folder"]).mkdir(parents=True, exist_ok=True)
        if self.config["option"] != "64tr":
            Path(self.config["tv_folder"]).mkdir(parents=True, exist_ok=True)
        if self.config["option"] in ["4t4r", "4t4r_tdl", "4t4r_cdl"]:
            self.run_subprocess(
                f"mkdir -p {self.config['tv_folder']}/{self.config['antenna']}T{self.config['antenna']}R"
            )
            self.config["tv_folder"] = (
                Path(self.config["tv_folder"])
                / f"{self.config['antenna']}T{self.config['antenna']}R"
            )

        self.logger.info("Start to generate cuMAC test combinations.")
        self.logger.info(f"tv folder: {self.config['tv_folder']}")
        self.logger.info(f"log folder: {self.config['log_folder']}")
        self.logger.info(f"log file: {self.log_file}")
        self.logger.info(f"antenna:{self.config['antenna']}")
        self.logger.info(f"diretion: {self.config['direction']}")
        self.logger.info(f"option: {self.config['option']}")
        self.logger.info(f"test:{self.config['test']}")
        self.logger.info(f"cmake:{self.config['cmake']}")
        self.logger.info(f"build command:{self.build_command}")

        if self.config["option"] in ["4t4r", "4t4r_tdl", "4t4r_cdl"]:
            if self.config["test"] in [
                "UE_selection",
                "PRG_allocation",
                "Layer_selection",
                "MCS_selection",
                "Scheduler_pipeline",
            ]:
                if self.config["tv_index"] == "all":
                    self.single_tti_test_all()
                else:
                    self.single_tti_test_perTV()
            elif self.config["test"] in ["ct"]:
                self.continuous_time_test_perTV()

            self.logger.info(
                f"Complete to run cuMAC {self.config['option']} {self.config['test']} tests on TC {self.config['tv_index']} of {self.config['direction']} ."
            )
            self.check_and_rename_log_file(
                self.log_file,
                f"{self.config['option']} {self.config['test']} TEST",
                "fail|Fail|ERROR|FAIL",
            )
        elif self.config["option"] in ["f1", "f2", "f3", "f4", "64tr", "drl", "srs"]:
            if self.config["tv_index"] == "all" and self.config["option"] in [
                "f1",
                "f2",
            ]:
                self.run_tdl_test_all()
            else:
                self.run_cumac_test()
            self.logger.info(
                f"Complete to run cuMAC {self.config['option']} tests on {self.config['antenna']}T{self.config['antenna']}R {self.config['direction']} {self.config['tv_index']}."
            )
            self.check_and_rename_log_file(
                self.log_file, f"{self.config['option']} TEST", "fail|Fail|ERROR|FAIL"
            )
        elif self.config["option"] == "pfmsort":
            if self.config["tv_index"] == "all":
                self.run_pfmsort_test_all()
            else:
                self.run_pfmsort_test_single()
            self.logger.info(
                f"Complete to run cuMAC PFM sort tests {self.config['tv_index']}."
            )
            self.check_and_rename_log_file(
                self.log_file, "PFM Sort TEST", "fail|Fail|ERROR|FAIL"
            )

    def main():
        parser = argparse.ArgumentParser(
            description="Generate TV with updated parameters"
        )
        parser.add_argument(
            "-d", "--direction", choices=["DL", "UL"], help="Direction (DL or UL)"
        )
        parser.add_argument(
            "-a",
            "--antenna",
            choices=["4", "64"],
            help="Antenna configuration (4 or 64)",
        )
        parser.add_argument("-c", "--cmake", help="CMake command")
        parser.add_argument("-l", "--log_folder", help="Log folder path")
        parser.add_argument("-tv", "--tv_folder", help="TV folder path")
        parser.add_argument(
            "-t",
            "--test",
            choices=[
                "UE_selection",
                "PRG_allocation",
                "Layer_selection",
                "MCS_selection",
                "Scheduler_pipeline",
                "ct",
            ],
            help="Options of 4T4R tests",
        )
        parser.add_argument(
            "-o",
            "--option",
            choices=["4t4r", "4t4r_tdl", "4t4r_cdl", "f1",
                     "f2", "f3", "f4", "64tr", "drl", "srs", "pfmsort"],
            help="Options for cuMAC Tests, 4t4r - 4T4R tests, 4t4r_tdl - 4T4R tdl tests, 4t4r_cdl - 4T4R cdl tests, f1/f2 - TDL tests, f3/f4 - CDL tests, 64tr - mMIMO tests, drl - DRL tests, srs - SRS tests, pfmsort - PFM sorting tests",
        )
        parser.add_argument(
            "-i",
            "--tv_index",
            help="TV Index in cuMAC/scripts/cuMAC_*.csv; default: none",
        )
        parser.add_argument(
            "--allow-bigger-cpu-gpu-gap",
            dest="allow_bigger_cpu_gpu_gap",
            action="store_true",
            help="If present (i.e., --allow-bigger-cpu-gpu-gap), tdl and cdl tests will relax allow bigger cpu/gpu perf gap, cpuGpuPerfGapPerUeConst = 0.05, cpuGpuPerfGapSumRConst = 0.05."
        )
        parser.add_argument(
            "--smoke",
            dest="smoke",
            action="store_true",
            help="If present, run smoke tests with reduced parameter combinations (e.g., fewer seeds for pfmsort tests)."
        )
        parser.add_argument("-g", "--gpu_id", help="GPU device ID")
        parser.add_argument("-s", "--cubb_sdk", help="cuBB SDK folder")

        args = parser.parse_args()   
        config = {
            "direction": args.direction if args.direction is not None else "DL",
            "antenna": args.antenna if args.antenna is not None else "4",
            "cubb_sdk": (
                args.cubb_sdk if args.cubb_sdk is not None else "/opt/nvidia/cuBB"
            ),           
            "cmake": (
                args.cmake
                if args.cmake is not None
                #else "cmake -Bbuild -GNinja && cmake --build build"
                else f"cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE={args.cubb_sdk}/cuPHY/cmake/toolchains/grace-cross -DNVIPC_FMTLOG_ENABLE=ON && cmake --build build --target cumac_examples"
            ),
            "log_folder": (
                args.log_folder if args.log_folder is not None else os.getcwd()
            ),
            "tv_folder": (
                args.tv_folder
                if args.tv_folder is not None
                else f"{args.cubb_sdk}/testVectors/cumac"
            ),
            "test": args.test if args.test is not None else "",
            "option": args.option if args.option is not None else "4t4r",
            "tv_index": args.tv_index,
            "gpu_id": args.gpu_id if args.gpu_id is not None else "0",
            "allow_bigger_cpu_gpu_gap": args.allow_bigger_cpu_gpu_gap,
            "smoke": args.smoke,
        }
        test = cuMACTest(config)
        test.test_main()

    def _create_tv_generator(self, tv_index=None):
        """Create a new TV Generator instance with current configuration parameters."""
        return TVGenerator(
            self.config["direction"],
            self.config["antenna"],
            None,
            self.config["log_folder"],
            f"{self.config['tv_folder']}",
            False,
            "4" if self.config["option"] in ["4t4r_cdl", "4t4r_tdl"] else "1",
            tv_index or self.config["tv_index"],
            self.config["gpu_id"],
            f"{self.config['cubb_sdk']}",
        )


if __name__ == "__main__":
    cuMACTest.main()
    """
    Examples:
        Note: Parameters in {} are optional

        1. Run cuMAC 4T4R Single-TTI tests:
           python3 run_cumac_test.py -o 4t4r|4t4r_tdl|4t4r_cdl -d DL|UL -a 4 -t TEST_TYPE -i INDEX \
                                    {-l LOG_DIR -tv TV_DIR}
           where:
           TEST_TYPE: UE_selection, PRG_allocation, Layer_selection, MCS_selection,
                     Scheduler_pipeline, ct
           INDEX: 0001 ... 0032, all

        2. Run cuMAC TDL tests:
           python3 run_cumac_test.py -o f1|f2 -d DL|UL -a 4 {-l LOG_DIR -tv TV_DIR}
           Some tests can fail because of the cpu/gpu perf gap, if allow bigger cpu/gpu perf gap, add --allow-bigger-cpu-gpu-gap

        3. Run cuMAC CDL tests:
           python3 run_cumac_test.py -o f3|f4 -d DL|UL -a 4 {-l LOG_DIR -tv TV_DIR}
           Some tests can fail because of the cpu/gpu perf gap, if allow bigger cpu/gpu perf gap, add --allow-bigger-cpu-gpu-gap 

        4. Run cuMAC DRL tests:
           python3 run_cumac_test.py -o drl -d DL|UL {-a 4 -l LOG_DIR -tv TV_DIR}

        5. Run cuMAC mMIMO tests, note that this test is replaced by cumac_64tr_test.py after Rel 25-2
           python3 run_cumac_test.py -o 64tr -d DL|UL -a 64 {-l LOG_DIR -tv TV_DIR}

        6. Run cuMAC SRS tests:
           python3 run_cumac_test.py -o srs {-d DL|UL -a 4 -l LOG_DIR -tv TV_DIR}

        Default values if not specified:
        LOG_DIR = current directory
        TV_DIR = /opt/nvidia/cuBB/testVectors/cumac
    """
