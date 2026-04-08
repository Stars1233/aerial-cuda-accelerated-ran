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
import itertools
from datetime import datetime
import argparse
import csv
import glob
import re
import logging
from logging.handlers import RotatingFileHandler


class TVGenerator:
    def __init__(
        self,
        direction,
        antenna,
        cmake,
        logfolder,
        tvfolder,
        name,
        option,
        tvIndex,
        gpuid,
        cubb,
    ):
        self.direction = direction if direction is not None else "DL"
        self.antenna = antenna if antenna is not None else "4"
        self.cubb = cubb if cubb is not None else "/opt/nvidia/cuBB"
        self.cmake = (
            cmake
            if cmake is not None
            else f"cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE={self.cubb}/cuPHY/cmake/toolchains/grace-cross && cmake --build build --target cumac_examples"
        )
        self.logfolder = logfolder if logfolder is not None else os.getcwd()
        self.tvfolder = (
            tvfolder if tvfolder is not None else f"{self.cubb}/testVectors/cumac"
        )
        self.name = name
        self.option = option if option is not None else "1"
        self.tvIndex = tvIndex
        self.gpuid = gpuid if gpuid is not None else "0"
        # if cmake is None:
        #    self.build_path = f"{self.cubb}/cuMAC"
        # else:
        #    self.build_path = f"{self.cubb}"
        self.buildCommand = f"cd {self.cubb} && {self.cmake}"
        if self.option == "3":
            self.tvIndex = "testMAC"
            self.log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_generate_cuMAC_{self.tvIndex}TV_main.log"
        elif self.option == "2":
            self.tvIndex = "CPU"
            self.log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_generate_cuMAC_{self.tvIndex}TV_main.log"
        else:
            self.log_filename = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_generate_cuMAC_TV_main.log"
            )
        self.log_file = os.path.join(self.logfolder, self.log_filename)
        self.csv_filename = f"{self.cubb}/cuMAC/scripts/cumac_tv_parameters.csv"
        self.param_file = f"{self.cubb}/cuMAC/examples/parameters.h"
        self.param_file_bak = f"{self.cubb}/cuMAC/examples/parameters.h_bak"
        self.checkcommand = f"grep 'nBsAntConst\|nUeAntConst\|numSimChnRlz\|gpuDeviceIdx\|seedConst\|numCellConst\|numUePerCellConst\|numActiveUePerCellConst\|nPrbsPerGrpConst\|numUePerCellConst\|gpuAllocTypeConst\|cpuAllocTypeConst\|prdSchemeConst\|rxSchemeConst' {self.param_file}"
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
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
            # print(f"command: {command}")
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    command,
                    output="".join(stdout),
                    stderr="".join(stderr),
                )
            if log_file:
                with open(log_file, "a") as f:
                    f.write(f"{command} return code: {process.returncode}\n")

            return "".join(stdout)
        except subprocess.CalledProcessError as e:
            if (
                "multiCellSchedulerUeSelection" in command or "tvLoadingTest" in command
            ) and process.returncode == 1:
                return 0
            else:
                error_message = f"Error executing command: {command}\n {e.stderr}\n Return code: {e.returncode}\n"
                print(error_message)
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(error_message)
                return 1
        except Exception as e:
            error_message = f"Unexpected error: {e}\n"
            print(error_message)
            if log_file:
                with open(log_file, "a") as f:
                    f.write(error_message)
            return 1

    def redefine_tv_name(self):
        main_file = f"{self.cubb}/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp"
        old_tv_name = 'std::string saveTvName = "TV_cumac_F08-MC-CC-" + std::to_string(net->cellGrpPrmsGpu->nCell) +"PC_" + (DL ? "DL" : "UL") + ".h5";'
        new_tv_name = 'std::string saveTvName = "TV_cumac_" + std::to_string(net->cellGrpPrmsGpu->nCell) +"C_" + std::to_string(numUePerCellConst) +"UEPerTTI_" + std::to_string(numActiveUePerCellConst) +"UEPerCell_" + std::to_string(nPrbGrpsConst) +"PRG_" +"AllocType"+ std::to_string(gpuAllocTypeConst)+ (prdSchemeConst ? "_SVDPrecoder_" : "_NoPrecoder_") +"MMSE-IRC_" + (DL ? "DL" : "UL") + ".h5";'
        backup_file = f"{main_file}_default"
        if not os.path.exists(backup_file):
            backup = f"cp -f {main_file} {backup_file}"
            subprocess.run(backup, shell=True)
        with fileinput.FileInput(main_file, inplace=True) as file:
            for line in file:
                if old_tv_name in line:
                    line = "    " + new_tv_name + "\n"
                print(line, end="")
        self.logger.info(f"Updated {main_file} with new TV name's definition.")

    def define_parameters_and_combinations(self):
        """Define parameters and generate all combinations."""
        if self.antenna == "4":
            nBsAntConst = [4]
            nUeAntConst = [4]
        elif self.antenna == "64":
            nBsAntConst = [64]
            nUeAntConst = [64]

        # numSimChnRlz = [5000]  # for QA release testing
        # for ENG CI/CD testing reduce 100
        numSimChnRlz = [100]
        gpuDeviceIdx = [self.gpuid]
        seedConst = [0]
        numCellConst = [8, 20]
        numUePerCellConst = [6, 16]
        numActiveUePerCellConst = [100, 500]
        nPrbsPerGrpConst = [4]
        nPrbGrpsConst = [272 // n for n in nPrbsPerGrpConst]
        gpuAllocTypeConst = [0, 1]
        cpuAllocTypeConst = [0, 1]
        prdSchemeConst = [0, 1]
        rxSchemeConst = [1]
        cpuGpuPerfGapPerUeConst = [0.01]
        cpuGpuPerfGapSumRConst = [0.03]

        all_combinations = list(
            itertools.product(
                nBsAntConst,
                nUeAntConst,
                numSimChnRlz,
                gpuDeviceIdx,
                seedConst,
                numCellConst,
                numUePerCellConst,
                numActiveUePerCellConst,
                nPrbsPerGrpConst,
                nPrbGrpsConst,
                gpuAllocTypeConst,
                cpuAllocTypeConst,
                prdSchemeConst,
                rxSchemeConst,
                cpuGpuPerfGapPerUeConst,
                cpuGpuPerfGapSumRConst,
            )
        )
        # csv_filename = "cumac_tv_parameters.csv"
        with open(self.csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Index",
                    "nBsAntConst",
                    "nUeAntConst",
                    "numSimChnRlz",
                    "gpuDeviceIdx",
                    "seedConst",
                    "numCellConst",
                    "numUePerCellConst",
                    "numActiveUePerCellConst",
                    "nPrbsPerGrpConst",
                    "nPrbGrpsConst",
                    "gpuAllocTypeConst",
                    "cpuAllocTypeConst",
                    "prdSchemeConst",
                    "rxSchemeConst",
                    "cpuGpuPerfGapPerUeConst",
                    "cpuGpuPerfGapSumRConst",
                ]
            )
            i = 1
            for combo in all_combinations:
                if combo[10] != combo[11]:
                    continue
                padded_index = f"{i:04}"
                csv_writer.writerow([padded_index] + list(combo))
                i += 1
        self.logger.info(
            f"The parameters and values mapping table per TV has been written to {self.csv_filename}."
        )
        return all_combinations

    def generate_tv(self):
        """Generate TV and copy the result to the specified folder."""
        # Disable the build path check after 25-3 release,all build path is in /opt/nvidia/cuBB
        # if "cuMAC" in self.cubb:
        #    cmd = f"timeout -s 9 7200 {self.cubb}/build/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
        # else:
        cmd = f"timeout -s 9 3600 {self.cubb}/build/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
        param_file = f"{self.cubb}/cuMAC/examples/parameters.h"

        if self.option in ["1", "4"]:
            tvfolder = f"{self.tvfolder}/{self.direction}"
            print(
                f"Clean the {tvfolder} in case there is existing h5 file which will cause moving tv files failed."
            )
            # Clean the tvfolder in case there is existing h5 file which will cause moving tv files failed.
            self.run_subprocess(f"rm -rf {tvfolder}/*", self.log_file)
        else:
            tvfolder = self.tvfolder

        # Base command construction
        if self.direction == "UL":
            base_cmd = ["cd", tvfolder, "&&", cmd, "-d", "0"]
        else:
            base_cmd = ["cd", tvfolder, "&&", cmd]

        base_tv_name = f"TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_TC"
        UePerCell = self.run_subprocess(
            f"grep '#define numActiveUePerCellConst' {self.param_file} | awk '{{print $3}}'",
            self.log_file,
        )
        # Handle different options
        if self.option == "4" and "_f" not in self.tvIndex:
            print(f"generate_tv.py: self.tvIndex: {self.tvIndex}")
            # Run command 4 times with different -f values
            for f_value in [1, 2, 3, 4]:
                original_tvIndex = self.tvIndex
                self.tvIndex = f"{original_tvIndex}_f{f_value}"
                fast_fading_tv_name = f"{base_tv_name}{self.tvIndex}.h5"
                generate_tv_log = f"{self.logfolder}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_generate_{fast_fading_tv_name}.log"
                generate_fast_fading_tv_cmd = base_cmd + [
                    "-t",
                    "1",
                    "-f",
                    str(f_value),
                    ">",
                    generate_tv_log,
                ]
                self.logger.info(
                    f"Running command: {generate_fast_fading_tv_cmd}")

                if not os.path.exists(f"{self.cubb}/build"):
                    self.logger.info(
                        "Building cuMAC again as no build folder found ......."
                    )
                    self.run_subprocess(self.buildCommand, self.log_file)

                if UePerCell.strip() == "500" and f_value in [2, 4]:
                    self.logger.info(
                        f"Skip this TV, as 4T4R fast fading {f_value} test doesn't support 500 UEs per cell."
                    )
                    self.tvIndex = original_tvIndex
                    continue
                else:
                    if fast_fading_tv_name in os.listdir(self.tvfolder):
                        self.logger.info(
                            f"Found {fast_fading_tv_name}, no need to generate it."
                        )
                    else:
                        self.logger.info(
                            f"Generating TV with -f {f_value}, it will take few minutes, log is stored into {generate_tv_log} ......"
                        )
                        generateTV = self.run_subprocess(
                            " ".join(
                                generate_fast_fading_tv_cmd), self.log_file
                        )
                        print(
                            f"Generate TV result code for -f {f_value}: {generateTV}")

                        if generateTV == 1:
                            self.logger.error(
                                f"Failures found during generating TV with -f {f_value}"
                            )
                            rename_log = f"mv {generate_tv_log} {generate_tv_log}.FAIL"
                            self.run_subprocess(rename_log, self.log_file)
                        else:
                            h5_files = glob.glob(f"{tvfolder}/*.h5")
                            print(f"generate_tv.py: h5_files: {h5_files}")
                            if h5_files:
                                self.logger.info(
                                    f"Generate {h5_files} successfully.")
                                # Use the first (and should be only) h5 file found
                                source_file = h5_files[0]
                                new_name = f"{self.tvfolder}/{fast_fading_tv_name}"

                                # Move the file with explicit source and destination
                                mv_command = f"mv '{source_file}' '{new_name}'"
                                self.run_subprocess(mv_command, self.log_file)
                                self.logger.info(
                                    f"Renamed TV file from {source_file} to {new_name}"
                                )

                                rename_log = (
                                    f"mv {generate_tv_log} {generate_tv_log}.PASS"
                                )
                                self.run_subprocess(rename_log, self.log_file)
                            else:
                                self.logger.error(
                                    f"No TV file generated, please check the {self.log_file}!!!"
                                )
                                rename_log = (
                                    f"mv {generate_tv_log} {generate_tv_log}.FAIL"
                                )
                                self.run_subprocess(rename_log, self.log_file)

                self.tvIndex = (
                    original_tvIndex  # Restore original tvIndex after each iteration
                )

        else:
            if self.option == "3":
                self.logger.info(
                    "Updating the gpuDeviceIdx before generate testMAC TV ....."
                )
                command1 = f"python3 update_parameter.py -f {param_file} -p gpuDeviceIdx -v {self.gpuid}"
                command2 = f"python3 update_parameter.py -f {param_file} -p cpuGpuPerfGapPerUeConst -v 0.01"
                command3 = f"python3 update_parameter.py -f {param_file} -p cpuGpuPerfGapSumRConst -v 0.03"

                self.run_subprocess(command1, self.log_file)
                self.run_subprocess(command2, self.log_file)
                self.run_subprocess(command3, self.log_file)
                self.run_subprocess(self.checkcommand, self.log_file)
                self.logger.info(
                    "Building cuMAC to apply new parameters .......")
                self.run_subprocess(self.buildCommand, self.log_file)

            if not os.path.exists(f"{self.cubb}/build"):
                self.logger.info(
                    "Building cuMAC again as no build folder found ......."
                )
                self.run_subprocess(self.buildCommand, self.log_file)

            generate_tv_log = (
                f"{self.logfolder}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_generate_{base_tv_name}{self.tvIndex}.log"
            )
            # Original behavior for other options
            if "_f" in self.tvIndex:
                f_value = self.tvIndex.split("_f")[1]
                generate_tv_cmd = base_cmd + [
                    "-t",
                    "1",
                    "-f",
                    str(f_value),
                    ">",
                    generate_tv_log,
                ]
                if UePerCell.strip() == "500" and f_value in [2, 4]:
                    self.logger.info(
                        f"Skip this TV, as 4T4R fast fading type f{f_value} test doesn't support 500 UEs per cell."
                    )
                    return
            else:
                generate_tv_cmd = base_cmd + ["-t", str(self.option), ">", generate_tv_log]
            self.logger.info(
                f"Generating TV, it will take few minutes, log is stored into {generate_tv_log} ......"
            )
            generateTV = self.run_subprocess(
                " ".join(generate_tv_cmd), self.log_file)
            print(f"Generate TV result code: {generateTV}")

            if generateTV == 1:
                self.logger.error(
                    f"Failures found during generating TV, please check the {generate_tv_log}"
                )
                rename_log = f"mv {generate_tv_log} {generate_tv_log}.FAIL"
                self.run_subprocess(rename_log, self.log_file)
                return
            else:
                rename_log = f"mv {generate_tv_log} {generate_tv_log}.PASS"
                self.run_subprocess(rename_log, self.log_file)

                h5_files = glob.glob(f"{tvfolder}/*.h5")
                if h5_files:
                    self.logger.info(f"Generate {h5_files} successfully.")
                    if self.option in ["1", "4"] and self.name is False:
                        new_h5_file = f"{self.tvfolder}/{base_tv_name}{self.tvIndex}.h5"
                        rename_command = f"mv {tvfolder}/*.h5 {new_h5_file}"
                        self.logger.info(f"Rename tv file to {new_h5_file}.")
                        rename_tv = self.run_subprocess(
                            rename_command, self.log_file)
                        print(f"rename TV return code: {rename_tv}")
                        if rename_tv == 1:
                            self.logger.error("Failed to rename tv.")
                else:
                    self.logger.error(
                        f"No TV file generated, please check the {self.log_file}!!!"
                    )

    def generate_tv_all_with_tvIndex(self):
        """Generate cuMAC TVs for all parameter combinations in scripts/cumac_tv_parameters.csv."""
        if not os.path.exists(self.param_file_bak):
            backup = f"cp -f {self.param_file} {self.param_file_bak}"
            self.run_subprocess(backup, self.log_file)

        if not os.path.exists(self.csv_filename):
            self.define_parameters_and_combinations()
        with open(self.csv_filename, "r") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            parameters_list = list(csv_reader)

        index = 1
        for params in parameters_list:
            if params["gpuAllocTypeConst"] != params["cpuAllocTypeConst"]:
                continue

            self.tvIndex = f"{index:04}"
            self.logger.info(
                f"Updating TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_TC{self.tvIndex} parameters: {params} ......."
            )
            for param, value in params.items():
                if param != "Index" and value != self.tvIndex:
                    command = f"python3 update_parameter.py -f {self.param_file} -p {param} -v {value}"
                    print(f"command: {command}")
                    self.run_subprocess(command, self.log_file)
            self.run_subprocess(self.checkcommand, self.log_file)
            self.logger.info("Parameters have been updated.")
            gpuAllocTypeConst = self.run_subprocess(
                f"grep 'gpuAllocTypeConst' {self.param_file} | awk '{{print $3}}'",
                self.log_file,
            )
            prdSchemeConst = self.run_subprocess(
                f"grep 'prdSchemeConst' {self.param_file} | awk '{{print $3}}'",
                self.log_file,
            )
            if self.direction.strip() == "UL" and gpuAllocTypeConst.strip() == "0":
                self.logger.info(
                    "Skip this TV, as UL doesn't support gpu allocation type."
                )
                index += 1
                continue
            if self.direction.strip() == "UL" and prdSchemeConst.strip() == "0":
                self.logger.info(
                    "Skip this TV, as UL doesn't support no precoding.")
                index += 1
                continue

            # Define base TV filename pattern
            base_tv_path = os.path.join(
                self.tvfolder,
                f"TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_TC{self.tvIndex}",
            )

            if self.option == "4" and not any(
                f"f{i}" in str(self.tvIndex) for i in (1, 2, 3, 4)
            ):
                # For option 4 fast fading tests, generate paths for f1-f4
                tvfiles = [f"{base_tv_path}_f{f}.h5" for f in (1, 2, 3, 4)]
            else:
                # For standard tests or non-fast-fading option 4
                tvfiles = [f"{base_tv_path}.h5"]

            missing_tvfiles = [f for f in tvfiles if not os.path.exists(f)]
            if missing_tvfiles:
                self.logger.info(
                    f"Not found {missing_tvfiles}, start to generate them."
                )
                self.logger.info(
                    f"Building cuMAC to apply new parameters for TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_TC{self.tvIndex} ......."
                )
                self.run_subprocess(self.buildCommand, self.log_file)
                self.generate_tv()
            else:
                self.logger.info(
                    f"Found all required TV files exist, no need to generate them."
                )

            new_filename = f"{self.cubb}/cuMAC/examples/parameter_{self.tvIndex}.h"
            if not os.path.exists(new_filename):
                rename = f"cp -f {self.param_file} {new_filename}"
                mv = f"mv {new_filename} {self.logfolder}"
                self.run_subprocess(rename, self.log_file)
                self.run_subprocess(mv, self.log_file)
            index += 1

        self.run_subprocess(
            f"cp {self.param_file_bak} {self.param_file}", self.log_file
        )
        self.check_and_rename_log_file(
            self.log_file, "Generate all TVs and rename them by index", "Fail|Err"
        )

    def generate_tv_all_with_name(self):
        """Generate cuMAC TVs for all parameter combinations in scripts/cumac_tv_parameters.csv and rename them with parameters"""
        filepath = f"{self.cubb}/cuMAC/examples"
        command = ""
        if not os.path.exists(self.param_file_bak):
            backup = f"cp -f {self.param_file} {self.param_file_bak}"
            self.run_subprocess(backup)

        if not os.path.exists(self.csv_filename):
            self.define_parameters_and_combinations()

        with open(self.csv_filename, "r") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            parameters_list = list(csv_reader)
        index = 1
        for params in parameters_list:
            if params["gpuAllocTypeConst"] != params["cpuAllocTypeConst"]:
                continue
            tvIndex = f"{index:04}"
            self.logger.info(
                f"Updating TV_cuMAC_{tvIndex} parameters: {params} ......")
            param_combin = ""
            for param, value in params.items():
                if param != "Index" and value != tvIndex:
                    command = f"python3 update_parameter.py -f {self.param_file} -p {param} -v {value}"
                    if param == "numCellConst":
                        param_combin += f"{value}C"
                    elif param == "numUePerCellConst":
                        param_combin += f"_{value}UEPerTTI"
                    elif param == "numActiveUePerCellConst":
                        param_combin += f"_{value}UEPerCell"
                    elif param == "gpuAllocTypeConst":
                        param_combin += f"_gpuAllocType{value}"
                    elif param == "cpuAllocTypeConst":
                        param_combin += f"_cpuAllocType{value}"
                    elif param == "prdSchemeConst":
                        param_combin += f"_Precoder{value}"
                    print(f"command: {command}")
                    self.run_subprocess(command, self.log_file)
            # checkcommand = f"grep 'nBsAntConst\|nUeAntConst\|numSimChnRlz\|gpuDeviceIdx\|seedConst\|numCellConst\|numUePerCellConst\|numActiveUePerCellConst\|nPrbsPerGrpConst\|numUePerCellConst\|gpuAllocTypeConst\|cpuAllocTypeConst\|prdSchemeConst\|rxSchemeConst' {param_file}"
            self.run_subprocess(self.checkcommand, self.log_file)
            self.logger.info("Parameters have been updated.")
            gpuAllocTypeConst = self.run_subprocess(
                f"grep 'gpuAllocTypeConst' {self.param_file} | awk '{{print $3}}'"
            )
            prdSchemeConst = self.run_subprocess(
                f"grep 'prdSchemeConst' {self.param_file} | awk '{{print $3}}'"
            )
            if self.direction.strip() == "UL" and gpuAllocTypeConst.strip() == "0":
                self.logger.info(
                    "Skip this TV, as UL doesn't support gpu allocation type."
                )
                index += 1
                continue
            if self.direction.strip() == "UL" and prdSchemeConst.strip() == "0":
                self.logger.info(
                    "Skip this TV, as UL doesn't support no precoding.")
                index += 1
                continue
            self.logger.info(
                f"Building cuMAC to apply new parameters for TV_{tvIndex} ......."
            )
            self.run_subprocess(self.buildCommand, self.log_file)
            build = self.run_subprocess(self.buildCommand, self.log_file)
            if build != 0:
                self.logger.info(
                    f"Built cuMAC successfully, ready to generate TV_{tvIndex}."
                )
            else:
                self.logger.error(
                    "Failed to build cuMAC, please check the {self.log_file}"
                )
            self.logger.info(
                f"Start to generate TV_{tvIndex} and rename it as TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_{param_combin}.h5."
            )
            self.tvIndex = param_combin
            tvfile = f"{self.tvfolder}/TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_{param_combin}.h5"
            if not os.path.exists(tvfile):
                self.logger.info(f"Not found {tvfile}, start to generate it.")
                self.generate_tv()
            else:
                self.logger.inf(
                    f"Found TV in {self.tvfolder},no need to gerate it")

            new_filename = f"{filepath}/parameter_{tvIndex}.h"
            if not os.path.exists(new_filename):
                rename = f"cp -f {self.param_file} {new_filename}"
                mv = f"mv {new_filename} {self.logfolder}"
                self.run_subprocess(rename, self.log_file)
                self.run_subprocess(mv, self.log_file)
            index += 1
        self.logger.info(
            f"All cuMAC {self.antenna}T{self.antenna}R {self.direction} TVs have been generated and renamed to parameters' name."
        )
        self.run_subprocess(
            f"cp {self.param_file_bak} {self.param_file}", self.log_file
        )
        main_file = f"{self.cubb}/cuMAC/examples/multiCellSchedulerUeSelection/main.cpp"
        restore_main = f"mv {main_file}_default {main_file}"
        self.run_subprocess(restore_main, self.log_file)
        self.check_and_rename_log_file(
            self.log_file,
            "Generate all TVs and rename them with parma names",
            "Fail|Err",
        )

    def generate_single_tv(self):
        """Generate TV file for provided index in scripts/cumac_tv_parameters.csv."""
        command = ""
        # Build base TV filename
        base_tv_name = (
            f"TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_TC{self.tvIndex}"
        )
        base_path = os.path.join(self.tvfolder, base_tv_name)
        base_index = self.tvIndex.split("_")[0]

        # For fast fading tests (option 4), check if we need multiple TV files
        print(f"generate_single_tv: self.tvIndex: {self.tvIndex}")
        if self.option == "4":
            if "_f" not in self.tvIndex:
                tvfiles = [f"{base_path}_f{i}.h5" for i in range(1, 5)]
            else:
                tvfiles = [f"{base_path}.h5"]
        else:
            # For standard tests or specific fast fading test, generate single TV file
            tvfiles = [f"{base_path}.h5"]

        if not os.path.exists(self.param_file_bak):
            backup = f"cp -f {self.param_file} {self.param_file_bak}"
            self.run_subprocess(backup, self.log_file)

        if not os.path.exists(self.csv_filename):
            self.define_parameters_and_combinations()

        # Check if any of the required TV files are missing
        missing_tvfiles = [f for f in tvfiles if not os.path.exists(f)]
        if missing_tvfiles:
            self.logger.info(
                f"Not found {missing_tvfiles}, start to generate them.")
            if check_csv_does_not_contain(self.csv_filename, base_index):
                self.logger.error(
                    f"Not found tv index {base_index} in {self.csv_filename}, please input correct tvIndex."
                )
            else:
                with open(self.csv_filename, "r") as csvfile:
                    csv_reader = csv.DictReader(csvfile)
                    for row in csv_reader:
                        if row["Index"] == base_index:
                            params = row
                            self.logger.info(
                                f"Updating {base_tv_name} parameters: {params} ......"
                            )
                            for param, value in params.items():
                                if param != "Index" and value != base_index:
                                    command = f"python3 update_parameter.py -f {self.param_file} -p {param} -v {value}"
                                    print(f"command: {command}")
                                    self.run_subprocess(command, self.log_file)

                            self.run_subprocess(
                                self.checkcommand, self.log_file)
                            self.logger.info("Parameters have been updated.")
                            gpuAllocTypeConst = self.run_subprocess(
                                f"grep 'gpuAllocTypeConst' {self.param_file} | awk '{{print $3}}'",
                                self.log_file,
                            )
                            prdSchemeConst = self.run_subprocess(
                                f"grep 'prdSchemeConst' {self.param_file} | awk '{{print $3}}'",
                                self.log_file,
                            )
                            if (
                                self.direction.strip() == "UL"
                                and gpuAllocTypeConst.strip() == "0"
                            ):
                                self.logger.info(
                                    "Skip this TV, as UL doesn't support gpu allocation type."
                                )
                                return
                            elif (
                                self.direction.strip() == "UL"
                                and prdSchemeConst.strip() == "0"
                            ):
                                self.logger.info(
                                    "Skip this TV, as UL doesn't support no precoding."
                                )
                                return
                            else:
                                self.logger.info(
                                    f"Building cuMAC to apply parameters for TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_TC{self.tvIndex} ......."
                                )
                                build = self.run_subprocess(
                                    self.buildCommand, self.log_file
                                )
                                if build != 0:
                                    self.logger.info(
                                        f"Built cuMAC successfully, ready to generate TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_TC{self.tvIndex}."
                                    )
                                else:
                                    self.logger.error(
                                        f"Failed to build cuMAC, please check the {self.log_file}"
                                    )
                                # Generate TV
                                self.generate_tv()
                                new_filename = f"{self.cubb}/cuMAC/examples/parameter_{self.tvIndex}.h"
                                # Back up the parameters.h which used by this tvIndex
                                if not os.path.exists(new_filename):
                                    rename = f"cp -f {self.param_file} {new_filename}"
                                    mv = f"mv {new_filename} {self.logfolder}"
                                    self.run_subprocess(rename, self.log_file)
                                    self.run_subprocess(mv, self.log_file)
                                # Roll back to the default parameters.sh
                                self.run_subprocess(
                                    f"cp {self.param_file_bak} {self.param_file}",
                                    self.log_file,
                                )
        else:
            self.logger.info(
                f"Found all required TV files exist, no need to generate them."
            )
        self.check_and_rename_log_file(
            self.log_file, "Generating Single TV", "Fail|ERR"
        )

    def generate(self):
        """Main function to generate TV."""
        antenna_path = f"{self.antenna}T{self.antenna}R"

        self.run_subprocess(f"mkdir -p {self.logfolder}")
        self.run_subprocess(f"mkdir -p {self.tvfolder}")
        self.run_subprocess(f"mkdir -p {self.tvfolder}/{self.direction}")
        if self.option in ["1", "4"]:
            if antenna_path not in self.tvfolder:
                self.run_subprocess(f"mkdir -p {self.tvfolder}/{antenna_path}")
                self.tvfolder = f"{self.tvfolder}/{antenna_path}"
                self.run_subprocess(
                    f"mkdir -p {self.tvfolder}/{self.direction}")

        self.logger.info("Start to generate cuMAC TV.")
        self.logger.info(f"tv folder: {self.tvfolder}")
        self.logger.info(f"log folder: {self.logfolder}")
        self.logger.info(f"log file: {self.log_file}")
        self.logger.info(f"antenna:{self.antenna}")
        self.logger.info(f"diretion: {self.direction}")
        self.logger.info(f"option: {self.option}")
        self.logger.info(f"name:{self.name}")
        self.logger.info(f"cmake:{self.cmake}")
        self.logger.info(f"buildCommand:{self.buildCommand}")

        if self.option in ["1", "4"]:
            if self.name:
                self.redefine_tv_name()
                self.generate_tv_all_with_name()
                self.logger.info(
                    f"Complete to generate cuMAC {self.antenna}T{self.antenna}R {self.direction} all TVs."
                )
            elif self.tvIndex is not None:
                self.generate_single_tv()
            else:
                self.tvIndex = ""
                self.generate_tv_all_with_tvIndex()
                self.logger.info(
                    f"Complete to generate cuMAC {self.antenna}T{self.antenna}R {self.direction} all TVs."
                )
        elif self.option in ["0", "2", "3"]:
            self.generate_tv()
            self.logger.info(
                f"Complete to generate cuMAC {self.antenna}T{self.antenna}R {self.direction} {self.tvIndex} TVs."
            )

    def check_and_rename_log_file(self, log_file, test_type, key_word):
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

                # Rename the file
                os.rename(log_file, new_log_file)
                self.logger.info("Renamed %s to %s", log_file, new_log_file)

        except Exception as e:
            self.logger.error(
                "An error occurred while processing %s: %s", log_file, str(e)
            )

    def parse_args():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Generate TV with updated parameters"
        )
        parser.add_argument(
            "--cmake",
            "-c",
            required=False,
            type=str,
            help="Cmake option; default: cmake -Bbuild -GNinja && cmake --build build",
        )
        parser.add_argument(
            "--dir",
            "-d",
            required=False,
            type=str,
            help="Traffic direction: DL - Downlink TV,UL - Uplink TV; default: DL",
        )
        parser.add_argument(
            "--ant",
            "-a",
            required=False,
            type=str,
            help="Antenna configurations: 4 - for 4T4R,64 - for 64T64R; default: 4",
        )
        parser.add_argument(
            "--log",
            "-l",
            required=False,
            type=str,
            help="log folder; default: current folder",
        )
        parser.add_argument(
            "--tvfolder",
            "-t",
            required=False,
            type=str,
            help="tv folder; default: /opt/nvidia/cuBB/testVectors/cumac",
        )
        parser.add_argument(
            "--name",
            "-n",
            action="store_true",
            help="Generate TV files listed under cuMAC/scripts/cumac_tv_parameters.csv and rename them with parameters when it presents; default:none, generate TV and rename with tvIndex",
        )
        parser.add_argument(
            "--option",
            "-o",
            required=False,
            type=str,
            help="Saving TV options: 0 - not saving TV, 1 - save TV for GPU scheduler, 2 - save TV for CPU scheduler, 3 - save per-cell TVs for testMAC/cuMAC-CP, 4 - save TV for 4T4R fast fading tests; default: 1",
        )
        parser.add_argument(
            "--index",
            "-i",
            required=False,
            type=str,
            help="TV Index in cuMAC/scripts/cumac_tv_parameters.csv,generate TV for this index only when it presents; default: none",
        )
        parser.add_argument(
            "--gpuid",
            "-g",
            required=False,
            type=str,
            help="GPU Device ID; default: 0",
        )
        parser.add_argument(
            "--cubb",
            "-s",
            required=False,
            type=str,
            help="cuBB_SDK folder,default: /opt/nvidia/cuBB",
        )
        return parser.parse_args()

    def main():
        args = TVGenerator.parse_args()
        direction = args.dir
        antenna = args.ant
        cmake = args.cmake
        logfolder = args.log
        tvfolder = args.tvfolder
        name = args.name
        option = args.option
        tvIndex = args.index
        gpuid = args.gpuid
        cubb = args.cubb
        cuMACTV = TVGenerator(
            direction=direction,
            antenna=antenna,
            cmake=cmake,
            logfolder=logfolder,
            tvfolder=tvfolder,
            name=name,
            option=option,
            tvIndex=tvIndex,
            gpuid=gpuid,
            cubb=cubb,
        )
        cuMACTV.generate()


def check_csv_does_not_contain(file_path, content):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            for cell in row:
                if content in str(cell):
                    return False
    return True


if __name__ == "__main__":
    TVGenerator.main()
    """
    examples:
        [] means optional
        1. Generate tv for the provided tvIndex:
          python3 generate_tv.py -i $tvIndex [-d UL -a 4 -l /home/aerial/nfs/Log -t /opt/nvidia/cuBB/testVector/cumac]
        2. Generate tv for testMAC testing:
          python3 generate_tv.py -o 3 [-l /home/aerial/nfs/Log -t /opt/nvidia/cuBB/testVector/cumac]
        3. Generate tv with multiple -f values (1-4):
          python3 generate_tv.py -o 4 [-l /home/aerial/nfs/Log -t /opt/nvidia/cuBB/testVector/cumac]
        4. Generate tv from csv file and save tv with tvIndex:
          python3 generate_tv.py -d UL [-a 4 -o 1 -l /home/aerial/nfs/Log -t /opt/nvidia/cuBB/testVector/cumac]
          python3 generate_tv.py -d DL [-a 4 -o 1 -l /home/aerial/nfs/Log -t /opt/nvidia/cuBB/testVector/cumac]
        5. Generate tv from csv file and save tv with parameters' name:
          python3 generate_tv.py -n -d UL [-a 4 -o 1 -l /home/aerial/nfs/Log -t /opt/nvidia/cuBB/testVector/cumac]
          python3 generate_tv.py -n -d DL [-a 4 -o 1 -l /home/aerial/nfs/Log -t /opt/nvidia/cuBB/testVector/cumac]
    """

    def _get_required_tvfiles(self):
        """
        Determine which TV files need to be generated based on the option.

        Returns:
            list: List of TV file paths that need to be generated
        """

        def _build_tv_path(suffix=""):
            """Helper function to build TV file path with optional suffix."""
            filename = f"TV_cuMAC_{self.antenna}T{self.antenna}R_{self.direction}_TC{self.tvIndex}{suffix}.h5"
            return os.path.join(self.tvfolder, filename)

        # For fast fading tests (option 4)
        if self.option == "4":
            # Check if tvIndex already includes a fast fading identifier
            has_fast_fading = any(f"f{i}" in str(self.tvIndex)
                                  for i in range(1, 5))

            if not has_fast_fading:
                # Generate paths for all four fast fading variations
                return [_build_tv_path(f"_f{i}") for i in range(1, 5)]

        # For all other cases (including option 4 with existing fast fading identifier)
        return [_build_tv_path()]
