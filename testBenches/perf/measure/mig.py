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

import os
import numpy as np
import glob
from functools import partial

import shutil
from multiprocessing import Pool

from .FDD.run import run_FDD
from .TDD.run import run_TDD


def measure(base, args):

    if os.geteuid() == 0:
        sudo = ""
    else:
        sudo = "sudo "
    
    os.system(f"{sudo}nvidia-smi mig -i {args.gpu} -lgip >buffer.txt")

    ifile = open("buffer.txt", "r")
    lines = ifile.readlines()
    ifile.close()
    os.remove("buffer.txt")

    mig_ids = {}

    if args.mig_instances > 1:
        if args.is_debug:
            base.error("Debug mode cannot be run with maximal MIG parallelization")

    for line in lines:
        lst = line.split()

        if len(lst) > 2 and lst[2] == "MIG" and int(lst[1]) == args.gpu:
            mig_ids[int(lst[4])] = (int(lst[5].split("/")[0]), int(lst[8]))

    sms = mig_ids[args.mig][1]

    if args.mig_instances > mig_ids[args.mig][0]:
        print(
            f"Warning: the HW can provide at most {mig_ids[args.mig][0]} GPU instances for the desired configuration"
        )

    number_of_instances = np.min([mig_ids[args.mig][0], args.mig_instances])

    if number_of_instances == 0:
        base.error("no HW resources available for the desidered configuration")

    instances = [str(args.mig)] * number_of_instances

    for instance in instances:

        os.system(f"{sudo}nvidia-smi mig -i {args.gpu} -cgi {instance} > buffer.txt")

        ifile = open("buffer.txt")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        if len(lines) != 1:
            raise SystemError

        if "Successfully created GPU instance ID" not in lines[0]:
            raise SystemError

        lst = lines[0].split()

        gi_id = int(lst[5])

        os.system(
            f"{sudo}nvidia-smi mig -i {args.gpu} -gi {gi_id} --list-compute-instance-profiles >buffer.txt"
        )

        ifile = open("buffer.txt", "r")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        gi_ids = {}
        
        # Check if the command failed due to insufficient permissions
        failed_permissions = any("Insufficient Permissions" in line for line in lines)
        
        if failed_permissions:
            raise RuntimeError("Cannot get MIG profiles due to permissions issue.")
        
        # Parse the normal output
        for line in lines:
            lst = line.split()

            # Handle the actual format with | characters
            # Format: |   0      1       MIG 1c.2g.48gb       0      2/2           46        2     2     0   |
            if len(lst) > 7 and len(lst[0]) == 1 and lst[0] == "|" and lst[1].isdigit():
                try:
                    gpu_id = int(lst[1])
                    if gpu_id == args.gpu and "MIG" in line:
                        # Find SM count (look for number after instances like "2/2")
                        for i, item in enumerate(lst):
                            if "/" in item and i + 1 < len(lst) and lst[i + 1].isdigit():
                                sm_count = int(lst[i + 1])
                                # Find profile ID (look for number with optional *)
                                for j, profile_item in enumerate(lst):
                                    if profile_item.replace("*", "").isdigit() and j > 3:
                                        profile_id = profile_item.replace("*", "")
                                        gi_ids[sm_count] = profile_id
                                        break
                                break
                except (ValueError, IndexError):
                    continue
        
        # Validate that gi_ids is not empty before checking if sms exists
        if not gi_ids:
            base.error(f"No compute instance profiles found. The parsing loop did not populate any entries.")
        
        if sms not in gi_ids:
            base.error(f"No compute instance profile found for {sms} SMs. Available mappings: {gi_ids}")
            
        gi_profile = gi_ids[sms]

        os.system(
            f"{sudo}nvidia-smi mig -i {args.gpu} -cci {gi_profile} -gi {gi_id} >buffer.txt"
        )

        ifile = open("buffer.txt")
        lines = ifile.readlines()
        ifile.close()
        os.remove("buffer.txt")

        if len(lines) != 1:
            raise SystemError

        if "Successfully created compute instance ID" not in lines[0]:
            raise SystemError

    os.system(f"{sudo}nvidia-smi -L >buffer.txt")
    ifile = open("buffer.txt", "r")
    lines = ifile.readlines()
    ifile.close()
    os.remove("buffer.txt")

    uuid = []

    # Extract UUIDs from nvidia-smi -L output
    for line in lines:
        if "MIG" in line and "Device" in line and "UUID:" in line:
            # Extract UUID from line like: "  MIG 2g.48gb     Device  0: (UUID: MIG-...)"
            uuid_part = line.split("UUID:")[-1].strip().replace(")", "")
            uuid.append(uuid_part)
            if len(uuid) >= number_of_instances:
                break

    if args.is_power:
        if os.path.exists("power.txt"):
            os.remove("power.txt")

    pMIGS = glob.glob("MIG*")

    for pMIG in pMIGS:
        if os.path.isdir(pMIG):
            shutil.rmtree(pMIG, ignore_errors=True)

    try:
        if "FDD" in args.uc:
            pool = Pool(number_of_instances)
            functor = partial(run_FDD, args, sms)
            pool.map(functor, uuid)
        else:
            if number_of_instances == 1:
                run_TDD(args, sms, uuid[0])
            else:
                # Run TDD in parallel on multiple MIG instances
                pool = Pool(number_of_instances)
                functor = partial(run_TDD, args, sms)
                pool.map(functor, uuid)
                pool.close()
                pool.join()
    finally:
        if not args.is_no_mps:
            for gpu_instance in uuid:
                gpu_instance_folder = gpu_instance.replace("/", "-")
                # quit MPS server
                os.system(
                    f"echo quit | CUDA_VISIBLE_DEVICES={gpu_instance} CUDA_MPS_PIPE_DIRECTORY={gpu_instance_folder} CUDA_LOG_DIRECTORY={gpu_instance_folder} nvidia-cuda-mps-control"
                )
                shutil.rmtree(gpu_instance_folder, ignore_errors=True)
                if args.is_test:
                    print(f"Removed: {gpu_instance_folder}")

        # clean up MIG
        try:
            os.system(f"{sudo}nvidia-smi -i {args.gpu} -mig 0 >/dev/null")
        except Exception as e:
            print(f"Error cleaning up MIG: {e}")

        # no need to reset GPU
        # os.system(f"{sudo}nvidia-smi -i {args.gpu} -rgc")
