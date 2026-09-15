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
from cuda.bindings import runtime

from .FDD.run import run_FDD
from .TDD.run import run_TDD


def measure(base, args):

    if os.geteuid() == 0:
        sudo = ""
    else:
        sudo = "sudo "

    # Get device count
    err, device_count = runtime.cudaGetDeviceCount()
    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Failed to get device count: {err}")

    # Determine which device to use
    if device_count == 1:  # with CUDA_VISIBLE_DEVICES set, only 1 device is visible
        device_id = 0
    else:
        device_id = args.gpu

    # Set the device
    err = runtime.cudaSetDevice(device_id)
    if err != (runtime.cudaError_t.cudaSuccess,):
        raise RuntimeError(f"Failed to set device {device_id}: {err}")

    # Get multiprocessor count (SM count)
    err, sms = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrMultiProcessorCount, device_id)
    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Failed to get multiprocessor count: {err}")

    if args.is_power:
        if os.path.exists("power.txt"):
            os.remove("power.txt")

    try:

        if "FDD" in args.uc:
            run_FDD(args, sms)
        else:
            run_TDD(args, sms)

    finally:
        if not args.is_no_mps:
            # quit MPS server
            os.system(
                "if pgrep -f '(^|/)nvidia-cuda-mps-control -d$' >/dev/null; then "
                f"echo quit | CUDA_VISIBLE_DEVICES={args.gpu} "
                "CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. "
                "nvidia-cuda-mps-control; fi"
            )
        # no need to reset GPU
        # os.system(f"{sudo}nvidia-smi -i {args.gpu} -rgc")
