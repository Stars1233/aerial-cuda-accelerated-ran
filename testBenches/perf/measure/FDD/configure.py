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

def configure(
    args, designated, mig, mig_gpu, connections, command, vectors, mode, k, target
):

    subs, streams, steps = designated

    if args.is_no_mps:
        if args.mig is None:
            system = f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
        else:
            system = (
                f"CUDA_VISIBLE_DEVICES={mig} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
            )
    else:
        if args.mig is None:
            system = f"CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={target} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
        else:
            system = f"CUDA_MPS_PIPE_DIRECTORY={mig_gpu} CUDA_LOG_DIRECTORY={mig_gpu} CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={target} CUDA_DEVICE_MAX_CONNECTIONS={connections}"

    if args.numa is not None:
        system = " ".join(
            [system, f"numactl --cpunodebind={args.numa} --membind={args.numa}"]
        )

    system = " ".join(
        [
            system,
            f"{command} -i {vectors} -r {args.iterations} -w {args.delay} -u 4 -d 0 -m {mode} -C {subs} -S {streams} -I {steps}",
        ]
    )

    if args.is_groups_pdsch:
        system = " ".join([system, "--G"])
        if args.is_pack_pdsch:
            system = " ".join([system, "--b"])

    if args.is_groups_pusch:
        system = " ".join([system, "--g"])

    if args.is_2_cb_per_sm:
        system = " ".join([system, "-L"])

    if args.is_priority:
        system = " ".join([system, "-a"])

    if args.is_check_traffic:
        system = " ".join([system, "-k -b"])

    if args.is_use_green_contexts:
        system = " ".join([system, "-n"])

    if args.is_enable_nvprof:
        system = " ".join([system, "-v"])

    if args.is_ref_check:
        system = " ".join([system, "-k --k -b --c PUSCH,PDSCH,PDCCH,PUCCH,SSB,DLBFW,ULBFW,CSIRS,PRACH,SRS"])

    if mig is None:
        system = " ".join([system, f">buffer-{str(k).zfill(2)}.txt"])
    else:
        system = " ".join([system, f">buffer-{mig_gpu}-{str(k).zfill(2)}.txt"])

    return system
