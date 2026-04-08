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

# import numpy as np


def configure(args, mig, mig_gpu, connections, command, vectors, mode, k, target):

    if args.is_no_mps:
        if args.mig is None:
            system = f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
        else:
            system = (
                f"CUDA_VISIBLE_DEVICES={mig} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
            )
    else:
        if args.mig is None:
            system = f"CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS={connections}"
        else:
            system = f"CUDA_MPS_PIPE_DIRECTORY={mig_gpu} CUDA_LOG_DIRECTORY={mig_gpu} CUDA_DEVICE_MAX_CONNECTIONS={connections}"

    if args.pattern == "dddsu":
        if args.prach_tgt is not None:
            system = " ".join(
                [system, f"PRACH_TPC_MASK_LOW={format(args.prach_tgt,'#x')}"]
            )

    if args.numa is not None:
        system = " ".join(
            [system, f"numactl --cpunodebind={args.numa} --membind={args.numa}"]
        )

    if args.pattern == "dddsu":

        system = " ".join(
            [
                system,
                f"{command} -i {vectors} -r {args.iterations} -w {args.delay} -u 3 -d 0 -m {mode}",
            ]
        )

    else:

        patternUMode = 6 if args.pattern == "dddsuudddd_mMIMO" else 5
        system = " ".join(
            [
                system,
                f"{command} -i {vectors} -r {args.iterations} -w {args.delay} -u {patternUMode} -d 0 -m {mode}",
            ]
        )

        if args.is_pusch_cascaded:
            system = " ".join([system, "-B"])

    if args.is_ldpc_parallel:
        system = " ".join([system, "-K 1"])

    if not args.is_no_mps and args.is_prach and args.is_isolated_prach:
        system = " ".join([system, "--P"])

    if not args.is_no_mps and args.is_pdcch and args.is_isolated_pdcch:
        system = " ".join([system, "--Q"])

    if not args.is_no_mps and args.is_pucch and args.is_isolated_pucch:
        system = " ".join([system, "--X"])

    if args.is_groups_pdsch:
        system = " ".join([system, "--G"])
        if args.is_pack_pdsch:
            system = " ".join([system, "--b"])

    if args.is_groups_pusch:
        system = " ".join([system, "--g"])

    if not args.is_no_mps:
        flat_target = ",".join(map(str, target))
        system = " ".join([system, f"--M {flat_target}"])

    if args.is_2_cb_per_sm:
        system = " ".join([system, "-L"])

    if args.is_priority:
        system = " ".join([system, "-a"])

    if args.is_check_traffic:
        system = " ".join([system, "-k -b"])

    if args.is_srs_isolate:
        system = " ".join([system, "--Z"])

    if args.is_ssb_isolate:
        system = " ".join([system, "--B"])
        
    if args.is_mac:
        system = " ".join([system, "--T"])

    if args.mac2 > 0:
        system = " ".join([system, "--V"])

    if args.is_mac_timer:
        system = " ".join([system, "--R"])
        
    if args.is_use_green_contexts:
        system = " ".join([system, "-n"])

    if not args.is_no_pusch:
        system = " ".join([system, "--U"])
  
    if not args.is_no_pdsch:
        system = " ".join([system, "--D"])

    if args.is_enable_nvprof:
        system = " ".join([system, "-v"])

    if args.is_ref_check:
        system = " ".join([system, "-k --k -b --c PUSCH,PDSCH,PDCCH,PUCCH,SSB,DLBFW,ULBFW,CSIRS,PRACH,SRS"])

    if mig is None:
        system = " ".join([system, f">buffer-{str(k).zfill(2)}.txt"])
    else:
        system = " ".join([system, f">buffer-{mig_gpu}-{str(k).zfill(2)}.txt"])

    return system
