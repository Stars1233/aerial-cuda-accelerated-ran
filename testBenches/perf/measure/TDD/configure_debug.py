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
import argparse
from datetime import datetime
import shlex


def _uses_ncu_mps(args: argparse.Namespace) -> bool:
    """Return whether NCU must attach through its MPS control workflow."""
    return (
        args.debug_mode == "ncu"
        and not args.is_no_mps
        and not args.is_use_green_contexts
    )


def configure(args, mig, mig_gpu, connections, command, vectors, mode, k, target):

    uses_ncu_mps = _uses_ncu_mps(args)

    if args.is_no_mps:
        if args.mig is None:
            system = f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
        else:
            system = (
                f"CUDA_VISIBLE_DEVICES={mig} CUDA_DEVICE_MAX_CONNECTIONS={connections}"
            )
    else:
        if args.mig is None:
            system = "CUDA_MPS_PIPE_DIRECTORY=. CUDA_MPS_LOG_DIRECTORY=. CUDA_LOG_DIRECTORY=."
        else:
            system = (
                f"CUDA_MPS_PIPE_DIRECTORY={mig_gpu} "
                f"CUDA_MPS_LOG_DIRECTORY={mig_gpu} CUDA_LOG_DIRECTORY={mig_gpu}"
            )

        if len(system) > 0:
            system = " ".join([system, f"CUDA_DEVICE_MAX_CONNECTIONS={connections}"])
        else:
            system = f"CUDA_DEVICE_MAX_CONNECTIONS={connections}"

    if args.pattern == "dddsu":
        if args.prach_tgt is not None:
            system = " ".join(
                [system, f"PRACH_TPC_MASK_LOW={format(args.prach_tgt,'#x')}"]
            )

    timestamp = "-".join(
        str(datetime.now()).split(".")[0].replace("-", ".").replace(":", ".").split()
    )
    nsys_capture_range = " --capture-range=cudaProfilerApi" if args.is_enable_nvprof else ""
    nsys_export_sqlite = " --export sqlite" if args.is_enable_sqlite else ""

    try:
        from .._debug_mode_extensions import configure as configure_extension
    except ModuleNotFoundError as error:
        optional_module = f"{__package__.rsplit('.', 1)[0]}._debug_mode_extensions"
        if error.name != optional_module:
            raise
        extension_config = None
    else:
        extension_config = configure_extension("tdd", args, timestamp, k)

    if extension_config is not None:
        system = " ".join([system, extension_config.command])
    elif args.debug_mode in ("nsys", "nsys_simple"):
        nsys_gpu_metrics = ""
        if args.is_enable_gpu_metric or args.gpu_metric_set:
            nsys_gpu_metrics = f" --gpu-metrics-devices {args.gpu}"
            if args.gpu_metric_set:
                nsys_gpu_metrics += f" --gpu-metrics-set=file:{args.gpu_metric_set}"
        # nsys runs under `sudo -n -E` for GPU-metrics privilege (RmProfilingAdminOnly=1). Because the NFS
        # workspace is root_squash, it writes into a per-run PRIVATE temp dir (mktemp -d, mode 0700, random
        # name -- no predictable /tmp path or pre-created symlink); the report is moved back + removed below.
        system = " ".join(['__nsysdir="$(mktemp -d /tmp/nsys-XXXXXX)" || { echo "ERROR: mktemp -d failed for nsys output" >&2; exit 1; };', system])
        _rep = f"$__nsysdir/profiler-{str(k).zfill(2)}-{timestamp}"
        if args.debug_mode == "nsys":
            nsys_cuda_trace = "--trace cuda-sw,nvtx" if args.is_nsys_use_cuda_sw else "--trace cuda,nvtx"
            system = " ".join(
                [
                    system,
                    f"sudo -n -E nsys profile --resolve-symbols=false -o {_rep} --cpuctxsw system-wide{nsys_export_sqlite} {nsys_cuda_trace} --cuda-graph-trace node{nsys_gpu_metrics}{nsys_capture_range}",
                ]
            )
        else:
            nsys_simple_trace = "-t cuda-sw" if args.is_nsys_use_cuda_sw else "-t cuda"
            system = " ".join(
                [
                    system,
                    f"sudo -n -E nsys profile --resolve-symbols=false -o {_rep} -s none --cpuctxsw=none --cuda-event-trace=false {nsys_simple_trace}{nsys_gpu_metrics}{nsys_capture_range}",
                ]
            )
    elif args.debug_mode == "ncu":
        if uses_ncu_mps:
            # cubb_gpu_test_bench uses CUDA graphs in its normal performance
            # configuration. NCU MPS range replay cannot capture cuGraphLaunch,
            # so use the supported MPS kernel-replay workflow. Kernel replay
            # intentionally profiles kernels individually; it does not preserve
            # cross-context concurrency in each result.
            ncu = (
                "ncu --mps control --replay-mode kernel "
                "--kernel-name 'regex:^(?!.*(convert_kernel|delay_kernel_us|delay_ns_wait_until))'"
            )
        else:
            ncu = "CUDA_LAUNCH_BLOCKING=1 ncu"

        system = " ".join(
            [
                system,
                f"{ncu} --target-processes all --force-overwrite --section InstructionStats --section ComputeWorkloadAnalysis --section LaunchStats --section Occupancy --section WarpStateStats --section SpeedOfLight --section MemoryWorkloadAnalysis --import-source yes --clock-control none --cache-control none --export ncu-{str(k).zfill(2)}-{timestamp}",
            ]
        )
    else:
        raise NotImplementedError(f"Debug mode {args.debug_mode} is not supported")

    if args.numa is not None:
        system = " ".join(
            [system, f"numactl --cpunodebind={args.numa} --membind={args.numa}"]
        )

    if args.pattern == "dddsu":

        if extension_config is not None and extension_config.wraps_command:

            system = " ".join(
                [
                    system,
                    f'{command} "-i {vectors} -r {args.iterations} -w {args.delay} -u 3 -d 0 -m {mode}',
                ]
            )

        else:

            system = " ".join(
                [
                    system,
                    f"{command} -i {vectors} -r {args.iterations} -w {args.delay} -u 3 -d 0 -m {mode}",
                ]
            )

    else:
        patternUMode = 6 if args.pattern == "dddsuudddd_mMIMO" else 5

        if extension_config is not None and extension_config.wraps_command:

            system = " ".join(
                [
                    system,
                    f'{command} "-i {vectors} -r {args.iterations} -w {args.delay} -u {patternUMode} -d 0 -m {mode}',
                ]
            )

        else:
            system = " ".join(
                [
                    system,
                    f"{command} -i {vectors} -r {args.iterations} -w {args.delay} -u {patternUMode} -d 0 -m {mode}",
                ]
            )

        if args.is_pusch_cascaded:
            system = " ".join([system, "-B"])

        if args.is_pucch_cascaded:
            system = " ".join([system, "-A"])

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

        include_mps_target = (
            extension_config.include_mps_target
            if extension_config is not None
            else args.debug_mode != "ncu" or uses_ncu_mps
        )
        if include_mps_target:
            system = " ".join([system, f"--M {flat_target}"])

    if args.is_2_cb_per_sm:
        system = " ".join([system, "-L"])

    if args.is_priority:
        system = " ".join([system, "-a"])

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

    if getattr(args, "trt_chest_config", ""):
        system = " ".join([system, f"--E {shlex.quote(args.trt_chest_config)}"])

    if not args.is_no_pusch:
        system = " ".join([system, "--U"])
  
    if not args.is_no_pdsch:
        system = " ".join([system, "--D"])

    if args.is_enable_nvprof:
        system = " ".join([system, "-v"])

    if args.is_setup_once:
        system = " ".join([system, "--O"])

    if args.is_ref_check:
        system = " ".join([system, "-k --k -b --c PUSCH,PDSCH,PDCCH,PUCCH,SSB,DLBFW,ULBFW,CSIRS,PRACH,SRS"])

    if args.device_max_connections is not None:
        system = " ".join([system, f"--C {args.device_max_connections}"])

    if mig is None:
        system = " ".join([system, f">buffer-{str(k).zfill(2)}.txt"])
    else:
        system = " ".join([system, f">buffer-{mig_gpu}-{str(k).zfill(2)}.txt"])

    # Move nsys's report from the private temp dir into the working dir with `mv`, run as the invoking
    # user: the cross-filesystem move re-creates the files owned by that user (who CAN write the
    # root_squash NFS workspace), so no chown is needed; then the temp dir is removed. Exact extensions
    # only, symlinks skipped, and nsys's exit code is preserved so downstream error handling is unchanged.
    if args.debug_mode in ("nsys", "nsys_simple"):
        _rep = f"$__nsysdir/profiler-{str(k).zfill(2)}-{timestamp}"
        system = " ".join(
            [system, f'; __rc=$?; for _e in nsys-rep qdstrm sqlite; do _f="{_rep}.$_e"; [ -f "$_f" ] && [ ! -L "$_f" ] && mv -f "$_f" ./; done; rm -rf "$__nsysdir"; exit $__rc']
        )

    if extension_config is not None and extension_config.wraps_command:
        return system + '"'
    return system
