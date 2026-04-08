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

import numpy as np
import os
import subprocess
from .parse_power import parse_power


def run(args, designated, mig, mig_gpu, command, vectors, mode, target, k):

    results = None

    subs, streams, steps = designated

    if args.force is not None:
        if args.force == 0:
            connections = streams
        else:
            connections = args.force
    else:
        connections = np.min([32, int(np.power(2, np.floor(np.log2(96 / (subs + 1)))))])

    if args.is_power:
        from .configure_power import configure

        system = configure(
            args,
            designated,
            mig,
            mig_gpu,
            connections,
            command,
            vectors,
            mode,
            k,
            target,
        )
        ofile = None

        if not os.path.exists("power.txt"):

            if args.is_test:
                print(
                    " ".join(
                        [
                            "nvidia-smi",
                            "-i",
                            f"{args.gpu}",
                            "--query-gpu=clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu,utilization.memory,temperature.gpu",
                            "-lms",
                            "10",
                            "--format=csv",
                        ]
                    )
                )
            else:
                ofile = open("power.txt", "w")
                proc = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "-i",
                        f"{args.gpu}",
                        "--query-gpu=clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu,utilization.memory,temperature.gpu",
                        "-lms",
                        "10",
                        "--format=csv",
                    ],
                    stdout=ofile,
                )

        if args.is_test:
            print(system)
            if not args.is_save_buffers:
                os.remove(vectors)
        else:
            if args.is_unsafe:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)
            else:
                buffer = system.split(args.cfld)[0].strip().split()
                env = {}
                cmd_prefix_tokens = []

                for itm in buffer:
                    mapping = itm.split("=", 1)
                    if len(mapping) >= 2:
                        env[mapping[0]] = mapping[1]
                    else:
                        cmd_prefix_tokens.append(itm)

                cmd_tail = args.cfld + system.split(args.cfld)[-1].strip()
                cmd = (" ".join(cmd_prefix_tokens) + " " + cmd_tail).strip() if cmd_prefix_tokens else cmd_tail
                cmd, stdout = cmd.split(">")

                ofile = open(stdout, "w")

                try:
                    mproc = subprocess.Popen(cmd.split(), env=env, stdout=ofile)
                    mproc.wait(100)
                except subprocess.TimeoutExpired:
                    mproc.kill()
                finally:
                    ofile.close()
                    if not args.is_save_buffers:
                        os.remove(vectors)

        if ofile is not None:
            proc.kill()
            ofile.close()

            ifile = open("power.txt", "r")
            lines = ifile.readlines()
            ifile.close()
            os.remove("power.txt")

            if mig is None:
                os.remove(f"buffer-{str(k).zfill(2)}.txt")
            else:
                os.remove(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt")

            results = parse_power(lines)

    else:
        if args.is_debug:
            from .configure_debug import configure

            system = configure(
                args,
                designated,
                mig,
                mig_gpu,
                connections,
                command,
                vectors,
                mode,
                k,
                target,
            )
            if args.is_test:
                print(system)
                if not args.is_save_buffers:
                    os.remove(vectors)
            else:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)

        else:
            from .configure import configure

            system = configure(
                args,
                designated,
                mig,
                mig_gpu,
                connections,
                command,
                vectors,
                mode,
                k,
                target,
            )
            if args.is_test:
                print(system)
                if not args.is_save_buffers:
                    os.remove(vectors)
            else:
                try:
                    os.system(system)
                finally:
                    if not args.is_save_buffers:
                        os.remove(vectors)

                if mig is None:
                    ifile = open(f"buffer-{str(k).zfill(2)}.txt", "r")
                    lines = ifile.readlines()
                    ifile.close()
                    if not args.is_save_buffers:
                        os.remove(f"buffer-{str(k).zfill(2)}.txt")
                else:
                    ifile = open(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt", "r")
                    lines = ifile.readlines()
                    ifile.close()
                    if not args.is_save_buffers:
                        os.remove(f"buffer-{mig_gpu}-{str(k).zfill(2)}.txt")

                if args.is_ref_check:
                    from ..check_ref_mismatch import check_ref_mismatch
                    # Diagnostic only; does not gate sweep results.
                    _, ref_data = check_ref_mismatch(lines, cell_count=k)

                if args.is_check_traffic:
                    from ..error import parse
                else:
                    from .sweep import parse

                results = parse(args, lines)

                if args.is_ref_check and results is not None:
                    results.update(ref_data)

    return results
