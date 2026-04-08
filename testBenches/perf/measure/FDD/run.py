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

import json
import numpy as np
import os
import uuid
import sys

from .traffic import traffic_avg, traffic_het
from .execute import run
from .properties import auto_het_subs, auto_avg_subs


def run_FDD(args, sms, mig=None):

    ifile = open(args.config)
    config = json.load(ifile)
    ifile.close()

    ifile = open(args.uc)
    uc = json.load(ifile)
    ifile.close()

    uc_basename = os.path.basename(args.uc)
    if args.is_graph:
        output = "sweep_graphs_" + uc_basename.replace("uc_", "").replace(
            "_TDD.json", ""
        ).replace("_FDD.json", "")
        mode = 1
    else:
        output = "sweep_streams_" + uc_basename.replace("uc_", "").replace(
            "_TDD.json", ""
        ).replace("_FDD.json", "")
        mode = 0

    data_targets = None
    target = None

    if not args.is_no_mps:
        if args.target is not None:
            if args.target[0].isnumeric():
                target = int(np.round(100 * float(args.target[0]) / sms))
                output = args.target[0].zfill(3) + "_" + output
            else:
                ifile = open(args.target[0], "r")
                data_targets = json.load(ifile)
                ifile.close()
        else:
            raise NotImplementedError

    if mig is not None:
        mig_gpu = mig.replace("/", "-")
        output = output + "_" + mig_gpu
    else:
        mig_gpu = None

    if args.seed is not None:
        output = output + "_s" + str(args.seed) + "_" + str(uuid.uuid4())

    if not args.is_no_mps:
        if mig is None:
            system = f"CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=."
        else:
            os.mkdir(mig_gpu)

            if args.is_test:
                print(f"Created: {mig_gpu}")

            system = f"CUDA_VISIBLE_DEVICES={mig} CUDA_MPS_PIPE_DIRECTORY={mig_gpu} CUDA_LOG_DIRECTORY={mig_gpu}"

        system = " ".join([system, "nvidia-cuda-mps-control -d"])

        os.system(system)

    k = args.start

    sweeps = {}
    powers = {}

    # save test configs and GPU product name (exclude inline config/uc blobs; use vector_files when from YAML)
    def _test_config_for_save():
        d = dict(vars(args))
        d.pop("inline_config_obj", None)
        d.pop("inline_uc_obj", None)
        return d

    if args.is_power:
        powers['testConfig'] = _test_config_for_save()
        powers['testConfig']['smAllocation'] = target
    else:
        sweeps['testConfig'] = _test_config_for_save()
        sweeps['testConfig']['smAllocation'] = target
    
    if args.seed is not None:
        np.random.seed(args.seed)

    command = os.path.join(args.cfld, "cubb_gpu_test_bench/cubb_gpu_test_bench")

    is_scan = args.subs is None

    while k <= args.cap:

        if mig is None:
            vectors = os.path.join(os.getcwd(), "vectors-" + str(k).zfill(2) + ".yaml")
        else:
            vectors = os.path.join(
                os.getcwd(), "vectors-" + mig_gpu + "-" + str(k).zfill(2) + ".yaml"
            )

        if "_het_" in args.uc:

            designated_subs = None
            designated_streams = None
            designated_steps = None

            if is_scan:

                avail_subs = sorted(auto_het_subs, reverse=True)

                for subs in avail_subs:

                    if k % subs == 0 and k // subs <= 32:
                        designated_subs = subs
                        designated_streams = k // subs
                        designated_steps = 1
                        break

            else:

                if k % args.subs == 0:
                    designated_subs = args.subs
                    designated_streams = k // args.subs
                    designated_steps = 1

            if designated_subs is None:
                k += 1
                continue
            else:

                if not args.is_no_mps and data_targets is not None:

                    key = str(k).zfill(2)
                    target = data_targets.get(key, None)

                    if target is None:
                        k += 1
                        continue
                    else:
                        target = int(np.round(100 * float(target) / sms))

                uc_keys = list(uc.keys())

                uc_dl = [x for x in uc_keys if "PDSCH" in x]

                if len(uc_dl) != 1:
                    sys.exit("error: use case file exhibits an unexpected structure")
                else:
                    uc_dl = uc_dl[0]

                uc_ul = [x for x in uc_keys if "PUSCH" in x]

                if len(uc_ul) != 1:
                    sys.exit("error: use case file exhibits an unexpected structure")
                else:
                    uc_ul = uc_ul[0]

                testcases_dl = uc[uc_dl]
                testcases_ul = uc[uc_ul]

                filenames_dl = config[uc_dl]
                filenames_ul = config[uc_ul]

                message = "Number of active cells: " + str(k)

                if target is not None:
                    message += (
                        "("
                        + str(2 * int(np.round(target * sms / 200)))
                        + ","
                        + str(designated_subs)
                        + ")"
                    )

                print(message)

                traffic_het(
                    args,
                    vectors,
                    k,
                    (testcases_dl, testcases_ul),
                    (filenames_dl, filenames_ul),
                )

                designated = (designated_subs, designated_streams, designated_steps)

                if args.is_power:
                    powers[str(k).zfill(2)] = run(
                        args,
                        designated,
                        mig,
                        mig_gpu,
                        command,
                        vectors,
                        mode,
                        target,
                        k,
                    )
                else:
                    sweeps[str(k).zfill(2)] = run(
                        args,
                        designated,
                        mig,
                        mig_gpu,
                        command,
                        vectors,
                        mode,
                        target,
                        k,
                    )

        elif "_avg_" in args.uc:

            peak_key = "Peak: " + str(k)
            if peak_key not in uc:
                # Inline UC was built from YAML cap (e.g. 25); --start/--cap can exceed it. Create missing Peak entry.
                template_peaks = [key for key in uc.keys() if key.startswith("Peak: ")]
                if not template_peaks:
                    sys.exit("error: use case file has no Peak: N entry to template from")
                template_key = max(template_peaks, key=lambda x: int(x.split(":", 1)[1].strip()))
                template = uc[template_key]
                new_interval = {}
                for subcase, ch_dict in template.items():
                    new_interval[subcase] = {}
                    for ch_key, testcase_list in ch_dict.items():
                        base_ch = ch_key.split(" - ", 1)[1] if " - " in ch_key else ch_key
                        if base_ch in {"MAC", "MAC2"}:
                            new_interval[subcase][ch_key] = list(testcase_list)
                        else:
                            testcase_id = testcase_list[0] if testcase_list else "F08-PP-00"
                            new_interval[subcase][ch_key] = [testcase_id] * k
                uc[peak_key] = new_interval
            interval = uc[peak_key]

            for subcase in interval.keys():

                uc_keys = list(interval[subcase].keys())

                uc_dl = [x for x in uc_keys if "PDSCH" in x]

                if len(uc_dl) != 1:
                    sys.exit("error: use case file exhibits an unexpected structure")
                else:
                    uc_dl = uc_dl[0]

                uc_ul = [x for x in uc_keys if "PUSCH" in x]

                if len(uc_ul) != 1:
                    sys.exit("error: use case file exhibits an unexpected structure")
                else:
                    uc_ul = uc_ul[0]

                testcases_dl = interval[subcase][uc_dl]
                testcases_ul = interval[subcase][uc_ul]

                filenames_dl = config[uc_dl]
                filenames_ul = config[uc_ul]

                designated_subs = None
                designated_streams = None
                designated_steps = None

                # The assumption here is that the peak == 0

                if is_scan:

                    avail_subs = sorted(auto_avg_subs, reverse=True)

                    for subs in avail_subs:

                        if (
                            len(testcases_dl) % subs == 0
                            and len(testcases_dl) // subs <= 32
                        ):
                            designated_subs = subs
                            designated_streams = len(testcases_dl) // subs
                            designated_steps = 1
                            break

                else:

                    if len(testcases_dl) % args.subs == 0:
                        designated_subs = args.subs
                        designated_streams = len(testcases_dl) // args.subs
                        designated_steps = 1

                if designated_subs is None:
                    k += 1
                    continue
                else:
                    label = int(subcase.replace("Average: ", ""))

                    if not args.is_no_mps and data_targets is not None:

                        key = "+".join([str(k).zfill(2), str(label).zfill(2)])

                        target = data_targets.get(key, None)

                        if target is None:
                            k += 1
                            continue
                        else:
                            target = int(np.round(100 * float(target) / sms))

                    message = "Number of active cells: " + str(k) + "+" + str(label)

                    if target is not None:
                        message += (
                            "("
                            + str(2 * int(np.round(target * sms / 200)))
                            + ","
                            + str(designated_subs)
                            + ")"
                        )

                    print(message)

                    traffic_avg(
                        args,
                        vectors,
                        (testcases_dl, testcases_ul),
                        (filenames_dl, filenames_ul),
                    )

                    designated = (designated_subs, designated_streams, designated_steps)

                    if args.is_power:
                        powers["+".join([str(k).zfill(2), str(label).zfill(2)])] = run(
                            args,
                            designated,
                            mig,
                            mig_gpu,
                            command,
                            vectors,
                            mode,
                            target,
                            label,
                        )
                    else:
                        sweeps["+".join([str(k).zfill(2), str(label).zfill(2)])] = run(
                            args,
                            designated,
                            mig,
                            mig_gpu,
                            command,
                            vectors,
                            mode,
                            target,
                            label,
                        )
        else:
            raise NotImplementedError

        k += args.step_size

    if args.is_power:
        if len(list(powers.keys())) > 0:
            ofile = open(output.replace("sweep", "power") + ".json", "w")
            json.dump(powers, ofile, indent=2)
            ofile.close()
    else:
        if not args.is_debug:
            if args.is_check_traffic:
                output_file = output.replace("sweep", "error") + ".json"
            else:
                output_file = output + ".json"

            ofile = open(output_file, "w")
            json.dump(sweeps, ofile, indent=2)
            ofile.close()
