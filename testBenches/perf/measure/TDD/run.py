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
import yaml

from .traffic import traffic_avg, traffic_het
from .execute import run
from .check_cell_capacity import check_cell_capacity

def run_TDD(args, sms, mig=None):

    if getattr(args, "inline_config_obj", None) is not None:
        config = args.inline_config_obj
    else:
        with open(args.config) as ifile:
            config = json.load(ifile)

    if getattr(args, "inline_uc_obj", None) is not None:
        uc = args.inline_uc_obj
    else:
        with open(args.uc) as ifile:
            uc = json.load(ifile)

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

            if len(args.target) == 1:

                buffer_target = args.target[0]

                if buffer_target.isnumeric():
                    target = []
                    target.append(np.min([int(buffer_target), sms]))
                    if(int(buffer_target) > sms):
                        print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                    
                    output = buffer_target.zfill(3) + "_" + output
                else:
                    ifile = open(buffer_target, "r")
                    data_targets = json.load(ifile)
                    ifile.close()

            else:

                target = []

                for buffer_target in args.target:

                    if buffer_target.isnumeric():
                        target.append(np.min([int(buffer_target), sms]))
                        if(int(buffer_target) > sms):
                            print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                    else:
                        raise ValueError

                buffer = [x.zfill(3) for x in args.target]
                output = "_".join(buffer) + "_" + output
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

        # only enable MPS if not running in green contexts mode or if it was explicitly enabled; terminate otherwise.
        if not args.is_use_green_contexts:
            system = " ".join([system, "nvidia-cuda-mps-control -d"])
        elif args.is_enable_mps_for_green_contexts:
            system = " ".join([system, "nvidia-cuda-mps-control -d"])
        else:
            # If there is no MPS running to terminate, the following command will show an informative "Cannot find MPS control daemon process" message.
            system = f"echo quit | CUDA_VISIBLE_DEVICES={args.gpu} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control"
            system = " ".join([system])

        if args.is_test:
            if args.debug_mode not in ["ncu"]:
                print(system)

        if args.debug_mode not in ["ncu"]:
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

    def _load_mac_slot_and_light_weight(yaml_path, pattern):
        """Load mac_slot_config (MAC run per slot) and cumac_light_weight_flag from config YAML for compare.py."""
        mac_slot_config = None
        cumac_light_weight_flag = None
        if not yaml_path or not pattern:
            return mac_slot_config, cumac_light_weight_flag
        paths_to_try = [yaml_path]
        if os.path.isabs(yaml_path) and not os.path.isfile(yaml_path):
            paths_to_try.append(os.path.join(os.getcwd(), os.path.basename(yaml_path)))
        for p in paths_to_try:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    ycfg = yaml.safe_load(f) or {}
                cfg = ycfg.get("config") if isinstance(ycfg.get("config"), dict) else {}
                tdd_slot = cfg.get("tdd_slot_config") or {}
                slot_cfg = tdd_slot.get(pattern) if isinstance(tdd_slot, dict) else {}
                if isinstance(slot_cfg.get("MAC"), list):
                    mac_slot_config = slot_cfg["MAC"]
                cumac_opts = cfg.get("cumac_options") or {}
                if isinstance(cumac_opts.get("cumac_light_weight_flag"), list):
                    cumac_light_weight_flag = cumac_opts["cumac_light_weight_flag"]
                if mac_slot_config is not None or cumac_light_weight_flag is not None:
                    break
            except (yaml.YAMLError, FileNotFoundError, TypeError, OSError):
                continue
        return mac_slot_config, cumac_light_weight_flag

    if args.is_power:
        powers['testConfig'] = _test_config_for_save()
        powers['testConfig']['smAllocation'] = str(target)
    else:
        sweeps['testConfig'] = _test_config_for_save()
        sweeps['testConfig']['smAllocation'] = str(target)

    yaml_path = getattr(args, "yaml", None)
    pattern = getattr(args, "pattern", None)
    mac_slot_config, cumac_light_weight_flag = _load_mac_slot_and_light_weight(yaml_path, pattern)
    if mac_slot_config is not None:
        if args.is_power:
            powers['testConfig']['mac_slot_config'] = mac_slot_config
        else:
            sweeps['testConfig']['mac_slot_config'] = mac_slot_config
    if cumac_light_weight_flag is not None:
        if args.is_power:
            powers['testConfig']['cumac_light_weight_flag'] = cumac_light_weight_flag
        else:
            sweeps['testConfig']['cumac_light_weight_flag'] = cumac_light_weight_flag

    def _ensure_mac_cumac_in_test_config(tc):
        """If testConfig has yaml and pattern but missing MAC/cumac keys, try loading from YAML (e.g. for phase3 or re-saves)."""
        if not isinstance(tc, dict) or (tc.get("mac_slot_config") is not None and tc.get("cumac_light_weight_flag") is not None):
            return
        yp = tc.get("yaml") or tc.get("config_yaml")
        pat = tc.get("pattern")
        if not yp or not pat:
            return
        msc, clw = _load_mac_slot_and_light_weight(yp, pat)
        if msc is not None:
            tc["mac_slot_config"] = msc
        if clw is not None:
            tc["cumac_light_weight_flag"] = clw
    
    if args.seed is not None:
        np.random.seed(args.seed)

    command = os.path.join(args.cfld, "cubb_gpu_test_bench/cubb_gpu_test_bench")

    while k <= args.cap:

        if mig is None:
            vectors = os.path.join(os.getcwd(), "vectors-" + str(k).zfill(2) + ".yaml")
        else:
            vectors = os.path.join(
                os.getcwd(), "vectors-" + mig_gpu + "-" + str(k).zfill(2) + ".yaml"
            )

        if "_het_" in args.uc:

            if not args.is_no_mps and data_targets is not None:

                key = str(k).zfill(2)
                buffer_target = data_targets.get(key, None)

                if buffer_target is None:
                    k += 1
                    continue
                else:
                    if type(buffer_target) == int:
                        target = []
                        target.append(np.min([int(buffer_target), sms]))
                        if(int(buffer_target) > sms):
                            print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                    elif type(buffer_target) == list:

                        target = []

                        for itm in buffer_target:
                            target.append(np.min([int(buffer_target), sms]))
                            if(int(buffer_target) > sms):
                                print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                                
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
                message += "(" + ",".join(list(map(str, target))) + ")"

            print(message)

            traffic_het(
                args,
                vectors,
                k,
                (testcases_dl, testcases_ul),
                (filenames_dl, filenames_ul),
            )

            if args.is_power:
                powers[str(k).zfill(2)] = run(
                    args, mig, mig_gpu, command, vectors, mode, target, k, k
                )
            else:
                sweeps[str(k).zfill(2)] = run(
                    args, mig, mig_gpu, command, vectors, mode, target, k, k
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
                    sys.exit(
                        "error: use case file exhibits an unexpected structure (PDSCH)"
                    )
                else:
                    uc_dl = uc_dl[0]

                uc_ul = [x for x in uc_keys if "PUSCH" in x]

                if len(uc_ul) != 1:
                    sys.exit(
                        "error: use case file exhibits an unexpected structure (PUSCH)"
                    )
                else:
                    uc_ul = uc_ul[0]

                testcases_dl = interval[subcase][uc_dl]
                testcases_ul = interval[subcase][uc_ul]

                filenames_dl = config[uc_dl]
                filenames_ul = config[uc_ul]

                testcases_dlbf = None
                testcases_ulbf = None
                filenames_dlbf = None
                filenames_ulbf = None
                testcases_sr = None
                filenames_sr = None
                testcases_ra = None
                filenames_ra = None
                testcases_cdl = None
                filenames_cdl = None
                testcases_cul = None
                filenames_cul = None
                testcases_ssb = None
                filenames_ssb = None
                testcases_cr = None
                filenames_cr = None
                testcases_mac = None
                filenames_mac = None
                testcases_mac2 = None
                filenames_mac2 = None

                if args.is_rec_bf:
                    uc_dlbf = [x for x in uc_keys if "DLBFW" in x]
                    if len(uc_dlbf) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (DL BFW)"
                        )
                    else:
                        uc_dlbf = uc_dlbf[0]

                    testcases_dlbf = interval[subcase][uc_dlbf]
                    filenames_dlbf = config[uc_dlbf]

                    uc_ulbf = [x for x in uc_keys if "ULBFW" in x]
                    if len(uc_ulbf) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (UL BFW)"
                        )
                    else:
                        uc_ulbf = uc_ulbf[0]

                    testcases_ulbf = interval[subcase][uc_ulbf]
                    filenames_ulbf = config[uc_ulbf]
                    
                    uc_sr = [x for x in uc_keys if "SRS" in x]
                    if len(uc_sr) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (SRS)"
                        )
                    else:
                        uc_sr = uc_sr[0]

                    testcases_sr = interval[subcase][uc_sr]
                    filenames_sr = config[uc_sr]

                if args.is_prach:
                    uc_ra = [x for x in uc_keys if "PRACH" in x]
                    if len(uc_ra) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (PRACH)"
                        )
                    else:
                        uc_ra = uc_ra[0]

                    testcases_ra = interval[subcase][uc_ra]
                    filenames_ra = config[uc_ra]

                if args.is_pdcch:
                    uc_cdl = [x for x in uc_keys if "PDCCH" in x]
                    if len(uc_cdl) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (PDCCH)"
                        )
                    else:
                        uc_cdl = uc_cdl[0]

                    testcases_cdl = interval[subcase][uc_cdl]
                    filenames_cdl = config[uc_cdl]

                if args.is_ssb:
                    uc_ssb = [x for x in uc_keys if "SSB" in x]
                    if len(uc_ssb) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (SSB)"
                        )
                    else:
                        uc_ssb = uc_ssb[0]

                    testcases_ssb = interval[subcase][uc_ssb]
                    filenames_ssb = config[uc_ssb]

                if args.is_csirs:
                    uc_cr = [x for x in uc_keys if "CSIRS" in x]
                    if len(uc_cr) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (CSIRS)"
                        )
                    else:
                        uc_cr = uc_cr[0]

                    testcases_cr = interval[subcase][uc_cr]
                    filenames_cr = config[uc_cr]

                if args.is_pucch:
                    uc_cul = [x for x in uc_keys if "PUCCH" in x]
                    if len(uc_cul) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (PUCCH)"
                        )
                    else:
                        uc_cul = uc_cul[0]
                    testcases_cul = interval[subcase][uc_cul]
                    filenames_cul = config[uc_cul]

                if args.is_mac:
                    uc_mac = [x for x in uc_keys if "MAC" == x[-3:]] # 'FXX - MAC'
                    if len(uc_mac) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (MAC)"
                        )
                    else:
                        uc_mac = uc_mac[0]

                    testcases_mac = interval[subcase][uc_mac]
                    filenames_mac = config[uc_mac]

                if args.mac2 > 0:
                    uc_mac2 = [x for x in uc_keys if "MAC2" in x]
                    if len(uc_mac2) != 1:
                        sys.exit(
                            "error: use case file exhibits an unexpected structure (MAC2)"
                        )
                    else:
                        uc_mac2 = uc_mac2[0]

                    testcases_mac2 = interval[subcase][uc_mac2]
                    filenames_mac2 = config[uc_mac2]

                label = int(subcase.replace("Average: ", ""))

                if not args.is_no_mps and data_targets is not None:

                    key = "+".join([str(k).zfill(2), str(label).zfill(2)])

                    buffer_target = data_targets.get(key, None)

                    if buffer_target is None:
                        k += 1
                        continue
                    else:
                        if type(buffer_target) == int:
                            target = []
                            target.append(np.min([int(buffer_target), sms]))
                            if(int(buffer_target) > sms):
                                print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                        elif type(buffer_target) == list:

                            target = []

                            for itm in buffer_target:
                                target.append(np.min([int(buffer_target), sms]))
                                if(int(buffer_target) > sms):
                                    print(f"Warning: SM target ({buffer_target}) capped by maxSmCount ({sms}) on {args.gpuName}")
                                    
                message = "Number of active cells: " + str(k) + "+" + str(label)

                if target is not None:
                    message += "(" + ",".join(list(map(str, target))) + ")"

                print(message)

                if args.pattern == "dddsu" and args.is_mac == True:
                    sys.exit(
                        "error: cuMAC run with dddsu pattern not supported"
                    )
                elif args.pattern == "dddsu" and args.is_mac == True:
                    sys.exit(
                        "error: cuMAC2 run with dddsu pattern not supported"
                    )
                else: # no need to include mac and mac2
                    traffic_avg(
                        args,
                        vectors,
                        (
                            testcases_dl,
                            testcases_ul,
                            testcases_dlbf,
                            testcases_ulbf,
                            testcases_sr,
                            testcases_ra,
                            testcases_cdl,
                            testcases_cul,
                            testcases_ssb,
                            testcases_cr,
                            testcases_mac,
                            testcases_mac2,
                        ),
                        (
                            filenames_dl,
                            filenames_ul,
                            filenames_dlbf,
                            filenames_ulbf,
                            filenames_sr,
                            filenames_ra,
                            filenames_cdl,
                            filenames_cul,
                            filenames_ssb,
                            filenames_cr,
                            filenames_mac,
                            filenames_mac2,
                        ),
                    )

                if args.is_power:
                    powers["+".join([str(k).zfill(2), str(label).zfill(2)])] = run(
                        args,
                        mig,
                        mig_gpu,
                        command,
                        vectors,
                        mode,
                        target,
                        k,
                        len(testcases_dl),
                    )
                else:
                    sweeps["+".join([str(k).zfill(2), str(label).zfill(2)])] = run(
                        args,
                        mig,
                        mig_gpu,
                        command,
                        vectors,
                        mode,
                        target,
                        k,
                        len(testcases_dl),
                    )
        else:
            raise NotImplementedError

        k += args.step_size

    if args.is_power:
        if len(list(powers.keys())) > 0:
            _ensure_mac_cumac_in_test_config(powers.get("testConfig"))
            ofile = open(output.replace("sweep", "power") + ".json", "w")
            json.dump(powers, ofile, indent=2)
            ofile.close()
    else:
        if not args.is_debug:
            if args.is_check_traffic:
                output_file = output.replace("sweep", "error") + ".json"
            else:
                output_file = output + ".json"
            
            # auto dectect cell capacity for F08, F09, F14 of TDD long pattern (dddsuudddd)
            # the max cell count that all channels pass (within latency threshold) will be the cell capacity
            # a warning of unknown cell capacity will be given if all tested cell counts pass or none passes
            if not args.is_test:
                if args.is_ref_check:
                    print("enable_ref_check is on, skip cell capacity check")
                else:
                    check_cell_capacity(sweeps)
            
                _ensure_mac_cumac_in_test_config(sweeps.get("testConfig"))
                ofile = open(output_file, "w")
                json.dump(sweeps, ofile, indent=2)
                ofile.close()
