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

import argparse
from importlib import util
import subprocess
import json
import re
from pathlib import Path

from .FDD.properties import avg_subs, het_subs


def arguments():

    base = argparse.ArgumentParser()
    base.add_argument(
        "--yaml",
        type=str,
        dest="yaml",
        help=(
            "YAML file providing defaults for measure.py. "
            "Keys should match long option names without the leading '--'. "
            "(e.g., cuphy, vectors, config, uc, freq, cap, start, iterations, slots, power, "
            "target: [..], graph: true). "
            "You may also provide 'config_inline' (a dict matching the JSON testcases format); "
            "it will be written next to the YAML and used as --config. "
            "Explicit CLI flags override YAML."
        ),
        required=False,
    )
    base.add_argument(
        "--cuphy",
        type=str,
        dest="cfld",
        help="Specifies the folder where cuPHY has been built",
        required=True,
    )
    base.add_argument(
        "--vectors",
        type=str,
        dest="vfld",
        help="Specifies the folder for the test vectors",
        required=True,
    )
    base.add_argument(
        "--config",
        type=str,
        dest="config",
        help="Specifies the file contaning the test cases list",
        required=False,
    )
    base.add_argument(
        "--uc",
        type=str,
        dest="uc",
        help="Specifies the file contaning the use case config",
        required=False,
    )
    base.add_argument(
        "--target",
        type=str,
        nargs="+",
        dest="target",
        help="Specifies the SMs used by each sub-CTX",
    )
    base.add_argument(
        "--delay",
        type=int,
        dest="delay",
        default=1000,
        help="Specifies the duration of the delay kernel",
    )
    base.add_argument(
        "--gpu",
        type=int,
        dest="gpu",
        default=0,
        help="Specifies on which GPU to run the measurements",
    )
    base.add_argument(
        "--freq",
        type=int,
        dest="freq",
        help="Specifies the frequency at which the GPU will be set for the measurements",
        required=True,
    )
    base.add_argument(
        "--graph",
        action="store_true",
        dest="is_graph",
        default=False,
        help="Specifies whether to use graphs rather than streams for FDD use cases",
    )
    base.add_argument(
        "--start",
        type=int,
        dest="start",
        default=1,
        help="Specifies the minimum numbers of cells to try",
    )
    base.add_argument(
        "--cap",
        type=int,
        dest="cap",
        help="Specifies the maximum numbers of cells to try",
        required=True,
    )
    base.add_argument(
        "--step_size",
        type=int,
        dest="step_size",
        default=1,
        help="Specifies step size when sweeping the number of cells to try",
        required=False,
    )
    base.add_argument(
        "--iterations",
        type=int,
        dest="iterations",
        default=1,
        help="Specifies number of iterations to use in averaging latency results",
    )
    base.add_argument(
        "--slots",
        type=int,
        dest="sweeps",
        default=1,
        help="Specifies number of sweep iterations",
    )
    base.add_argument(
        "--power",
        type=int,
        dest="power",
        help="Specifies the maximum power draw for the GPU used for the measurements",
    )
    base.add_argument(
        "--fdd_subs",
        type=int,
        dest="subs",
        help="Specifies the number of sub-CTXs to use",
    )
    base.add_argument(
        "--force",
        type=int,
        dest="force",
        help="Specifies the number of connections to use",
    )
    base.add_argument(
        "--priority",
        action="store_true",
        dest="is_priority",
        help="Specifies whether PDSCH has higher priority over PUSCH",
    )
    base.add_argument(
        "--mig",
        type=int,
        dest="mig",
        help="Specifies the MIG ID to use to create the GIs",
    )
    base.add_argument(
        "--mig_instances",
        type=int,
        dest="mig_instances",
        default=1,
        help="Specifies the number of MIG GPU-instances to run with",
    )
    base.add_argument(
        "--disable_mps",
        action="store_true",
        dest="is_no_mps",
        help="Not supported any longer",
    )
    base.add_argument(
        "--seed",
        type=int,
        dest="seed",
        help="Specifies the seed to use in the simulations",
    )
    base.add_argument(
        "--measure_power",
        action="store_true",
        dest="is_power",
        default=False,
        help="Specifies whether to use the bench for measuring power rather than latency",
    )
    base.add_argument(
        "--test",
        action="store_true",
        dest="is_test",
        help="Specifies whether to enable test mode",
    )
    base.add_argument(
        "--debug",
        action="store_true",
        dest="is_debug",
        help="Specifies whether to enable debug mode",
    )
    base.add_argument(
        "--enable_nvprof",
        action="store_true",
        dest="is_enable_nvprof",
        help="Enable cudaProfilerStart/Stop in cubb_gpu_test_bench around each pattern run",
    )
    base.add_argument(
        "--enable_sqlite",
        action="store_true",
        dest="is_enable_sqlite",
        help="Enable --export sqlite in nsys profile command",
    )
    base.add_argument(
        "--debug_mode",
        type=str,
        dest="debug_mode",
        choices=["cta", "triage", "nsys", "ncu", "incu", "nsys_simple"],
        default="cta",
        help="Specifies which debug mode to use",
    )
    base.add_argument(
        "--rec_bf",
        action="store_true",
        dest="is_rec_bf",
        help="Specifies whether the use case involves reciprocal beamforming",
    )
    base.add_argument(
        "--prach",
        action="store_true",
        dest="is_prach",
        help="Specifies whether the use case involves PRACH",
    )
    base.add_argument(
        "--prach_isolate",
        action="store_true",
        dest="is_isolated_prach",
        help="Specifies whether PRACH needs to run on its own sub-CTX",
    )
    base.add_argument(
        "--ssb",
        action="store_true",
        dest="is_ssb",
        help="Specifies whether the use case involves SSB",
    )
    base.add_argument(
        "--csirs",
        action="store_true",
        dest="is_csirs",
        help="Specifies whether the use case involves CSI-RS",
    )
    base.add_argument(
        "--prach_tgt",
        type=int,
        dest="prach_tgt",
        help="Internal",
    )
    base.add_argument(
        "--pdcch",
        action="store_true",
        dest="is_pdcch",
        help="Specifies whether the use case involves PDCCH",
    )
    base.add_argument(
        "--pdcch_isolate",
        action="store_true",
        dest="is_isolated_pdcch",
        help="Specifies whether PDCCH needs to run on its own sub-CTX",
    )
    base.add_argument(
        "--pucch",
        action="store_true",
        dest="is_pucch",
        help="Specifies whether the use case involves PUCCH",
    )
    base.add_argument(
        "--pucch_isolate",
        action="store_true",
        dest="is_isolated_pucch",
        help="Specifies whether PUCCH needs to run on its own sub-CTX",
    )
    base.add_argument(
        "--unsafe",
        action="store_true",
        dest="is_unsafe",
        help="Specifies whether to measure power without timeouts",
    )
    base.add_argument(
        "--groups_dl",
        action="store_true",
        dest="is_groups_pdsch",
        help="Specifies whether to use cell groups for PDSCH",
    )
    base.add_argument(
        "--pack_pdsch",
        action="store_true",
        dest="is_pack_pdsch",
        help="Specifies whether to use packed cell groups for PDSCH",
    )
    base.add_argument(
        "--groups_pusch",
        action="store_true",
        dest="is_groups_pusch",
        help="Specifies whether to use cell groups for PUSCH",
    )
    base.add_argument(
        "--no_pusch",
        action="store_true",
        dest="is_no_pusch",
        help="Specifies whether to simulate PUSCH",
    )
    base.add_argument(
        "--no_pdsch",
        action="store_true",
        dest="is_no_pdsch",
        help="Specifies whether to simulate PDSCH",
    )
    base.add_argument(
        "--check_traffic",
        action="store_true",
        dest="is_check_traffic",
        help="Specifies whether to check for functional error in the traffic",
    )
    base.add_argument(
        "--numa", type=int, dest="numa", help="Specifies the NUMA node to use"
    )
    base.add_argument(
        "--2cb_per_sm",
        action="store_true",
        dest="is_2_cb_per_sm",
        help="Specifies whether to enable 2CB/SM on GA100",
    )
    base.add_argument(
        "--tdd_pattern",
        type=str,
        dest="pattern",
        choices=["dddsu", "dddsuudddd", "dddsuudddd_8slot", "dddsuudddd_mMIMO"],
        default="dddsu",
        help="Specifies the TDD pattern to run",
    )
    base.add_argument(
        "--save_buffers",
        action="store_true",
        dest="is_save_buffers",
        help="Specifies whether to save intermediate buffers (normally erased)",
    )
    base.add_argument(
        "--pusch_cascaded",
        action="store_true",
        dest="is_pusch_cascaded",
        help="Specifies whether for, DDDSUUDDDD, the second UL slot needs to be processed after the first",
    )
    base.add_argument("--triage_start", type=int, dest="triage_start", help="Internal")
    base.add_argument("--triage_end", type=int, dest="triage_end", help="Internal")
    base.add_argument(
        "--triage_sample", type=int, default=1024, dest="triage_sample", help="Internal"
    )

    base.add_argument(
        "--ldpc_parallel",
        action="store_true",
        dest="is_ldpc_parallel",
        help="Specifies whether for the PUSCH LDPC decoder runs the TB in parallel rather than serially",
    )

    base.add_argument(
        "--srs_isolate",
        action="store_true",
        dest="is_srs_isolate",
        help="Specifies whether SRS needs to run on its own sub-CTX",
    )

    base.add_argument(
        "--ssb_isolate",
        action="store_true",
        dest="is_ssb_isolate",
        help="Specifies whether SSB needs to run on its own sub-CTX",
    )
    
    base.add_argument(
        "--mac",
        action="store_true",
        dest="is_mac",
        help="Specifies whether MAC workload runs, cuMAC runs in a dedicated sub-CTX",
    )

    base.add_argument(
        "--mac2",
        type=int,
        dest="mac2",
        help="Specifies whether second MAC workload runs with fixed number of cells, second cuMAC runs in a dedicated sub-CTX",
        default=0,
        required=False,
    )
    
    base.add_argument(
        "--mac_timer",
        action="store_true",
        dest="is_mac_timer",
        help="Specifies whether MAC workload use its internal timer",
    )
        
    base.add_argument(
        "--GH200",
        action="store_true",
        dest="is_GH200", # true to overwrite; false will auto detect
        help="Specifies whether the hardware is GH200; automatically set if not overwrite by input",
    )
    
    base.add_argument(
        "--use_green_contexts",
        action="store_true",
        dest="is_use_green_contexts",
        help="Use green contexts instead of MPS",
    )

    base.add_argument(
        "--enable_mps_for_green_contexts",
        action="store_true",
        dest="is_enable_mps_for_green_contexts",
        help="Enable MPS for green contexts; if not set, MPS will be terminated",
    )

    base.add_argument(
        "--pusch_subslot_proc",
        type=str,
        dest="pusch_subslot_proc",
        choices=["0", "1", "00", "01", "10", "11"],
        default="0",
        help="Specifies the subslot processing setting for PUSCH",
    )

    base.add_argument(
        "--enable_ref_check",
        action="store_true",
        dest="is_ref_check",
        help=(
            "Enable reference checks for all channels "
            "(appends -k --k -b --c PUSCH,PDSCH,PDCCH,PUCCH,SSB,DLBFW,ULBFW,CSIRS,PRACH,SRS). "
            "WARNING: do not use during latency or power measurements -- "
            "the extra checking work will skew the timeline."
        ),
    )

    # Optional YAML config support (keeps legacy CLI fully working).
    #
    # Semantics:
    # - YAML provides defaults (expanded into argv)
    # - Explicit CLI flags override YAML (because CLI args are appended after YAML args)
    # - For store_true flags, YAML can enable a flag, but CLI cannot explicitly disable it
    #   (same behavior as legacy CLI).
    #
    # Note: Some YAMLs (e.g., phase3_test_config.yaml) include additional, non-CLI keys
    # (vector_files, latency_budget, etc.). We ignore any YAML keys that do not map to a
    # known CLI flag to avoid argparse "unrecognized arguments" errors.
    yaml_parser = argparse.ArgumentParser(add_help=False)
    yaml_parser.add_argument("--yaml", type=str, dest="yaml")
    yaml_args, remaining_argv = yaml_parser.parse_known_args()

    injected_argv: list[str] = []
    generated_config_obj: dict | None = None
    generated_uc_obj: dict | None = None
    yaml_vector_files_for_save: dict | None = None
    if yaml_args.yaml:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "YAML config requested via --yaml but PyYAML is not available. "
                "Install it with: pip install pyyaml"
            ) from e

        cfg_path = Path(yaml_args.yaml).expanduser()
        with cfg_path.open("r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        if not isinstance(raw_cfg, dict):
            raise ValueError(
                f"--yaml must contain a YAML mapping/dict, got: {type(raw_cfg).__name__}"
            )

        # Allow either top-level mapping, or nested "measure:" mapping.
        root: dict = raw_cfg.get("measure") if isinstance(raw_cfg.get("measure"), dict) else raw_cfg  # type: ignore[assignment]

        # phase3 YAMLs commonly place actual CLI defaults under "config:"; keep those,
        # but also consider other top-level keys (they will be filtered to known CLI flags).
        cfg: dict = {}
        if isinstance(root.get("config"), dict):
            cfg.update({k: v for k, v in root.items() if k != "config"})
            cfg.update(root["config"])  # config section wins
        else:
            cfg.update(root)

        # Friendly YAML aliases -> actual CLI option names (without leading '--').
        yaml_aliases: dict[str, str] = {
            "testbench_folder": "cuphy",
            "testvectors_folder": "vectors",
            "tdd_pattern": "tdd_pattern",
            "export_sqlite": "enable_sqlite",
        }

        # Optional: inline JSON configs (written next to the YAML file).
        # - config_inline -> --config
        # - uc_inline     -> --uc
        if "config_inline" in cfg and "--config" not in remaining_argv:
            inline = cfg.get("config_inline")
            if not isinstance(inline, dict):
                raise ValueError(
                    "'config_inline' must be a mapping/dict (same structure as the JSON file)"
                )
            out_json = cfg_path.with_suffix(".testcases.json")
            out_json.write_text(
                json.dumps(inline, indent=4, sort_keys=False) + "\n", encoding="utf-8"
            )
            cfg["config"] = str(out_json)
            cfg.pop("config_inline", None)

        if "uc_inline" in cfg and "--uc" not in remaining_argv:
            inline = cfg.get("uc_inline")
            if not isinstance(inline, dict):
                raise ValueError(
                    "'uc_inline' must be a mapping/dict (same structure as the JSON file)"
                )
            out_json = cfg_path.with_suffix(".uc.json")
            out_json.write_text(
                json.dumps(inline, indent=4, sort_keys=False) + "\n", encoding="utf-8"
            )
            cfg["uc"] = str(out_json)
            cfg.pop("uc_inline", None)

        # Optional: derive config/uc JSONs from YAML-only TV lists (phase3 style).
        # This allows running with just --yaml when the YAML provides:
        #   config:
        #     vector_files: { PDSCH: [..], PUSCH: [..], ... }
        vector_files = cfg.get("vector_files")
        if isinstance(vector_files, dict):
            # Infer usecase (F01/F08/F09/F14) if not explicitly provided.
            usecase = cfg.get("usecase")
            if not usecase:
                # Best-effort: infer from any filename that contains e.g. "F08".
                for v in vector_files.values():
                    items = v if isinstance(v, list) else [v]
                    for item in items:
                        if not isinstance(item, str):
                            continue
                        # Match e.g. "F08" even in tokens like "TV_cumac_F08-...".
                        m = re.search(r"F\d\d(?!\d)", item)
                        if m:
                            usecase = m.group(0)
                            break
                    if usecase:
                        break

            if not usecase or not isinstance(usecase, str):
                raise ValueError(
                    "YAML provides 'vector_files' but no 'usecase'. Add e.g.:\n"
                    "  config:\n"
                    "    usecase: F08\n"
                    "or include an Fxx marker in a filename (e.g. MAC TV)."
                )

            is_tdd = "tdd_pattern" in cfg
            duplex = "TDD" if is_tdd else "FDD"
            testcase_id = f"{usecase}-PP-00"

            # Generate config JSON (example_cubb_* format) if missing.
            if "config" not in cfg and "--config" not in remaining_argv:
                config_inline: dict[str, dict[str, object]] = {}
                for ch, tvs in vector_files.items():
                    if tvs is None:
                        continue
                    if not isinstance(ch, str):
                        continue

                    key = f"{usecase} - {ch}"

                    # Normalize to either string or list[str]
                    if isinstance(tvs, str):
                        tv_obj: object = tvs
                    elif isinstance(tvs, list):
                        tv_obj = [str(x) for x in tvs]
                    else:
                        continue

                    # Match existing convention: MAC uses a single string when only one TV is provided.
                    if ch in {"MAC", "MAC2"} and isinstance(tv_obj, list) and len(tv_obj) == 1:
                        tv_obj = tv_obj[0]

                    config_inline[key] = {testcase_id: tv_obj}

                generated_config_obj = config_inline
                # Keep a synthetic name for downstream metadata/usecase checks.
                cfg["config"] = str(cfg_path.with_suffix(f".{usecase}.testcases.json"))

            # Generate UC JSON (uc_avg_* format) if missing.
            if "uc" not in cfg and "--uc" not in remaining_argv:
                uc_inline: dict[str, dict[str, dict[str, list[str]]]] = {}
                channels = [f"{usecase} - {ch}" for ch in vector_files.keys() if isinstance(ch, str)]

                # Build Peak: N -> Average: 0 -> channel -> [testcase_id ...]
                cap = int(cfg.get("cap", 1))
                for peak in range(1, max(cap, 1) + 1):
                    avg0: dict[str, list[str]] = {}
                    for ch_key in channels:
                        base_ch = ch_key.split(" - ", 1)[1] if " - " in ch_key else ch_key
                        if base_ch in {"MAC", "MAC2"}:
                            avg0[ch_key] = [testcase_id]
                        else:
                            avg0[ch_key] = [testcase_id] * peak
                    uc_inline[f"Peak: {peak}"] = {"Average: 0": avg0}

                generated_uc_obj = uc_inline
                # Keep a synthetic name for downstream mode checks/output naming.
                cfg["uc"] = str(cfg_path.with_name(f"uc_avg_{usecase}_{duplex}.json"))
                # Normalize for saved JSON: channel -> list of .h5 (no inline config/uc blobs).
                yaml_vector_files_for_save = {}
                for ch, tvs in vector_files.items():
                    if not isinstance(ch, str) or tvs is None:
                        continue
                    yaml_vector_files_for_save[ch] = [
                        str(x) for x in (tvs if isinstance(tvs, list) else [tvs])
                    ]

        known_flags = set(base._option_string_actions.keys())
        # YAML-only keys that are intentionally consumed by downstream modules
        # (traffic generation, capacity checks, JSON derivation), not argparse.
        non_cli_yaml_keys = {
            "vector_files",
            "usecase",
            "tdd_priorities",
            "tdd_slot_config",
            "start_delay",
            "latency_budget",
            "override_test_vectors",
        }

        ignored_yaml_keys: list[str] = []
        # Expand YAML into argv; only inject options not explicitly present in CLI argv.
        for k, v in cfg.items():
            if v is None:
                continue

            key = yaml_aliases.get(str(k), str(k))
            flag = f"--{key}"

            if flag not in known_flags:
                if str(k) not in non_cli_yaml_keys:
                    ignored_yaml_keys.append(str(k))
                continue

            if flag in remaining_argv:
                continue  # CLI overrides YAML

            if isinstance(v, bool):
                if v:
                    injected_argv.append(flag)
                continue

            if isinstance(v, (list, tuple)):
                injected_argv.append(flag)
                injected_argv.extend([str(x) for x in v])
                continue

            injected_argv.extend([flag, str(v)])

        if ignored_yaml_keys:
            # Best-effort visibility without failing the run.
            print(
                "Warning: ignoring unknown YAML keys (not CLI options): "
                + ", ".join(sorted(set(ignored_yaml_keys)))
            )

    # Parse YAML-expanded argv first, then append CLI argv (CLI wins on repeated flags).
    args = base.parse_args(injected_argv + remaining_argv)
    # Preserve YAML path for downstream modules that consume extra YAML-only sections
    # (e.g., tdd_priorities/start_delay/override_test_vectors/latency_budget).
    args.yaml = yaml_args.yaml
    # Preserve in-memory config/uc objects when vector_files is used in YAML mode.
    args.inline_config_obj = generated_config_obj
    args.inline_uc_obj = generated_uc_obj
    if yaml_vector_files_for_save is not None:
        args.vector_files = yaml_vector_files_for_save

    if args.config is None and args.inline_config_obj is None:
        base.error("Missing --config (or provide YAML vector_files to generate config in-memory)")
    if args.uc is None and args.inline_uc_obj is None:
        base.error("Missing --uc (or provide YAML vector_files to generate uc in-memory)")

    if args.start > args.cap:
        base.error(
            "The minimum number of cells to try cannot be higher than the maximum"
        )

    if args.is_no_mps:
        args.is_no_mps = False

    if "FDD" in args.uc:
        if "_avg_" in args.uc:
            if args.subs is not None:
                if args.subs not in avg_subs:
                    base.error(
                        f"At this stage, the number of sub-CTXs can only be chosen in {avg_subs}"
                    )
        else:
            if args.subs is not None:
                if args.subs not in het_subs:
                    base.error(
                        f"At this stage, the number of sub-CTXs can only be chosen in {het_subs}"
                    )

        if args.subs is not None:
            if args.subs > 1:
                if args.is_no_mps:
                    base.error(
                        "Disabling MPS is not supported from a number of sub-CTXs larger than 1"
                    )

    if "TDD" in args.uc:
        if args.subs is not None:
            base.error("The number of sub-CTXs to use cannot be set with TDD use cases")
        if "_het_" in args.uc:
            if args.pattern in {"dddsuudddd", "dddsuudddd_8slot", "dddsuudddd_mMIMO"}:
                base.error(
                    "extended TDD pattern is not supported for heterogeneous traffic"
                )

    # pattern_len is the number of slots per pattern
    if args.pattern == "dddsu":
        args.pattern_len = 4
    elif args.pattern == "dddsuudddd":
        args.pattern_len = 10
    elif args.pattern == "dddsuudddd_8slot":
        args.pattern_len = 8
    elif args.pattern == "dddsuudddd_mMIMO":
        args.pattern_len = 15
    else:
        raise NotImplementedError

    if args.is_rec_bf:
        if (
            "TDD" not in args.uc
            or "_avg_" not in args.uc
            or ("F14" not in args.uc and "F09" not in args.uc)
        ):
            base.error(
                "Reciprocal beamforming can only be activated with the F09/F14 use cases with avg. cells"
            )

    if args.is_pdcch:
        if "TDD" not in args.uc or "_avg_" not in args.uc or args.is_no_pdsch:
            base.error(
                "PDCCH can only be activated with avg. cells, when PDSCH is also present"
            )

    if args.is_pucch:
        if "TDD" not in args.uc or "_avg_" not in args.uc or args.is_no_pusch:
            base.error(
                "PUCCH can only be activated with avg. cells, when PUSCH is also present"
            )

    if args.is_debug:
        if args.mig is not None:
            if args.mig_instances > 1:
                base.error(
                    "Profiling with multiple CPU processes running in parallel is not supported"
                )

        if "FDD" in args.uc:
            buffer = util.find_spec("measure.FDD.configure_debug")
            if buffer is None:
                base.error(
                    "This is a released version of the performance scripts, and debug mode cannot be enabled."
                )

        else:
            buffer = util.find_spec("measure.TDD.configure_debug")
            if buffer is None:
                base.error(
                    "This is a released version of the performance scripts, and debug mode cannot be enabled."
                )

    if args.is_debug:
        if args.debug_mode == "triage":
            if (
                args.triage_start is None
                or args.triage_end is None
                or args.triage_sample is None
            ):
                base.error("Triage mode is not supported without all of its parameters")

    if args.is_power:
        if args.is_debug:
            if args.debug_mode not in ["nsys"]:
                if "TDD" not in args.uc:
                    base.error(
                        "For power measurements, only trace with Nsight System is supported, and exclusively for F14 use case"
                    )
    else:
        if args.is_unsafe:
            print("Warning: --unsafe is only applicable for power measurements")

    if args.is_ref_check:
        if args.is_power:
            print(
                "WARNING: --enable_ref_check is enabled with power measurement. "
                "Reference checks will skew the timeline and power results. "
                "Some cuPHY channels may not support ref check with power mode and may crash."
            )
        elif not args.is_test:
            print(
                "WARNING: --enable_ref_check is enabled. "
                "Reference checks will skew latency results; do not use for latency measurements."
            )

    if args.is_rec_bf or args.is_prach:
        if "TDD" not in args.uc or "_avg_" not in args.uc:
            base.error(
                "PRACH and/or reciprocal beamforming can only be enabled for the TDD use case with the peak + avg traffic model"
            )

    if args.is_pdcch or args.is_pucch or args.is_ssb or args.is_csirs:
        if "TDD" not in args.uc or "_avg_" not in args.uc:
            base.error(
                "PDCCH, PUCCH, SSB and/or CSI-RS can only be activated for the TDD use case with the peak + avg traffic model"
            )

    selected_channels = sum(
        list(
            map(
                int,
                [
                    args.is_prach and args.is_isolated_prach,
                    args.is_pdcch and args.is_isolated_pdcch,
                    args.is_pucch and args.is_isolated_pucch,
                    not args.is_no_pdsch,
                    not args.is_no_pusch,
                    args.is_ssb and args.is_ssb_isolate,
                    args.is_srs_isolate,
                    args.is_mac,
                    args.mac2 > 0,
                ],
            )
        )
    )

    if len(args.target) > 1 and len(args.target) != selected_channels:
        base.error(
            "The number of arguments for --target does no match the isolation strategy for the selected channels"
        )

    if args.prach_tgt is not None:
        if not args.is_prach or not args.is_isolated_prach:
            base.error(
                "--prach_tgt can only be enabled with --prach and --prach_isolate"
            )

    if args.is_pack_pdsch:
        if not args.is_groups_pdsch:
            base.error("--pack_pdsch can only be used with --groups_pdsch")

    # if args.is_groups_pdsch:
    #     if args.is_rec_bf or args.is_prach or args.is_pdcch:
    #         base.error("--groups_pdsch can only be used with PDSCH-only traffic for DL")

    if not args.is_no_mps:
        if args.target is None:
            base.error("Missing MPS target")

    if args.is_no_mps:
        if args.target is not None:
            print("Warning: with MPS disabled, the provided target will be ignored")

    if args.is_pusch_cascaded:
        if "TDD" not in args.uc or "_avg_" not in args.uc or args.pattern == "dddsu":
            base.error(
                "back-to-back UL workloads can only be enabled for TDD DDDSUUDDDD links and for the peak+avg. traffic model"
            )

    # check whether its GH200
    if not args.is_GH200: # auto detect
        command = f"nvidia-smi -i {args.gpu} --query-gpu=name --format=csv,noheader"
        result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, encoding="utf-8")
        args.gpuName = result.stdout.strip()
        args.is_GH200 = ("GH200" in args.gpuName)
        if "GH200" in args.gpuName:
            print(f"Auto detect GPU enabled: GH200 detected, running on {args.gpuName}")
        else:
            print(f"Auto detect GPU enabled: GH200 not detected, running on {args.gpuName}")
    else:
        print(f"Auto detect GPU disabled: User explicitly set GH200, running on {args.gpuName}")
    
    return (base, args)
