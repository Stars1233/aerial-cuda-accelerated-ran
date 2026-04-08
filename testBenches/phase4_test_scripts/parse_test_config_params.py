#!/usr/bin/env python3

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

"""
Parse test case configuration string and generate parameters for cuBB test scripts.

This script prints export statements to stdout that can be evaluated by the shell.

Usage:
    python3 parse_test_config_params.py <test_case_string> <host_config> [options]

Examples:
    # Basic usage - use with eval or the shell wrapper
    eval $(python3 parse_test_config_params.py "F08_6C_69_MODCOMP_STT480000_1P" "CG1_R750")

    # Save to file for later sourcing
    python3 parse_test_config_params.py "F08_6C_69_MODCOMP_STT480000_1P" "CG1_R750" -o env.sh
    source env.sh

    # Use custom variable names
    eval $(python3 parse_test_config_params.py "F08_6C_69_MODCOMP_STT480000_1P" "CG1_R750" \
        --copy-test-files-params MY_COPY_PARAMS \
        --test-config-params MY_CONFIG_PARAMS)
"""

import sys
import re
import argparse
import os
import subprocess
import glob
from pathlib import Path

def load_mimo_patterns_from_shell_script() -> list[str]:
    """
    Load MIMO patterns from valid_perf_patterns.sh by sourcing it.

    Returns:
        List of MIMO pattern strings

    Raises:
        FileNotFoundError: If valid_perf_patterns.sh cannot be found
        subprocess.CalledProcessError: If sourcing the script fails
        ValueError: If mimo_patterns array is empty
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    shell_script_path = script_dir / "valid_perf_patterns.sh"

    if not shell_script_path.exists():
        raise FileNotFoundError(
            f"Cannot find valid_perf_patterns.sh at {shell_script_path}. "
            "MIMO patterns cannot be determined."
        )

    # Source the shell script and extract mimo_patterns array
    # Using printf '%s\n' to output each array element on a new line
    bash_cmd = f'source "{shell_script_path}" && printf "%s\\n" "${{mimo_patterns[@]}}"'

    try:
        result = subprocess.run(
            ['bash', '-c', bash_cmd],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            e.returncode,
            e.cmd,
            output=e.output,
            stderr=f"Failed to source {shell_script_path}: {e.stderr}"
        )

    # Parse the output (one pattern per line)
    patterns = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]

    if not patterns:
        raise ValueError(
            f"No MIMO patterns found in mimo_patterns array from {shell_script_path}"
        )

    return patterns

# Load MIMO patterns from valid_perf_patterns.sh
try:
    MIMO_PATTERNS = load_mimo_patterns_from_shell_script()
except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
    print(f"Error loading MIMO patterns: {e}", file=sys.stderr)
    sys.exit(1)

# Valid host configurations
VALID_HOST_CONFIGS = ["CG1_R750", "CG1_CG1", "GL4_R750"]

def parse_test_case_string(test_case):
    """Parse the test case string and extract all parameters."""

    # Check if it starts with F08
    if test_case.startswith("F08_") :
        params = parse_f08_test_case_string(test_case)
    elif test_case.isdigit():
        params = parse_nrsim_test_case_string(test_case)
    else:
        print(f"Error: Test case string must start with 'F08_' or be an integer NRSIM test case. Got: {test_case}", file=sys.stderr)
        sys.exit(1)

    return params

def parse_nrsim_test_case_string(test_case):
    if not test_case.isdigit():
        print(f"Error: NRSIM test case string must be an integer value. Got: {test_case}", file=sys.stderr)
        sys.exit(1)

    preset = lookup_preset(test_case)
    channel = lookup_nrsim_channel(int(test_case))

    params = {
        'pattern': test_case,
        'preset': preset,
        'channel': channel
    }

    return params

def parse_f08_test_case_string(test_case):
    if not test_case.startswith("F08_"):
        print(f"Error: F08 test case string must start with 'F08_'. Got: {test_case}", file=sys.stderr)
        sys.exit(1)

    # Remove F08_ prefix
    remaining = test_case[4:]

    # Parse cells and pattern - expecting format like "6C_69_..."
    match = re.match(r'^(\d+)C_([0-9a-z]+)(?:_(.*))?$', remaining)
    if not match:
        print(f"Error: Invalid test case format. Expected F08_<num>C_<pattern>_... Got: {test_case}", file=sys.stderr)
        sys.exit(1)

    num_cells = match.group(1)
    pattern = match.group(2)
    modifiers = match.group(3) if match.group(3) else ""

    # Parse modifiers
    params = {
        'num_cells': num_cells,
        'pattern': pattern,
        'compression': 1,  # Default BFP
        'bfp': 9,  # Default BFP value
        'stt': 0,  # Default to 0 if not specified
        'num_ports': 1,
        'ehq': 0,  # Default disabled
        'green_ctx': 0,  # Default disabled
        'work_cancel': 2,  # Default
        'pmu': 0,  # Default disabled
        'num_slots': 600000,  # Default
        'log_nic_timings': False,
        'ru_worker_tracing': False,  # Default disabled
        'reduced_logging': False,  # Default disabled
        'mumimo': 1 if pattern in MIMO_PATTERNS else 0,
        'cupti': False,  # Default disabled
        'preset': 'perf'
    }

    # Split modifiers by underscore and process each
    if modifiers:
        modifier_parts = modifiers.split('_')
        i = 0
        while i < len(modifier_parts):
            mod = modifier_parts[i]

            if mod == "MODCOMP":
                params['compression'] = 4
            elif mod == "BFP9":
                params['compression'] = 1
                params['bfp'] = 9
            elif mod == "BFP14":
                params['compression'] = 1
                params['bfp'] = 14
            elif mod.startswith("STT") and len(mod) > 3:
                params['stt'] = int(mod[3:])
            elif mod == "1P":
                params['num_ports'] = 1
            elif mod == "2P":
                params['num_ports'] = 2
            elif mod == "EH":
                params['ehq'] = 1
            elif mod == "GC":
                params['green_ctx'] = 1
            elif mod.startswith("WC") and len(mod) > 2:
                try:
                    params['work_cancel'] = int(mod[2:])
                except ValueError:
                    print(f"Error: Invalid work cancel value in '{mod}'", file=sys.stderr)
                    sys.exit(1)
            elif mod.startswith("PMU") and len(mod) > 3:
                try:
                    params['pmu'] = int(mod[3:])
                except ValueError:
                    print(f"Error: Invalid PMU value in '{mod}'", file=sys.stderr)
                    sys.exit(1)
            elif mod.startswith("NS") and len(mod) > 2:
                try:
                    params['num_slots'] = int(mod[2:])
                except ValueError:
                    print(f"Error: Invalid number of slots in '{mod}'", file=sys.stderr)
                    sys.exit(1)
            elif mod == "NICD":
                params['log_nic_timings'] = True
            elif mod == "RUWT":
                params['ru_worker_tracing'] = True
            elif mod == "NOPOST":
                params['reduced_logging'] = True
            elif mod == "CUPTI":
                params['cupti'] = True
            elif mod.startswith("RUN") and len(mod) > 3 and mod[3:].isdigit():
                # Run/instance identifier for CI/CD parallelization - informational only
                params['run_instance'] = int(mod[3:])
            elif mod == "":
                # Empty modifier, skip
                pass
            else:
                # Unknown modifier, fail with error
                print(f"Error: Unknown modifier '{mod}' in test case string", file=sys.stderr)
                print(f"Valid modifiers: BFP9, BFP14, MODCOMP, STT<number>, 1P, 2P, EH, GC, WC<number>, PMU<number>, NS<number>, NICD, RUWT, NOPOST, CUPTI, RUN<number>", file=sys.stderr)
                sys.exit(1)

            i += 1

    return params

def lookup_preset(test_case):
    """Returns the build preset to use based on the test_case name"""
    default_preset = "perf"

    if test_case.startswith("F08_"):
        return "perf"

    try:
        test_number = int(test_case)
    except ValueError:
        print(f"Warning: unable to parse test case '{test_case}'. Using default preset: {default_preset}")
        return default_preset

    channel = lookup_nrsim_channel(test_number)

    if channel is None:
        print(f"Warning: unable to match NRSIM to channel for '{test_case}'. Using default preset: {default_preset}")
        return default_preset

    nrsim_10_04_32dl = [90156, 90157, 90158]
    channels_10_04 = ["SRS", "MIX", "mSlot_mCell"]
    channels_10_02 = ["PBCH", "PDCCH_DL", "PDSCH", "CSI_RS", "PRACH", "PUCCH", "PUSCH", "BFW"]

    # Exceptions to the channel mapping are checked first
    if int(test_case) in nrsim_10_04_32dl:
        return "10_04_32dl"
    elif channel in channels_10_04:
        return "10_04"
    elif channel in channels_10_02:
        return "10_02"
    else:
        print(f"Warning: unable to match NRSIM '{test_case}' to a predetermined preset. Using default preset: {default_preset}")
        return default_preset

def lookup_nrsim_channel(test_case_number):
    """Checks test_number for NRSIM and maps to the channel type"""
    channel_ranges = {
        "MIX":         {"min": 0,     "max": 999},
        "PBCH":        {"min": 1000,  "max": 1999},
        "PDCCH_DL":    {"min": 2000,  "max": 2999},
        "PDSCH":       {"min": 3000,  "max": 3999},
        "CSI_RS":      {"min": 4000,  "max": 4999},
        "PRACH":       {"min": 5000,  "max": 5999},
        "PUCCH":       {"min": 6000,  "max": 6999},
        "PUSCH":       {"min": 7000,  "max": 7999},
        "SRS":         {"min": 8000,  "max": 8999},
        "BFW":         {"min": 9000,  "max": 9999},
        "mSlot_mCell": {"min": 90000, "max": 90999}
    }

    for channel, rng in channel_ranges.items():
        if rng["min"] <= test_case_number <= rng["max"]:
            return channel
    return None


def parse_host_config(host_config):
    """Parse and validate host configuration."""
    if host_config not in VALID_HOST_CONFIGS:
        print(f"Error: Invalid host configuration '{host_config}'. Must be one of: {', '.join(VALID_HOST_CONFIGS)}", file=sys.stderr)
        sys.exit(1)

    du_host, ru_host = host_config.split('_')
    return du_host, ru_host

def generate_script_params(args, params, du_host, ru_host):
    """Generate parameters for each of the 8 scripts."""

    if args.test_case.startswith('F08_'):
        script_params = generate_f08_script_params(args, params, du_host, ru_host)
    elif args.test_case.isdigit():
        script_params = generate_nrsim_script_params(args, params, du_host, ru_host)
    else:
        print(f"Error: Unable to generate script parameters for test_case: {args.test_case}")
        sys.exit(1)

    return script_params

def generate_nrsim_script_params(args, params, du_host, ru_host):
    """Generate parameters for each of the 8 scripts - for NRSIM"""
    script_params = {}

    # 1. copy_test_files.sh parameters
    script_params['copy_test_files'] = f"{params['pattern']}"

    # 2. build_aerial_sdk.sh parameters
    build_params = []
    build_params.append(f"--preset {params['preset']}")
    if args.custom_build_dir:
        build_params.append(f"--build_dir {args.custom_build_dir}.$(uname -m)")
    else:
        build_params.append(f"--build_dir build.{params['preset']}.$(uname -m)")
    script_params['build_aerial'] = " ".join(build_params) if build_params else ""

    # 3. setup1_DU.sh parameters
    setup1_params = []
    setup1_params.append(f"-y nrSim_SCF_CG1_{params['pattern']}")
    script_params['setup1_du'] = " ".join(setup1_params)

    # 4. setup2_RU.sh parameters
    # This script typically doesn't need special parameters, uses info from setup1_DU
    script_params['setup2_ru'] = ""

    # 5. test_config_nrSim.sh parameters
    config_params = []

    # mSlot_mCell is special - all other channels explicitly pass the channel name
    if params['channel'] == "mSlot_mCell":
        config_params.append(f"--channels 0x1ff")
    else:
        config_params.append(f"--channels {params['channel']}")

    script_params['test_config'] = " ".join(config_params) if config_params else ""

    # 6. run1_RU.sh parameters
    # Typically uses parameters set by test_config.sh
    run1_ru_params = []
    if args.custom_build_dir:
        run1_ru_params.append(f"--build_dir {args.custom_build_dir}.$(uname -m)")
    else:
        run1_ru_params.append(f"--build_dir build.{params['preset']}.$ARCH")
    script_params['run1_ru'] = " ".join(run1_ru_params) if run1_ru_params else ""

    # 7. run2_cuPHYcontroller.sh parameters
    # Typically uses parameters set by test_config.sh
    run2_cuphycontroller_params = []
    if args.custom_build_dir:
        run2_cuphycontroller_params.append(f"--build_dir {args.custom_build_dir}.$(uname -m)")
    else:
        run2_cuphycontroller_params.append(f"--build_dir build.{params['preset']}.$(uname -m)")
    script_params['run2_cuphycontroller'] = " ".join(run2_cuphycontroller_params) if run2_cuphycontroller_params else ""

    # 8. run3_testMAC.sh parameters
    # Typically uses parameters set by test_config.sh
    run3_testmac_params = []
    if args.custom_build_dir:
        run3_testmac_params.append(f"--build_dir {args.custom_build_dir}.$(uname -m)")
    else:
        run3_testmac_params.append(f"--build_dir build.{params['preset']}.$(uname -m)")
    script_params['run3_testmac'] = " ".join(run3_testmac_params) if run3_testmac_params else ""

    return script_params

def generate_f08_script_params(args, params, du_host, ru_host):
    """Generate parameters for each of the 8 scripts - for F08"""
    script_params = {}

    # 1. copy_test_files.sh parameters
    script_params['copy_test_files'] = f"{params['pattern']} --max_cells {params['num_cells']}"

    # 2. build_aerial_sdk.sh parameters
    build_params = []
    preset = params['preset']
    build_params.append(f"--preset {preset}")
    if args.custom_build_dir:
        build_params.append(f"--build_dir {args.custom_build_dir}.$(uname -m)")
    else:
        build_params.append(f"--build_dir build.{preset}.$(uname -m)")
    script_params['build_aerial'] = " ".join(build_params) if build_params else ""

    # 3. setup1_DU.sh parameters
    setup1_params = []
    if params['mumimo']:
        setup1_params.append("--mumimo 1")
    # Add YAML selection based on DU host
    yaml_suffix = f"F08_{du_host}"
    setup1_params.append(f"--cuphy-yaml={yaml_suffix}")
    setup1_params.append(f"--ru-host-type=_{ru_host}")
    script_params['setup1_du'] = " ".join(setup1_params)

    # 4. setup2_RU.sh parameters
    # This script typically doesn't need special parameters, uses info from setup1_DU
    script_params['setup2_ru'] = ""

    # 5. test_config.sh parameters
    config_params = [params['pattern']]
    config_params.append(f"--num-cells={params['num_cells']}")
    config_params.append(f"--num-ports={params['num_ports']}")
    config_params.append(f"--num-slots={params['num_slots']}")
    config_params.append(f"--compression={params['compression']}")
    if params['compression'] == 1:
        config_params.append(f"--BFP={params['bfp']}")
    config_params.append(f"--ehq={params['ehq']}")
    config_params.append(f"--green-ctx={params['green_ctx']}")
    config_params.append(f"--work-cancel={params['work_cancel']}")
    config_params.append(f"--pmu={params['pmu']}")
    config_params.append(f"--STT={params['stt']}")
    if params['log_nic_timings']:
        config_params.append("--log-nic-timings")
    if params['ru_worker_tracing']:
        config_params.append("--ru-worker-tracing")
    if params['reduced_logging']:
        config_params.append("--reduced-logging")
    if params['cupti']:
        config_params.append("--cupti")
    script_params['test_config'] = " ".join(config_params)

    # 6. run1_RU.sh parameters
    # Typically uses parameters set by test_config.sh
    run1_ru_params = []
    if args.custom_build_dir:
        run1_ru_params.append(f"--build_dir {args.custom_build_dir}.$(uname -m)")
    else:
        run1_ru_params.append(f"--build_dir build.{preset}.$(uname -m)")
    script_params['run1_ru'] = " ".join(run1_ru_params) if run1_ru_params else ""

    # 7. run2_cuPHYcontroller.sh parameters
    # Typically uses parameters set by test_config.sh
    run2_cuphycontroller_params = []
    if args.custom_build_dir:
        run2_cuphycontroller_params.append(f"--build_dir {args.custom_build_dir}.$(uname -m)")
    else:
        run2_cuphycontroller_params.append(f"--build_dir build.{preset}.$(uname -m)")
    script_params['run2_cuphycontroller'] = " ".join(run2_cuphycontroller_params) if run2_cuphycontroller_params else ""

    # 8. run3_testMAC.sh parameters
    # Typically uses parameters set by test_config.sh
    run3_testmac_params = []
    if args.custom_build_dir:
        run3_testmac_params.append(f"--build_dir {args.custom_build_dir}.$(uname -m)")
    else:
        run3_testmac_params.append(f"--build_dir build.{preset}.$(uname -m)")
    script_params['run3_testmac'] = " ".join(run3_testmac_params) if run3_testmac_params else ""

    return script_params


def find_threshold_file(script_dir: Path, threshold_type: str, pattern: str, compression: int,
                        bfp: int, ehq: int, green_ctx: int, num_cells: str) -> str:
    """
    Find a gating/warning perf_requirements file matching the test case parameters.

    Searches in perf_requirements/warning_gating/ for files matching:
    {threshold_type}_gh_F08_{pattern}_{compression_str}_{eh_str}_{gc_str}_{cells}C.csv

    Where:
    - threshold_type: 'gating_perf_requirements' or 'warning_perf_requirements'
    - compression_str: 'MODCOMP' or 'BFP{bfp}'
    - eh_str: 'EH' if EH enabled, or omitted
    - gc_str: 'GC' if green context enabled, or omitted

    Returns:
        Full path to the file if found, empty string otherwise
    """
    req_dir = script_dir / "aerial_postproc" / "scripts" / "cicd" / "perf_requirements" / "warning_gating"

    if not req_dir.exists():
        return ""

    if compression == 4:  # MODCOMP
        compression_str = "MODCOMP"
    else:
        compression_str = f"BFP{bfp}"

    components = [compression_str]
    if ehq:
        components.append("EH")
    if green_ctx:
        components.append("GC")

    cells_str = f"{int(num_cells):02d}C"

    exact_pattern = f"{threshold_type}_gh_F08_{pattern}_{'_'.join(components)}_{cells_str}.csv"
    exact_path = req_dir / exact_pattern
    if exact_path.exists():
        return str(exact_path)

    glob_pattern = f"{threshold_type}_*F08_{pattern}_*{cells_str}.csv"
    matches = list(req_dir.glob(glob_pattern))

    if not matches:
        return ""

    for match in matches:
        filename = match.name
        if compression == 4 and "MODCOMP" not in filename:
            continue
        if compression == 1 and f"BFP{bfp}" not in filename:
            continue

        has_eh = "_EH" in filename or "_EH_" in filename
        if ehq and not has_eh:
            continue
        if not ehq and has_eh:
            continue

        has_gc = "_GC" in filename or "_GC_" in filename
        if green_ctx and not has_gc:
            continue
        if not green_ctx and has_gc:
            continue

        return str(match)

    return ""


def find_absolute_threshold_file(script_dir: Path, is_mimo: bool, ehq: int) -> str:
    """
    Find the absolute threshold (perf requirements) file.

    Based on mMIMO pattern and EH modifier:
    - non-mMIMO + EH     -> perf_requirements_4tr_eh.csv
    - non-mMIMO + no EH  -> perf_requirements_4tr_noneh.csv
    - mMIMO + EH         -> perf_requirements_64tr_eh.csv
    - mMIMO + no EH      -> perf_requirements_64tr_noneh.csv

    Returns:
        Full path to the requirements file if found, empty string otherwise
    """
    req_dir = script_dir / "aerial_postproc" / "scripts" / "cicd" / "perf_requirements" / "absolute"

    if not req_dir.exists():
        return ""

    if is_mimo:
        tr_suffix = "64tr"
    else:
        tr_suffix = "4tr"

    if ehq:
        eh_suffix = "eh"
    else:
        eh_suffix = "noneh"

    filename = f"perf_requirements_{tr_suffix}_{eh_suffix}.csv"
    filepath = req_dir / filename

    if filepath.exists():
        return str(filepath)

    return ""


def generate_post_processing_params(args, params) -> dict:
    """
    Generate parameters for post-processing scripts.

    Returns a dictionary with the following keys:
    - post_processing_cicd_params: All flags for post_processing_cicd.sh
    - parse_logs_params: Flags for post_processing_parse.sh
    - post_processing_perf_params: Flags for post_processing_analyze.sh --perf-metrics
    - post_processing_compare_params: Flags for post_processing_analyze.sh --compare-logs
    - post_processing_gating_params: Flags for post_processing_analyze.sh --gating-threshold
    - post_processing_warning_params: Flags for post_processing_analyze.sh --warning-threshold
    - post_processing_absolute_params: Flags for post_processing_analyze.sh --absolute-threshold
    - post_processing_latency_params: Flags for post_processing_analyze.sh --latency-summary
    """
    script_dir = Path(__file__).parent
    pp_params = {}

    # For NRSIM tests, post-processing is not supported (yet)
    if not args.test_case.startswith('F08_'):
        pp_params['post_processing_cicd_params'] = ""
        pp_params['parse_logs_params'] = ""
        pp_params['post_processing_perf_params'] = ""
        pp_params['post_processing_compare_params'] = ""
        pp_params['post_processing_gating_params'] = ""
        pp_params['post_processing_warning_params'] = ""
        pp_params['post_processing_absolute_params'] = ""
        pp_params['post_processing_latency_params'] = ""
        return pp_params

    # Check for NOPOST modifier (reduced_logging)
    if params.get('reduced_logging', False):
        pp_params['post_processing_cicd_params'] = ""
        pp_params['parse_logs_params'] = ""
        pp_params['post_processing_perf_params'] = ""
        pp_params['post_processing_compare_params'] = ""
        pp_params['post_processing_gating_params'] = ""
        pp_params['post_processing_warning_params'] = ""
        pp_params['post_processing_absolute_params'] = ""
        pp_params['post_processing_latency_params'] = ""
        return pp_params

    # Extract parameters
    pattern = params.get('pattern', '')
    num_cells = params.get('num_cells', '6')
    compression = params.get('compression', 1)
    bfp = params.get('bfp', 9)
    ehq = params.get('ehq', 0)
    green_ctx = params.get('green_ctx', 0)
    is_mimo = params.get('mumimo', 0) == 1
    is_nicd = params.get('log_nic_timings', False)

    # Build common optional flags
    mmimo_flag = "--mmimo" if is_mimo else ""
    label_flag = f"--label {args.test_case}"

    # Find perf_requirements files
    gating_file = find_threshold_file(script_dir, "gating_perf_requirements", pattern, compression,
                                       bfp, ehq, green_ctx, num_cells)
    warning_file = find_threshold_file(script_dir, "warning_perf_requirements", pattern, compression,
                                        bfp, ehq, green_ctx, num_cells)
    absolute_file = find_absolute_threshold_file(script_dir, is_mimo, ehq)

    # Generate parse_logs_params
    parse_logs_opts = ["--perf-metrics"]
    if is_nicd:
        parse_logs_opts.append("--latency-summary")
    if is_mimo:
        parse_logs_opts.append("--mmimo")
    pp_params['parse_logs_params'] = " ".join(parse_logs_opts)

    # Generate post_processing_perf_params
    perf_opts = ["--perf-metrics"]
    if is_mimo:
        perf_opts.append("--mmimo")
    perf_opts.append(label_flag)
    pp_params['post_processing_perf_params'] = " ".join(perf_opts)

    # Generate post_processing_compare_params
    compare_opts = ["--compare-logs"]
    if is_mimo:
        compare_opts.append("--mmimo")
    compare_opts.append(label_flag)
    pp_params['post_processing_compare_params'] = " ".join(compare_opts)

    # Generate post_processing_gating_params
    if gating_file:
        pp_params['post_processing_gating_params'] = f"--gating-threshold {gating_file}"
    else:
        pp_params['post_processing_gating_params'] = ""

    # Generate post_processing_warning_params
    if warning_file:
        pp_params['post_processing_warning_params'] = f"--warning-threshold {warning_file}"
    else:
        pp_params['post_processing_warning_params'] = ""

    # Generate post_processing_absolute_params
    if absolute_file:
        pp_params['post_processing_absolute_params'] = f"--absolute-threshold {absolute_file}"
    else:
        pp_params['post_processing_absolute_params'] = ""

    # Generate post_processing_latency_params
    if is_nicd:
        latency_opts = ["--latency-summary"]
        if is_mimo:
            latency_opts.append("--mmimo")
        latency_opts.append(label_flag)
        pp_params['post_processing_latency_params'] = " ".join(latency_opts)
    else:
        pp_params['post_processing_latency_params'] = ""

    # Generate post_processing_cicd_params (combines everything for CICD wrapper)
    cicd_opts = []
    if gating_file:
        cicd_opts.append(f"--gating-threshold {gating_file}")
    if warning_file:
        cicd_opts.append(f"--warning-threshold {warning_file}")
    if absolute_file:
        cicd_opts.append(f"--absolute-threshold {absolute_file}")
    if is_nicd:
        cicd_opts.append("--latency-summary")
    if is_mimo:
        cicd_opts.append("--mmimo")
    cicd_opts.append(label_flag)
    pp_params['post_processing_cicd_params'] = " ".join(cicd_opts)

    return pp_params


def main():
    parser = argparse.ArgumentParser(
        description='Parse test case configuration and generate script parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Positional arguments
    parser.add_argument('test_case', help='Test case string (e.g., F08_6C_69_MODCOMP_STT480000_1P)')
    parser.add_argument('host_config', help='Host configuration (e.g., CG1_R750)')

    # Optional arguments for environment variable names
    parser.add_argument('--copy-test-files-params', dest='copy_test_files_var',
                        help='Environment variable name for copy_test_files.sh parameters')
    parser.add_argument('--build-aerial-params', dest='build_aerial_var',
                        help='Environment variable name for build_aerial_sdk.sh parameters')
    parser.add_argument('--setup1-du-params', dest='setup1_du_var',
                        help='Environment variable name for setup1_DU.sh parameters')
    parser.add_argument('--setup2-ru-params', dest='setup2_ru_var',
                        help='Environment variable name for setup2_RU.sh parameters')
    parser.add_argument('--test-config-params', dest='test_config_var',
                        help='Environment variable name for test_config.sh parameters')
    parser.add_argument('--run1-ru-params', dest='run1_ru_var',
                        help='Environment variable name for run1_RU.sh parameters')
    parser.add_argument('--run2-cuphycontroller-params', dest='run2_cuphycontroller_var',
                        help='Environment variable name for run2_cuPHYcontroller.sh parameters')
    parser.add_argument('--run3-testmac-params', dest='run3_testmac_var',
                        help='Environment variable name for run3_testMAC.sh parameters')

    # Build directory options
    parser.add_argument('--custom-build-dir', dest='custom_build_dir',
                        help='Custom build directory. Use $CUSTOM.$(uname -m) instead of using build.$PRESET.$(uname -m)')

    # Output options
    parser.add_argument('--output-file', '-o', dest='output_file',
                        help='Write export statements to file (in addition to stdout)')

    args = parser.parse_args()

    # Parse test case and host config
    params = parse_test_case_string(args.test_case)
    du_host, ru_host = parse_host_config(args.host_config)

    # Generate script parameters
    script_params = generate_script_params(args, params, du_host, ru_host)

    # Generate post-processing parameters
    pp_params = generate_post_processing_params(args, params)

    # Generate environment variable mappings
    env_vars = {}

    # Map argument names to script parameters
    var_mapping = {
        'copy_test_files_var': 'copy_test_files',
        'build_aerial_var': 'build_aerial',
        'setup1_du_var': 'setup1_du',
        'setup2_ru_var': 'setup2_ru',
        'test_config_var': 'test_config',
        'run1_ru_var': 'run1_ru',
        'run2_cuphycontroller_var': 'run2_cuphycontroller',
        'run3_testmac_var': 'run3_testmac'
    }

    for arg_name, param_key in var_mapping.items():
        var_name = getattr(args, arg_name)
        if var_name:
            env_vars[var_name] = script_params[param_key]

    # If no specific variables requested, use default names
    if not any(getattr(args, arg_name) for arg_name in var_mapping.keys()):
        env_vars['COPY_TEST_FILES_PARAMS'] = script_params['copy_test_files']
        env_vars['BUILD_AERIAL_PARAMS'] = script_params['build_aerial']
        env_vars['SETUP1_DU_PARAMS'] = script_params['setup1_du']
        env_vars['SETUP2_RU_PARAMS'] = script_params['setup2_ru']
        env_vars['TEST_CONFIG_PARAMS'] = script_params['test_config']
        env_vars['RUN1_RU_PARAMS'] = script_params['run1_ru']
        env_vars['RUN2_CUPHYCONTROLLER_PARAMS'] = script_params['run2_cuphycontroller']
        env_vars['RUN3_TESTMAC_PARAMS'] = script_params['run3_testmac']

        # Add post-processing environment variables
        env_vars['POST_PROCESSING_CICD_PARAMS'] = pp_params['post_processing_cicd_params']
        env_vars['PARSE_LOGS_PARAMS'] = pp_params['parse_logs_params']
        env_vars['POST_PROCESSING_PERF_PARAMS'] = pp_params['post_processing_perf_params']
        env_vars['POST_PROCESSING_COMPARE_PARAMS'] = pp_params['post_processing_compare_params']
        env_vars['POST_PROCESSING_GATING_PARAMS'] = pp_params['post_processing_gating_params']
        env_vars['POST_PROCESSING_WARNING_PARAMS'] = pp_params['post_processing_warning_params']
        env_vars['POST_PROCESSING_ABSOLUTE_PARAMS'] = pp_params['post_processing_absolute_params']
        env_vars['POST_PROCESSING_LATENCY_PARAMS'] = pp_params['post_processing_latency_params']

    # Generate output in test_config_summary.sh format
    output_lines = []

    # Add a header comment
    output_lines.append("# Generated by parse_test_config_params.py")
    output_lines.append(f"# Test case: {args.test_case}")
    output_lines.append(f"# Host config: {args.host_config}")

    # Add the VARS line that lists all variables
    var_names = list(env_vars.keys())
    output_lines.append(f'VARS="{" ".join(var_names)}"')

    # Add each variable assignment (without export)
    for var_name, value in env_vars.items():
        output_lines.append(f'{var_name}="{value}"')

    output_text = '\n'.join(output_lines) + '\n'

    # Write to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(output_text)
        os.chmod(args.output_file, 0o755)  # Make executable like test_config_summary.sh
        print(f"# Configuration written to {args.output_file}", file=sys.stderr)
    else:
        # If no file specified, print export statements for backward compatibility
        for var_name, value in env_vars.items():
            print(f'export {var_name}="{value}"')
        return

    # When file is written, print sourcing instructions
    print(f"# To use: source {args.output_file}", file=sys.stderr)

if __name__ == '__main__':
    main()
