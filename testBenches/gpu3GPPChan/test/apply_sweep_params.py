#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Expand sweep_params.yaml into one-parameter-at-a-time variant configs.

Reads a sweep spec (see test/sweep_params.yaml) and a baseline sls_chan
YAML config, and writes one patched config per (parameter, value) pair into
the output directory. Global `overrides` are applied to every variant
(including the optional unmodified baseline) so runs stay small.

Spec keys:
  base_config       baseline YAML (relative paths resolve against --sdk-root)
  overrides         dotted-key -> constant, applied to every variant.
                    Constants only; a list value is an error.
  seeds             optional list; every variant runs once per seed
                    (test_bench.rand_seed), names suffixed _s<seed>
  include_baseline  also emit the unmodified baseline (default true)
  sweep             dotted-key -> list; one variant per value

A sweep value that leaves the config identical to the baseline (e.g. sweeping
the base config's own default) is skipped with a note on stderr, so listing
defaults for documentation costs nothing.

Prints a tab-separated manifest to stdout, one line per run:

    <variant_name>\t<config_path>\t<description>

which run_param_sweep.sh consumes to drive the runs.
"""

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set an existing value selected by a dotted configuration key.

    Args:
        config: Configuration mapping to update in place.
        dotted_key: Dot-separated path such as ``system_level.scenario``.
        value: Replacement value for the selected key.

    Returns:
        None.

    Raises:
        KeyError: If a section or leaf in ``dotted_key`` does not exist.

    Examples:
        >>> config = {"system_level": {"scenario": "UMa"}}
        >>> set_dotted(config, "system_level.scenario", "UMi")
        >>> config["system_level"]["scenario"]
        'UMi'
    """
    keys = dotted_key.split(".")
    node = config
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"'{dotted_key}': section '{k}' not found in base config")
        node = node[k]
    if not isinstance(node, dict) or keys[-1] not in node:
        raise KeyError(f"'{dotted_key}': leaf key '{keys[-1]}' not found in base config")
    node[keys[-1]] = value


def variant_name(dotted_key: str, value: Any) -> str:
    """Build a filesystem-safe name containing the full swept key.

    Args:
        dotted_key: Dot-separated configuration key being swept.
        value: Value assigned to the swept key.

    Returns:
        A stable name that distinguishes equal leaf keys in different sections.

    Raises:
        None.

    Examples:
        >>> variant_name("system_level.scenario", "UMi")
        'system_level.scenario_UMi'
    """
    key = re.sub(r"[^A-Za-z0-9._+-]+", "-", dotted_key).strip("-")
    val = re.sub(r"[^A-Za-z0-9._+-]+", "-", str(value)).strip("-")
    return f"{key}_{val}"


def main() -> int:
    """Expand a sweep specification and print its variant manifest.

    Args:
        None.

    Returns:
        Zero on success or one when the specification is invalid.

    Raises:
        OSError: If an input file cannot be read or an output cannot be written.
        yaml.YAMLError: If an input YAML file is malformed.

    Examples:
        >>> main()  # doctest: +SKIP
        0
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to sweep_params.yaml")
    parser.add_argument("--out", required=True, help="Directory to write variant configs")
    parser.add_argument(
        "--sdk-root",
        default=None,
        help="SDK root used to resolve a relative base_config path (default: cwd)",
    )
    args = parser.parse_args()

    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    if spec is None:
        spec = {}
    elif not isinstance(spec, dict):
        print("ERROR: sweep spec root must be a mapping", file=sys.stderr)
        return 1

    base_config_path = Path(spec.get("base_config", ""))
    if not base_config_path.is_absolute():
        base_config_path = Path(args.sdk_root or ".") / base_config_path
    if not base_config_path.is_file():
        print(f"ERROR: base config not found: {base_config_path}", file=sys.stderr)
        return 1

    with open(base_config_path) as f:
        base = yaml.safe_load(f)
    if not isinstance(base, dict):
        print("ERROR: base config root must be a mapping", file=sys.stderr)
        return 1

    overrides = spec.get("overrides")
    sweep = spec.get("sweep")
    if overrides is None:
        overrides = {}
    elif not isinstance(overrides, dict):
        print("ERROR: 'overrides' must be a mapping of dotted-key -> constant", file=sys.stderr)
        return 1
    if sweep is None:
        sweep = {}
    elif not isinstance(sweep, dict):
        print("ERROR: 'sweep' must be a mapping of dotted-key -> value/list", file=sys.stderr)
        return 1
    seeds = spec.get("seeds")
    include_baseline = bool(spec.get("include_baseline", True))

    for key, value in overrides.items():
        if isinstance(value, list):
            print(f"ERROR: override '{key}' has a list value {value} — overrides are "
                  f"constants applied to every variant. To run every variant at "
                  f"several seeds use the top-level 'seeds:' key; to sweep a "
                  f"parameter move it under 'sweep:'.", file=sys.stderr)
            return 1
    if seeds is not None and not isinstance(seeds, list):
        seeds = [seeds]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def patched(patches: dict[str, Any]) -> dict[str, Any]:
        cfg = copy.deepcopy(base)
        for key, value in {**overrides, **patches}.items():
            set_dotted(cfg, key, value)
        return cfg

    baseline_cfg = patched({})

    def write_variant(name: str, patches: dict[str, Any]) -> None:
        cfg = patched(patches)
        if patches and cfg == baseline_cfg:
            print(f"NOTE: skipping '{name}' — identical to baseline "
                  f"(sweeps the base config's own value)", file=sys.stderr)
            return
        desc = ",".join(f"{k}={v}" for k, v in patches.items()) or "baseline"
        for seed in (seeds if seeds is not None else [None]):
            run_name, run_cfg, run_desc = name, cfg, desc
            if seed is not None:
                run_name = f"{name}_s{seed}"
                run_cfg = copy.deepcopy(cfg)
                set_dotted(run_cfg, "test_bench.rand_seed", seed)
                run_desc = f"{desc},rand_seed={seed}"
            path = out_dir / f"{run_name}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(run_cfg, f, sort_keys=False)
            print(f"{run_name}\t{path}\t{run_desc}")

    if include_baseline:
        write_variant("baseline", {})

    for dotted_key, values in sweep.items():
        if not isinstance(values, list):
            values = [values]
        for value in values:
            write_variant(variant_name(dotted_key, value), {dotted_key: value})

    return 0


if __name__ == "__main__":
    sys.exit(main())
