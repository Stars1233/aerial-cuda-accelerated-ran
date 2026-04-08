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
import yaml
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from pathlib import Path


def _load_yaml_slot_config(args: Any) -> Optional[Dict[str, Any]]:
    """Load the `tdd_slot_config` mapping from the user YAML config.

    Args:
        args: Parsed arguments object. This function reads `args.yaml` as
            `yaml_path` and parses that file as `ycfg`.

    Returns:
        Optional[Dict[str, Any]]: The `tdd_slot_config` mapping when present
        (either top-level or under `config`), otherwise `None`. This function
        may return `None` when `yaml_path` is missing, on read/parse errors, if
        `ycfg` is not a dict, or when no `tdd_slot_config` section exists.
    """
    yaml_path = getattr(args, "yaml", None)
    if not yaml_path:
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as ifile:
            ycfg = yaml.safe_load(ifile) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None

    if not isinstance(ycfg, dict):
        return None

    if isinstance(ycfg.get("tdd_slot_config"), dict):
        return ycfg["tdd_slot_config"]

    cfg = ycfg.get("config")
    if isinstance(cfg, dict) and isinstance(cfg.get("tdd_slot_config"), dict):
        return cfg["tdd_slot_config"]

    return None


def _validate_slot_config(
    pattern: str,
    slot_config: Mapping[str, Sequence[Union[int, str]]],
    pattern_len: int,
) -> Dict[str, List[int]]:
    """Validate and normalize per-channel slot masks for a TDD pattern.

    Args:
        pattern: TDD pattern name used in validation error messages.
        slot_config: Mapping from channel name to sequence of slot mask values.
            Required channels are ``PDSCH``, ``PDCCH``, ``CSIRS``, ``PBCH``,
            and ``MAC``. Values are validated element-by-element and cast to
            ``int``.
        pattern_len: Expected number of slots in the pattern.

    Returns:
        Dict[str, List[int]]: Normalized mapping where each channel maps to a
        list of integers containing only ``0`` or ``1``.

    Raises:
        ValueError: If any required channel is missing, a channel value is not
        a list/sequence with length ``pattern_len``, an element cannot be
        interpreted as an allowed value, or any normalized value is not in
        ``{0, 1}``.

    Notes:
        ``MAC2`` defaults to the validated ``MAC`` cadence when omitted. If
        provided, ``MAC2`` is validated with the same casting and ``0/1`` rules
        and must also match ``pattern_len``.
    """
    required_channels = ("PDSCH", "PDCCH", "CSIRS", "PBCH", "MAC")
    normalized = {}
    for channel in required_channels:
        if channel not in slot_config:
            raise ValueError(
                f"Missing '{channel}' in slot config for pattern '{pattern}'"
            )

        values = slot_config[channel]
        if not isinstance(values, list):
            raise ValueError(
                f"Slot config '{pattern}.{channel}' must be a list of 0/1 values"
            )
        if len(values) != pattern_len:
            raise ValueError(
                f"Slot config '{pattern}.{channel}' length {len(values)} does not match pattern_len {pattern_len}"
            )

        channel_values = []
        for idx, v in enumerate(values):
            if not isinstance(v, (int, bool)):
                raise ValueError(
                    f"Slot config '{pattern}.{channel}[{idx}]' has invalid value {v!r} "
                    f"(type {type(v).__name__}); expected 0/1 (int or bool)"
                )
            iv = int(v)
            if iv not in (0, 1):
                raise ValueError(
                    f"Slot config '{pattern}.{channel}[{idx}]' has invalid value {v!r}; expected 0 or 1"
                )
            channel_values.append(iv)
        normalized[channel] = channel_values

    # MAC2 follows MAC cadence by default unless explicitly provided.
    mac2_values_raw = slot_config.get("MAC2", normalized["MAC"])
    normalized["MAC2"] = []
    for idx, v in enumerate(mac2_values_raw):
        if not isinstance(v, (int, bool)):
            raise ValueError(
                f"Slot config '{pattern}.MAC2[{idx}]' has invalid value {v!r} "
                f"(type {type(v).__name__}); expected 0/1 (int or bool)"
            )
        iv = int(v)
        if iv not in (0, 1):
            raise ValueError(
                f"Slot config '{pattern}.MAC2[{idx}]' has invalid value {v!r}; expected 0 or 1"
            )
        normalized["MAC2"].append(iv)
    if len(normalized["MAC2"]) != pattern_len:
        raise ValueError(
            f"Slot config '{pattern}.MAC2' length {len(normalized['MAC2'])} does not match pattern_len {pattern_len}"
        )

    return normalized


def traffic_het(args, vectors, k, testcases, filenames):

    testcases_dl, testcases_ul = testcases
    filenames_dl, filenames_ul = filenames

    ofile = open(vectors, "w")

    payload = {}
    payload["cells"] = k

    channels = []

    for sweep_idx in range(args.sweeps):

        channel = {}
        tidxs_dl = np.random.randint(0, len(testcases_dl), k)
        tidxs_ul = np.random.randint(0, len(testcases_ul), k)

        if not args.is_no_pdsch:
            channel["PDSCH"] = [
                os.path.join(args.vfld, filenames_dl[testcases_dl[tidx_dl]])
                for tidx_dl in tidxs_dl
            ]
        if not args.is_no_pusch:
            channel["PUSCH"] = [
                os.path.join(args.vfld, filenames_ul[testcases_ul[tidx_ul]])
                for tidx_ul in tidxs_ul
            ]

        channels.append(channel)

    payload["slots"] = channels

    ofile = open(vectors, "w")
    yaml.dump(payload, ofile)
    ofile.close()


def traffic_avg(args, vectors, testcases, filenames):

    script_dir = Path(__file__).parent
    with open(script_dir / "slotConfig.json", "r", encoding="utf-8") as ifile:
        slot_config_json = json.load(ifile)
    slot_config_yaml = _load_yaml_slot_config(args)
    if isinstance(slot_config_yaml, dict) and args.pattern in slot_config_yaml:
        slot_config_raw = slot_config_yaml[args.pattern]
    elif isinstance(slot_config_yaml, dict):
        raise KeyError(
            f"Pattern '{args.pattern}' is missing in YAML tdd_slot_config. "
            "Add it to your YAML, or remove tdd_slot_config to use measure/TDD/slotConfig.json defaults/examples."
        )
    elif args.pattern in slot_config_json:
        slot_config_raw = slot_config_json[args.pattern]
    else:
        raise KeyError(
            f"Pattern '{args.pattern}' is not defined in YAML tdd_slot_config or measure/TDD/slotConfig.json"
        )

    slot_config = _validate_slot_config(args.pattern, slot_config_raw, args.pattern_len)
    if args.sweeps % args.pattern_len != 0:
        raise ValueError(
            f"Configured slots (--slots={args.sweeps}) must be a multiple of "
            f"tdd_slot_config length ({args.pattern_len}) for pattern '{args.pattern}'"
        )

    if args.pattern == "dddsu":
        from .DDDSU.traffic.generate import run

        run(args, vectors, testcases, filenames, slot_config)
    else:
        from .DDDSUUDDDD.traffic.generate import run

        run(args, vectors, testcases, filenames, slot_config)
