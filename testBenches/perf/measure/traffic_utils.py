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

from __future__ import annotations

import logging
import os
import yaml
from typing import Any
from typing import Union

TVValue = Union[str, list[str]]


def get_tv_for_cell(filenames_dict: dict[str, TVValue], testcase: str, cell_idx: int) -> str:
    """Get the TV filename for a specific cell using modulo selection."""
    tv_value = filenames_dict[testcase]
    if isinstance(tv_value, list):
        if not tv_value:
            raise ValueError(f"Empty TV list for testcase: {testcase}")
        return tv_value[cell_idx % len(tv_value)]
    return tv_value


def expand_tvs_for_cells(
    filenames_dict: dict[str, TVValue], testcases_list: list[str], vfld: str
) -> list[str]:
    """Expand TVs for all cells (full paths under vfld)."""
    return [
        os.path.join(vfld, get_tv_for_cell(filenames_dict, testcase, cell_idx))
        for cell_idx, testcase in enumerate(testcases_list)
    ]


def drop_minus_one_overrides(node: Any) -> Any:
    """Recursively remove -1 values and empty containers from override trees."""
    if isinstance(node, dict):
        cleaned = {}
        for key, value in node.items():
            pruned = drop_minus_one_overrides(value)
            if pruned is None:
                continue
            cleaned[key] = pruned
        return cleaned if cleaned else None

    if isinstance(node, list):
        cleaned = [item for item in (drop_minus_one_overrides(v) for v in node) if item is not None]
        return cleaned if cleaned else None

    if isinstance(node, (int, float)) and node == -1:
        return None

    return node


def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def load_tdd_yaml_overrides(
    yaml_path: str | None, logger: logging.Logger
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Load optional TDD YAML overrides for priorities/start_delay/tv_overrides/cumac_options."""
    priorities = None
    start_delay = None
    tv_overrides = None
    cumac_options = None

    if not yaml_path:
        return priorities, start_delay, tv_overrides, cumac_options

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            ycfg = yaml.safe_load(f) or {}
        if isinstance(ycfg, dict):
            if isinstance(ycfg.get("tdd_priorities"), dict):
                priorities = ycfg["tdd_priorities"]
            elif isinstance(ycfg.get("config"), dict) and isinstance(ycfg["config"].get("tdd_priorities"), dict):
                priorities = ycfg["config"]["tdd_priorities"]

            if isinstance(ycfg.get("start_delay"), dict):
                start_delay = ycfg["start_delay"]
            elif isinstance(ycfg.get("config"), dict) and isinstance(ycfg["config"].get("start_delay"), dict):
                start_delay = ycfg["config"]["start_delay"]

            if isinstance(ycfg.get("override_test_vectors"), dict):
                tv_overrides = ycfg["override_test_vectors"]
            elif isinstance(ycfg.get("config"), dict) and isinstance(ycfg["config"].get("override_test_vectors"), dict):
                tv_overrides = ycfg["config"]["override_test_vectors"]

            if isinstance(ycfg.get("cumac_options"), dict):
                cumac_options = ycfg["cumac_options"]
            elif isinstance(ycfg.get("config"), dict) and isinstance(ycfg["config"].get("cumac_options"), dict):
                cumac_options = ycfg["config"]["cumac_options"]
    except (yaml.YAMLError, FileNotFoundError) as e:
        logger.warning(
            "YAML config %r failed to load: %s; falling back to JSON priorities.",
            yaml_path,
            e,
            exc_info=False,
        )
    except Exception as e:
        logger.warning(
            "Error reading YAML config %r: %s; falling back to JSON priorities.",
            yaml_path,
            e,
            exc_info=False,
        )

    return priorities, start_delay, tv_overrides, cumac_options

