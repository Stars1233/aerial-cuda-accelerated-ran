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

"""YAML loading for the link adaptation configuration dataclasses.

Three behaviours are worth knowing about, because every file under ``config/``
relies on them:

* **Organizational sections.** A configuration file may group its keys under
  arbitrary section names for readability. Sections carry no meaning; only the
  leaf keys do, and each leaf key must name a field of the target dataclass.
* **``base:`` inheritance.** A file may set ``base: other.yaml`` (resolved
  relative to itself) and override only what differs. Chains are allowed and
  cycles are rejected.
* **Unknown-key rejection.** A misspelled key is an error with a suggestion,
  never a silently ignored line. A typo in a training config otherwise costs a
  full run before anyone notices the setting never applied.
"""

from __future__ import annotations

__all__ = [
    "YamlConfig",
    "finite",
    "flatten_sections",
    "integer",
    "load_document",
    "nonnegative_int",
    "positive_int",
]

import difflib
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

_BASE_KEY = "base"

TConfig = TypeVar("TConfig", bound="YamlConfig")


def finite(name: str, value: object) -> float:
    """Validate that ``value`` is a finite real number and return it."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def integer(name: str, value: object) -> int:
    """Validate that ``value`` is an integer and return it."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def positive_int(name: str, value: object) -> int:
    """Validate that ``value`` is a positive integer and return it."""

    number = integer(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def nonnegative_int(name: str, value: object) -> int:
    """Validate that ``value`` is a nonnegative integer and return it."""

    number = integer(name, value)
    if number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return number


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge ``override`` onto ``base``, recursing into nested mappings."""

    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_document(
    path: str | Path,
    seen: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Read one YAML file and fold in its ``base:`` ancestry.

    Public so a simulator adapter can reuse the same loading semantics for its
    own configuration file.
    """

    resolved = Path(path).resolve()
    if resolved in seen:
        chain = " -> ".join(str(entry) for entry in (*seen, resolved))
        raise ValueError(f"circular base: chain {chain}")
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file {resolved} does not exist")
    with resolved.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if document is None:
        document = {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{resolved} must contain a YAML mapping")

    document = dict(document)
    base = document.pop(_BASE_KEY, None)
    if base is None:
        return document
    base_path = Path(str(base))
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return _deep_merge(load_document(base_path, (*seen, resolved)), document)


def _unknown_key(key: str, known: Mapping[str, Any], config_cls: type) -> ValueError:
    hint = difflib.get_close_matches(key, known, n=1)
    suffix = f"; did you mean {hint[0]}?" if hint else ""
    return ValueError(
        f"unknown {config_cls.__name__} field {key!r}{suffix}"
    )


def flatten_sections(
    document: Mapping[str, Any],
    config_cls: type,
) -> dict[str, Any]:
    """Collapse organizational sections into a flat field mapping.

    Public for the same reason as :func:`load_document`: an adapter's own
    configuration file then behaves exactly like the ones under ``config/``.
    """

    known = {config_field.name: config_field for config_field in fields(config_cls)}
    flat: dict[str, Any] = {}
    origin: dict[str, str] = {}

    def assign(key: str, value: Any, section: str) -> None:
        previous = origin.get(key)
        if previous is not None:
            # Usually a section was renamed in a file that inherits via
            # ``base:``, leaving the key present under both names. Name both so
            # the duplicate is obvious.
            raise ValueError(
                f"{config_cls.__name__} field {key!r} was set more than once, "
                f"under {previous} and under {section}; keep it in one section"
            )
        origin[key] = section
        flat[key] = value

    for key, value in document.items():
        if key in known:
            assign(key, value, "the top level")
        elif isinstance(value, Mapping):
            for leaf, leaf_value in value.items():
                if leaf not in known:
                    raise _unknown_key(str(leaf), known, config_cls)
                assign(str(leaf), leaf_value, f"section {key!r}")
        else:
            raise _unknown_key(str(key), known, config_cls)
    return flat


class YamlConfig:
    """Mixin adding ``from_yaml`` to a configuration dataclass."""

    @classmethod
    def from_yaml(
        cls: type[TConfig],
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> TConfig:
        """Build the dataclass from ``path``, applying ``overrides`` last.

        ``overrides`` are flat ``field: value`` pairs, which is what the
        launchers' ``--set section.field=value`` produces.
        """

        document = load_document(Path(path))
        values = flatten_sections(document, cls)
        known = {config_field.name: config_field for config_field in fields(cls)}
        for key, value in (overrides or {}).items():
            if key not in known:
                raise _unknown_key(str(key), known, cls)
            values[str(key)] = value
        return cls(**values)
