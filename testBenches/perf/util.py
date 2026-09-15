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

"""Shared helpers for testBenches/perf scripts."""

from typing import Any, List, Optional, Tuple


def to_int(v: Any) -> Optional[int]:
    """Best-effort int conversion. Returns None on failure (treated as N/A)."""
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def to_float(v: Any) -> float:
    """Best-effort float conversion. Returns NaN on failure."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return float('nan')


def well_formed_samples(samples: Any, min_fields: int = 4) -> Tuple[List[Any], int]:
    """Filter `samples` to rows that are list/tuple with at least `min_fields` items.

    Returns (filtered_rows, num_skipped).
    """
    if not isinstance(samples, list):
        return [], 0
    filtered = [row for row in samples if isinstance(row, (list, tuple)) and len(row) >= min_fields]
    return filtered, len(samples) - len(filtered)
