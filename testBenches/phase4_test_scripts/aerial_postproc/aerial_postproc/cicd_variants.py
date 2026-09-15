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

# Pure-stdlib shared source of truth for absolute perf_requirements template
# selection. Kept dependency-free (no pandas/numpy) so it can be imported both by
# the aerial_postproc venv tooling and by parse_test_config_params.py, which runs
# under plain python3 without the venv.

# Maps a traffic pattern to a variant-specific absolute template suffix. Patterns
# not listed here use the generic mMIMO x EH template.
ABSOLUTE_THRESHOLD_VARIANT_MAP: dict[str, str] = {
    "101":  "variantA",
    "101a": "variantA",
}


def absolute_template_candidates(tr_suffix: str, eh_suffix: str, pattern: str = "") -> list[str]:
    """Return candidate absolute-template filenames in preference order.

    Builds the list of absolute perf_requirements template filenames to try for a
    given configuration. When the pattern maps to a variant (via
    ABSOLUTE_THRESHOLD_VARIANT_MAP), the variant-specific filename is listed first
    and the generic mMIMO x EH filename is listed as a fallback; otherwise only the
    generic filename is returned. The caller is expected to pick the first
    candidate that exists on disk.

    Args:
        tr_suffix: Antenna/transceiver suffix, e.g. "4tr" or "64tr".
        eh_suffix: Early-HARQ suffix, e.g. "eh" or "noneh".
        pattern: Traffic pattern (e.g. "101a"). Matched case-insensitively against
            ABSOLUTE_THRESHOLD_VARIANT_MAP. Empty/unmapped patterns yield only the
            generic candidate.

    Returns:
        Ordered list of candidate filenames (variant-specific first when
        applicable, generic last). Always contains at least the generic filename.

    Examples:
        >>> absolute_template_candidates("4tr", "eh", "101a")
        ['perf_requirements_4tr_eh_variantA.csv', 'perf_requirements_4tr_eh.csv']
        >>> absolute_template_candidates("64tr", "noneh", "79")
        ['perf_requirements_64tr_noneh.csv']
    """
    variant = ABSOLUTE_THRESHOLD_VARIANT_MAP.get(pattern.lower(), "") if pattern else ""
    candidates = []
    if variant:
        candidates.append(f"perf_requirements_{tr_suffix}_{eh_suffix}_{variant}.csv")
    candidates.append(f"perf_requirements_{tr_suffix}_{eh_suffix}.csv")
    return candidates
