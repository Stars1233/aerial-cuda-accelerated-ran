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

"""DL BFW and SRS isolation modes for Phase-3 measure.py (GT-12714).

PUSCH/PDSCH isolation uses existing ``--no_pdsch`` / ``--no_pusch`` flags
(typically via YAML). This module covers modes that ``--rec_bf`` cannot express
(DL-BFW without SRS/UL-BFW, or SRS without PDSCH/PUSCH). When no channel-only
flag is set, configure_channel_only() and validate_channel_only() are no-ops.
"""

import argparse

_CHANNEL_ONLY_FLAGS = (
    "is_dl_bf_only",
    "is_srs_only",
)


def active_channel_only_modes(args: argparse.Namespace) -> list[str]:
    """Return active channel-only flag attribute names from *args*.

    Args:
        args: Parsed CLI namespace from :mod:`measure.cli`.

    Returns:
        Names from ``_CHANNEL_ONLY_FLAGS`` that are ``True`` on *args*, or an
        empty list when no channel-only mode is selected.

    Raises:
        None

    Examples:
        >>> import argparse
        >>> ns = argparse.Namespace(is_dl_bf_only=True, is_srs_only=False)
        >>> active_channel_only_modes(ns)
        ['is_dl_bf_only']
    """
    return [name for name in _CHANNEL_ONLY_FLAGS if getattr(args, name, False)]


def configure_channel_only(args: argparse.Namespace) -> None:
    """Apply implicit disable flags for ``--dl_bf_only`` / ``--srs_only``.

    Mutates *args* so downstream measure/TDD code sees the expected disable
    flags without callers passing them explicitly.

    Args:
        args: Parsed CLI namespace from :mod:`measure.cli`.

    Returns:
        None

    Raises:
        ValueError: If more than one ``--*_only`` flag is set.

    Examples:
        >>> import argparse
        >>> ns = argparse.Namespace(
        ...     is_dl_bf_only=False,
        ...     is_srs_only=True,
        ...     is_no_pdsch=False,
        ...     is_no_pusch=False,
        ...     is_srs_isolate=False,
        ... )
        >>> configure_channel_only(ns)
        >>> ns.is_no_pdsch, ns.is_no_pusch, ns.is_srs_isolate
        (True, True, True)
    """
    active = active_channel_only_modes(args)
    if len(active) > 1:
        raise ValueError(
            "Only one channel-only flag may be used at a time: "
            "--dl_bf_only, --srs_only"
        )

    if args.is_dl_bf_only:
        args.is_no_pdsch = False
        args.is_no_pusch = True
        return

    if args.is_srs_only:
        args.is_no_pdsch = True
        args.is_no_pusch = True
        args.is_srs_isolate = True


def validate_channel_only(base: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate DL BFW / SRS isolation flags against other CLI options.

    Args:
        base: Argument parser used to report fatal CLI errors via ``error()``.
        args: Parsed CLI namespace after :func:`configure_channel_only`.

    Returns:
        None

    Raises:
        SystemExit: Via ``base.error()`` when an invalid combination is found.

    Examples:
        >>> import argparse
        >>> p = argparse.ArgumentParser()
        >>> ns = argparse.Namespace(
        ...     is_dl_bf_only=False,
        ...     is_srs_only=False,
        ...     is_rec_bf=False,
        ...     is_srs_isolate=False,
        ...     uc="uc_avg_F14_TDD.json",
        ... )
        >>> validate_channel_only(p, ns)  # no-op when no isolation flag is set
    """
    active = active_channel_only_modes(args)
    if not active:
        return

    if args.is_rec_bf:
        base.error("Channel-only flags cannot be combined with --rec_bf")

    if args.is_dl_bf_only or args.is_srs_only:
        if (
            "TDD" not in args.uc
            or "_avg_" not in args.uc
            or ("F14" not in args.uc and "F09" not in args.uc)
        ):
            base.error(
                "--dl_bf_only and --srs_only require F09/F14 avg-cell use cases"
            )

    if args.is_dl_bf_only and args.is_srs_isolate:
        base.error("Cannot use --dl_bf_only with --srs_isolate")

    if args.is_dl_bf_only:
        if args.is_no_pdsch or not args.is_no_pusch:
            base.error("--dl_bf_only enables PDSCH+DLBFW without PUSCH")
    if args.is_srs_only:
        if not args.is_no_pdsch or not args.is_no_pusch or not args.is_srs_isolate:
            base.error("--srs_only enables isolated SRS without PDSCH/PUSCH")
