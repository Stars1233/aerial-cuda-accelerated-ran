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

# Shared helpers for the LDPC example bench scripts. Locates the example binary
# (cuphy_ex_ldpc / cuphy_ex_ldpc_rm) under the usual build-dir names so each
# sweep does not hard-code a single build directory.
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # -> repo root
# Build-dir names probed in order when --bin is not given.
BIN_CANDIDATES = ["build.aarch64", "build-minimal-arm", "build-minimal-x86", "build"]


def autodetect_bin(name: str = "cuphy_ex_ldpc") -> str | None:
    """Locate a built cuPHY example binary under the known build directories.

    Probes ``BIN_CANDIDATES`` in order and returns the first existing
    ``<candidate>/cuPHY/examples/error_correction/<name>`` path, so a sweep is
    not tied to a single build directory.

    Args:
        name: Example binary to look for, e.g. ``cuphy_ex_ldpc`` or
            ``cuphy_ex_ldpc_rm``.

    Returns:
        Absolute path to the first existing binary, or ``None`` if none of the
        candidate build directories contain it.

    Raises:
        None: Missing directories and files are treated as "not found", not
        errors.

    Examples:
        >>> autodetect_bin("cuphy_ex_ldpc")  # doctest: +SKIP
        '/opt/nvidia/cuBB/build/cuPHY/examples/error_correction/cuphy_ex_ldpc'
        >>> autodetect_bin("does_not_exist") is None  # doctest: +SKIP
        True
    """
    rel = os.path.join("cuPHY", "examples", "error_correction", name)
    for d in BIN_CANDIDATES:
        p = os.path.join(REPO, d, rel)
        if os.path.isfile(p):
            return p
    return None


def resolve_bin(arg_bin: str | None, name: str = "cuphy_ex_ldpc") -> str:
    """Resolve the path to a cuPHY example binary, or exit with a clear error.

    Uses ``arg_bin`` when provided, otherwise falls back to
    :func:`autodetect_bin`. Terminates the process with a helpful message when
    no usable binary can be found.

    Args:
        arg_bin: Explicit binary path from the caller, e.g. a ``--bin`` CLI
            argument, or ``None`` to auto-detect.
        name: Example binary to look for when auto-detecting, e.g.
            ``cuphy_ex_ldpc`` or ``cuphy_ex_ldpc_rm``.

    Returns:
        Absolute path to an existing example binary.

    Raises:
        SystemExit: If ``arg_bin`` is given but does not exist, if
            auto-detection finds no binary under ``BIN_CANDIDATES``, or if the
            resolved path exists but is not executable.

    Examples:
        >>> resolve_bin("/path/to/cuphy_ex_ldpc")  # doctest: +SKIP
        '/path/to/cuphy_ex_ldpc'
        >>> resolve_bin(None, "cuphy_ex_ldpc_rm")  # doctest: +SKIP
        '/opt/nvidia/cuBB/build/cuPHY/examples/error_correction/cuphy_ex_ldpc_rm'
    """
    p = arg_bin or autodetect_bin(name)
    if not p or not os.path.isfile(p):
        if arg_bin:
            raise SystemExit(f"{name} not found: {arg_bin}")
        raise SystemExit(
            f"{name} not found; looked under: {', '.join(BIN_CANDIDATES)} "
            f"(build it, or pass --bin <path>)")
    if not os.access(p, os.X_OK):
        raise SystemExit(f"{name} found but not executable: {p}")
    return p
