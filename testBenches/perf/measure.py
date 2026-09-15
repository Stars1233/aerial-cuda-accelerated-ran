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

import os
import subprocess
import tempfile
from contextlib import suppress
from functools import partial
from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")


def _nvsmi(extra_args: List[str], *, sudo_prefix: List[str]) -> None:
    """Run an nvidia-smi state-change command, raising on non-zero exit.

    Args:
        extra_args: Arguments appended after "nvidia-smi" on the command line
            (e.g. ["-i", "0", "-pl", "300"]).
        sudo_prefix: Tokens to prepend to the argv list when invoking as a
            non-root user (typically ["sudo"]); empty when already root.

    Raises:
        subprocess.CalledProcessError: When nvidia-smi exits with a non-zero
            status. Caller is expected to let the exception propagate so the
            test bench fails fast on GPU-config errors.
    """
    cmd = [*sudo_prefix, "nvidia-smi", *extra_args]
    subprocess.run(cmd, check=True)


def _mig_step(
    gpu_id: int,
    nvsmi_args: List[str],
    err_msg: str,
    *,
    sudo_prefix: List[str],
    base,
    buffer_path: str,
) -> None:
    """Run an nvidia-smi MIG-control command, capturing combined stdout+stderr.

    Output is written to ``buffer_path`` (truncate-overwrite) so the caller
    can parse the final step's output later (e.g. the "Enabled MIG …" check
    after `-mig 1`). On non-zero exit the buffer is removed and
    ``base.error(err_msg)`` is invoked.

    Args:
        gpu_id: GPU index passed as ``-i <gpu_id>``.
        nvsmi_args: Tail of the nvidia-smi argv (everything after ``-i <gpu_id>``).
        err_msg: Operator-facing message routed through ``base.error`` on failure.

    Keyword Args:
        sudo_prefix: ``["sudo"]`` when running unprivileged, ``[]`` as root.
        base: ``arguments()`` base object exposing ``.error()``.
        buffer_path: Per-invocation output capture path; pass a tempfile-derived
            path rather than a hardcoded relative path so concurrent
            ``measure.py`` runs sharing a cwd don't race on the same file.
    """
    with open(buffer_path, "w") as f:
        proc = subprocess.run(
            [*sudo_prefix, "nvidia-smi", "-i", str(gpu_id), *nvsmi_args],
            stdout=f, stderr=subprocess.STDOUT, check=False,
        )
    if proc.returncode != 0:
        if os.path.exists(buffer_path):
            os.remove(buffer_path)
        base.error(err_msg)


def parse_field(val: str, convert: Callable[[str], T]) -> Optional[T]:
    """Parse a single CSV field returned by nvidia-smi --format=csv.

    nvidia-smi emits "N/A" (and occasionally empty strings) for unavailable
    metrics; both map to ``None`` so callers can fall back to a default.
    Otherwise the first whitespace-delimited token is fed to ``convert``
    (e.g. ``int``, ``float``) — strips trailing units like "MHz" or "W".

    Args:
        val: Raw CSV cell value.
        convert: Callable applied to the first token when the cell is present.

    Returns:
        The converted value, or ``None`` when nvidia-smi reported no value.
    """
    if val in {"N/A", ""}:
        return None
    return convert(val.split()[0])


if __name__ == "__main__":

    from measure.cli import arguments

    sudo_prefix: List[str] = [] if os.geteuid() == 0 else ["sudo"]

    base, args = arguments()

    # save GPU status (current clock frequency, power limit, persistence mode) restore after running test
    gpuStatSave = {}
    query_cmd = [
        "nvidia-smi", "-i", str(args.gpu),
        "--query-gpu=clocks.current.graphics,power.limit,persistence_mode",
        "--format=csv,noheader",
    ]
    result = subprocess.run(query_cmd, capture_output=True, encoding="utf-8", check=False)
    fields = [f.strip().strip("[]") for f in result.stdout.strip().split(",")]
    if result.returncode != 0 or len(fields) < 3:
        # base.error() may only log (depending on the base implementation); we must NOT
        # fall through to fields[0]/[1]/[2] indexing with garbage data and lose the
        # nvidia-smi diagnostic in an IndexError traceback. SystemExit(2) terminates
        # cleanly with a non-zero exit code regardless of what base.error chooses to do.
        base.error(f"Failed to query GPU state with nvidia-smi: {result.stderr.strip()}")
        raise SystemExit(2)
    gpuStatSave['clockFreq']   = parse_field(fields[0], int)
    gpuStatSave['powerLimit']  = parse_field(fields[1], float)
    # Route persistence_mode through parse_field for symmetry: it normalises both
    # "N/A" and "" to None, so a blank cell from nvidia-smi won't end up as the
    # falsy-but-non-None empty string that would silently bypass the 'Disabled'
    # guard difference between setup and restore.
    gpuStatSave['persistMode'] = parse_field(fields[2], str)

    # check whether need to change GPU configs
    # Only change a setting when its original value is known, so restore (below) can put it
    # back symmetrically; an unreadable (N/A) original is left untouched rather than modified.
    #
    # Setup + test execution wrapped in try/finally so the restore block always runs,
    # even if a mid-setup _nvsmi() call or the measure.*.measure() body raises. _nvsmi
    # uses check=True (raises on non-zero nvidia-smi exit), so without the finally a
    # failed power-limit set after a successful clock-freq set would leave the GPU in
    # a modified state until the next reboot. See MR !5385 (Greptile P1).
    try:
        if gpuStatSave['clockFreq'] is not None and gpuStatSave['clockFreq'] != args.freq:
            lgc_args = ["-i", str(args.gpu), "-lgc", str(args.freq)]
            if args.is_GH200:
                lgc_args.append("--mode=1")
            _nvsmi(lgc_args, sudo_prefix=sudo_prefix)

        if (args.power is not None) and (gpuStatSave['powerLimit'] is not None) and (gpuStatSave['powerLimit'] != args.power):
            _nvsmi(["-i", str(args.gpu), "-pl", str(args.power)], sudo_prefix=sudo_prefix)

        # Symmetric with the restore branch below: both gate on the same
        # `== 'Disabled'` guard, so the None (N/A original) case is a no-op
        # on both sides and never permanently flips persistence mode.
        if gpuStatSave['persistMode'] == 'Disabled':
            _nvsmi(["-i", str(args.gpu), "-pm", "1"], sudo_prefix=sudo_prefix)

        # start testing
        if args.mig is not None:

            # Per-invocation capture file: avoids the cwd race that hardcoded
            # "buffer.txt" had when concurrent measure.py runs share a cwd
            # (CI matrix, side-by-side perf shells). mkstemp atomically reserves
            # a unique path under tempfile.gettempdir().
            _bufFd, buffer_path = tempfile.mkstemp(prefix="measure_mig_", suffix=".txt")
            os.close(_bufFd)  # _mig_step reopens with "w"; the fd is just for path reservation
            try:
                # Bind the per-call kwargs once so the 4 step calls stay readable.
                mig_step = partial(_mig_step, args.gpu,
                                   sudo_prefix=sudo_prefix, base=base, buffer_path=buffer_path)

                # Clean up any existing MIG instances first
                mig_step(["mig", "-dci"],
                         "Failed to destroy MIG compute instances. Make sure you have proper permissions.")
                mig_step(["mig", "-dgi"],
                         "Failed to destroy MIG GPU instances. Make sure you have proper permissions.")
                # Disable MIG mode first (to clean state)
                mig_step(["-mig", "0"],
                         "Failed to disable MIG mode. Make sure you have proper permissions and the GPU supports MIG.")
                # Enable MIG mode
                mig_step(["-mig", "1"],
                         "Failed to enable MIG mode. Make sure you have proper permissions and MIG-capable GPU.")

                with open(buffer_path) as ifile:
                    lines = ifile.readlines()

                # Defensive: nvidia-smi typically prints "Enabled MIG ..." on success.
                # Scan ALL captured lines rather than just lines[0] — on a system where
                # a warning interleaves before the success line (rare, but possible with
                # combined stdout+stderr capture), a strict first-line check would
                # spuriously fail. Tokens within each line still require ["Enabled","MIG"]
                # in positions 0/1, matching the nvidia-smi success message format.
                mig_enabled = any(
                    len(toks := line.split()) >= 2 and toks[0] == "Enabled" and toks[1] == "MIG"
                    for line in lines
                )
                if mig_enabled:
                    import measure.mig

                    measure.mig.measure(base, args)

                else:

                    base.error("encountered issues in enabling MIG on the selected GPU")
            finally:
                # Always remove the per-invocation buffer, even if an exception
                # short-circuits the happy-path cleanup above.
                with suppress(OSError):
                    os.remove(buffer_path)

        else:

            import measure.nomig

            measure.nomig.measure(base, args)

    finally:
        # restore GPU status — each step independent (own suppress() guard) so a failure
        # in one nvidia-smi call doesn't block the others. Without this, a clock-freq
        # restore failure would leave power-limit and persistence-mode unrestored
        # (Greptile P2 on MR !5385).
        if gpuStatSave['clockFreq'] is not None and gpuStatSave['clockFreq'] != args.freq:
            lgc_args = ["-i", str(args.gpu), "-lgc", str(gpuStatSave['clockFreq'])]
            if args.is_GH200:
                lgc_args.append("--mode=1")
            with suppress(subprocess.CalledProcessError):
                _nvsmi(lgc_args, sudo_prefix=sudo_prefix)

        # Restore only when this run actually changed the power limit (args.power was
        # provided AND differs from the captured original). If the operator omitted
        # --power on this invocation, we do NOT reset to any baseline — the GPU's
        # power limit is left at whatever it was on entry. Consequence: a prior run
        # that set --power 350 and crashed before restore leaves the GPU at 350; a
        # subsequent run without --power inherits that 350. This is intentional —
        # measure.py treats the on-entry state as ground truth for the current
        # session, not as a fixed reset point.
        if gpuStatSave['powerLimit'] is not None and (args.power is not None) and (gpuStatSave['powerLimit'] != args.power):
            with suppress(subprocess.CalledProcessError):
                _nvsmi(["-i", str(args.gpu), "-pl", str(gpuStatSave['powerLimit'])], sudo_prefix=sudo_prefix)

        # Mirror of the setup guard above; see comment there for the None-case rationale.
        if gpuStatSave['persistMode'] == 'Disabled':
            with suppress(subprocess.CalledProcessError):
                _nvsmi(["-i", str(args.gpu), "-pm", "0"], sudo_prefix=sudo_prefix)
