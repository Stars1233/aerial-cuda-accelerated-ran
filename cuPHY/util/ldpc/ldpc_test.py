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

# Unified LDPC decoder test-bench entry. This is a THIN DISPATCHER -- it adds
# no product (src/) code; each subcommand simply runs the existing mode tool with
# its own arguments:
#
#   llr-diff   -> llr_diff_sweep.py                (TV-free APP fidelity vs a ref decoder)
#   llr-check  -> llr_check_sweep.py               (TV-based APP fidelity vs the MATLAB golden)
#   find-snr   -> ldpc_rm.py find-snr              (BLER: SNR at a target TB BLER)
#   find-bler  -> ldpc_rm.py find-bler             (BLER over a fixed SNR range)
#
# Examples:
#   ldpc_test.py llr-diff  --algos 35 --bg 1 --z 32-384 --p 4-46
#   ldpc_test.py llr-check --algo 40 --testvectors-dir <dir>
#   ldpc_test.py find-snr  --n-prb 151 --nl 1 --mcs 27 --mcs-table 2 --algo 55 -n 10

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# subcommand -> (module to import, argv prefix injected before the user's args)
_DISPATCH = {
    "llr-diff":  ("llr_diff_sweep",  []),
    "llr-check": ("llr_check_sweep", []),
    "find-snr":  ("ldpc_rm",         ["find-snr"]),
    "find-bler": ("ldpc_rm",         ["find-bler"]),
}

_HELP = """\
usage: ldpc_test.py <command> [options]   (run 'ldpc_test.py <command> --help' for options)

Per-iteration APP fidelity -- do the decoder's beliefs track a reference?
  llr-diff      vs a reference GPU decoder (default algo40 box-plus), TV-free input
  llr-check     vs the MATLAB golden (/APP_history in a TV .h5)

BLER -- does it actually decode?
  find-snr      find the SNR at a target TB BLER
  find-bler     measure TB BLER over a fixed SNR range
"""


def main() -> int:
    """Dispatch a subcommand (sys.argv[1:]) to its tool's main() via _DISPATCH.

    Reads the subcommand from argv, maps it through _DISPATCH to a tool module,
    rewrites sys.argv for that tool, and calls its main().

    Returns:
        int: process exit status -- 0/2 for the built-in help, 2 for an unknown
        command, otherwise the dispatched tool's own main() return value.
    """
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_HELP)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if cmd not in _DISPATCH:
        sys.stderr.write(f"ldpc_test.py: unknown command '{cmd}'\n\n" + _HELP)
        return 2
    module_name, prefix = _DISPATCH[cmd]
    mod = __import__(module_name)                     # no top-level side effects
    sys.argv = [f"{module_name}.py"] + prefix + rest  # let the tool parse its own args
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
