#!/bin/bash

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

# Resolve pyAerial Python interpreter and pip from the project venv when present.

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
PYAERIAL_ROOT=$(dirname "$(dirname "$SCRIPT")")
VENV="${PYAERIAL_ROOT}/.venv"
PYAERIAL_VENV_PROMPT="pyaerial"

if [[ -x "${VENV}/bin/python" ]]; then
    export PYTHON="${VENV}/bin/python"
    export PIP="${VENV}/bin/pip"
else
    export PYTHON="${PYTHON:-python3}"
    export PIP="${PIP:-pip3}"
fi

export PYAERIAL_ROOT

# PyTorch CUDA 13.2 wheels (torch==2.12.1+cu132) live on the cu132 simple index.
export TORCH_EXTRA_INDEX_URL="${TORCH_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu132}"

# Mirror stdout/stderr to the controlling terminal.
#
# Ninja (used by cmake --build) captures each custom-command step into a log
# file and only prints it when the step finishes. Teeing through /dev/tty
# bypasses that capture so trtexec/pytest output appears live.

_pyaerial_enable_forced_colors() {
    # Child tools see a pipe (Ninja/tee), not a TTY, and disable ANSI by default.
    if [[ -t 1 ]]; then
        return 0
    fi
    export TERM="${TERM:-xterm-256color}"
    export PY_COLORS=1
    export PYTEST_COLOR=yes
    export FORCE_COLOR=1
    export CLICOLOR_FORCE=1
    export MYPY_FORCE_COLOR=1
}

pyaerial_stream_output_to_tty() {
    if [[ "${PYAERIAL_STREAM_TTY:-1}" != "1" ]]; then
        return 0
    fi
    # Only mirror when stdout is piped (e.g. Ninja capture). Skip on an interactive
    # terminal to avoid duplicate output from tee and the real stdout.
    if [[ -t 1 ]]; then
        return 0
    fi
    _pyaerial_enable_forced_colors
    if [[ -e /dev/tty ]] && [[ -w /dev/tty ]]; then
        exec > >(tee /dev/tty) 2>&1
    fi
    # Ninja prints the step comment without a trailing newline.
    echo
}

pyaerial_stream_output_to_tty
