#!/bin/bash

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

# Exit on first error
set -euo pipefail

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
cd $PROJECT_ROOT

# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"

echo "${SCRIPT} starting..."

echo "pyAerial: Static analysis with flake8..."
"${PYTHON}" -m flake8 --color=always --config=./scripts/.flake8 ./src/aerial ./tests
echo "Success."

echo "pyAerial: Static analysis with pylint..."
"${PYTHON}" -m pylint --output-format=colorized --rcfile scripts/.pylintrc ./src/aerial

echo "pyAerial: Static type checking..."
"${PYTHON}" -m mypy --color-output src/aerial \
    --no-incremental \
    --disallow-incomplete-defs \
    --disallow-untyped-defs \
    --no-strict-optional \
    --disable-error-code attr-defined

echo pyAerial: Verify docstring coverage...
pushd src > /dev/null
"${PYTHON}" -m interrogate -vv --omit-covered-files
popd > /dev/null

# Finished
echo $SCRIPT finished with success.
