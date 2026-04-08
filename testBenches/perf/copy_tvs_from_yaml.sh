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

# Copy .h5 TV files listed under config.vector_files in a cubb_gpu_test_config-style YAML
# from SRC to DST.
#
# Usage: ./copy_tvs_from_yaml.sh <yaml_file> <src_dir> <dst_dir>
# Example: ./copy_tvs_from_yaml.sh cubb_gpu_test_config.yaml /path/to/tv/archive <aerial_sdk>/testVectors

set -euo pipefail

usage() {
    echo "Usage: $0 <yaml_file> <src_dir> <dst_dir>"
    echo "  yaml_file  YAML config with config.vector_files containing .h5 TV filenames"
    echo "  src_dir    Directory containing the .h5 TV files"
    echo "  dst_dir    Destination directory (created if missing)"
    exit 1
}

if [[ $# -ne 3 ]]; then
    usage
fi

YAML="$1"

if [[ ! -f "$YAML" ]]; then
    echo "Error: YAML file not found: $YAML"
    exit 1
fi

if [[ ! -d "$2" ]]; then
    echo "Error: source directory does not exist: $2"
    exit 1
fi

SRC="$(realpath "$2")"
DST="$(realpath -m "$3")"

# Extract .h5 filenames from YAML (matches .h5 tokens anywhere in file; no yq dependency).
get_tv_files() {
    grep -oE '[A-Za-z0-9_.\-]+\.h5' "$YAML" || true
}

mkdir -p "$DST"
COPIED=0
MISSING=0

while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    src_file="${SRC}/${name}"
    dst_file="${DST}/${name}"
    if [[ -f "$src_file" ]]; then
        cp -v "$src_file" "$dst_file"
        ((COPIED++)) || true
    else
        echo "Missing: $src_file"
        ((MISSING++)) || true
    fi
done < <(get_tv_files | sort -u)

echo "Done: $COPIED copied, $MISSING missing."
