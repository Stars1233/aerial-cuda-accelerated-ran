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

#
# quickstart-aerial.sh - Quick start script for Aerial SDK development
#
# This script starts the Aerial container and builds the SDK with
# predefined settings for quick onboarding.
#
# Usage:
#   ./quickstart-aerial.sh              # Run container and build
#   ./quickstart-aerial.sh --tag        # Run tag image locally
#   ./quickstart-aerial.sh --dry-run    # Show commands without executing
#

set -e

# Identify SCRIPT_DIR and SDK root
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
[[ -f "$SCRIPT_DIR/includes.sh" ]] && source "$SCRIPT_DIR/includes.sh" || { echo "ERROR: includes.sh not found: $SCRIPT_DIR/includes.sh" >&2; exit 1; }

# Log to current directory (override default /var/log/...)
LOGFILE="quickstart-aerial.log"

# Parse common arguments (--dry-run, --verbose) and populate REMAINING_ARGS
parse_common_args "$@"
init_docker_cmd

# Parse script-specific arguments from REMAINING_ARGS
BUILD_ONLY_MODE=0
TAG_IMAGE=0

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Quick start script for Aerial SDK development. Starts the Aerial container
and builds the SDK with predefined settings.

OPTIONS:
    --tag                Tag pulled image as aerial-cuda-accelerated-ran:latest
    --build-only         Skip container start, only run build (must be inside container)
    --dry-run            Show commands without executing
    --verbose            Enable verbose output
    -h, --help           Display this help message and exit

ENVIRONMENT VARIABLES:
    AERIAL_VERSION_TAG   Container image tag (default: 26-1)
    PROFILE              Build profile file name (sources cmake-profiles/<PROFILE> if set).
                          Use 'oai.conf', 'fapi_10_02.conf', 'fapi_10_04.conf', or your own <name>.conf.
    BUILD_PRESET         Build preset: perf, 10_02, 10_04, 10_04_32dl (default: 10_02)
    PROFILE_CMAKE_FLAGS  Set by build profile; combined with platform and BUILD_CMAKE_FLAGS, will not override BUILD_PRESET if set.
    BUILD_CMAKE_FLAGS    User-added CMake flags; combined with profile + platform flags (not overridden by profile).
    AERIAL_BUILD_FLAGS   Options for build_aerial_sdk.sh before CMake-specific flags

EXAMPLES:
    # Default quickstart - start container and build
    $(basename "$0")

    # Build and tag image as aerial-cuda-accelerated-ran:latest
    $(basename "$0") --tag

    # Use different container tag
    AERIAL_VERSION_TAG=26-1 $(basename "$0")

    # Use different build preset
    BUILD_PRESET=perf $(basename "$0")

    # Use a build profile, e.g.:
    PROFILE=fapi_10_02.conf $(basename "$0")
    PROFILE=fapi_10_04.conf $(basename "$0")

    # Preview commands without executing
    $(basename "$0") --dry-run

EOF
    exit 0
}

for arg in "${REMAINING_ARGS[@]}"; do
    case $arg in
        --tag)
            TAG_IMAGE=1
            ;;
        --build-only)
            BUILD_ONLY_MODE=1
            ;;
        --test)
            TEST_AND_EXIT=1
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $arg"
            show_help
            ;;
    esac
done


cuBB_SDK_HOST=$(builtin cd "$SCRIPT_DIR/../.."; pwd)
cuBB_SDK=/opt/nvidia/cuBB
AERIAL_HOME_PATH="$HOME/aerial-cuda-accelerated-ran"

# Create symlink at ~/aerial-cuda-accelerated-ran if SDK is elsewhere
if [[ $cuBB_SDK_HOST != $AERIAL_HOME_PATH ]]; then
    LINK_PATH=$(readlink -f "$AERIAL_HOME_PATH" 2>/dev/null || true)
    if [[ "$cuBB_SDK_HOST" != "$LINK_PATH" ]]; then
        pushd ~ > /dev/null
        if [[ -L "$AERIAL_HOME_PATH" ]]; then
            rm "$AERIAL_HOME_PATH"
        elif [[ -e "$AERIAL_HOME_PATH" ]]; then
            echo "[ERROR] $AERIAL_HOME_PATH exists but is not a symlink. Please remove or rename it as L2 assumes this path to be correct." >&2
            exit 1
        fi
        execute "ln -sf $cuBB_SDK_HOST/ $AERIAL_HOME_PATH"
        popd > /dev/null
    fi
fi

# Load build profile if PROFILE is set (e.g. oai.conf, fapi_10_02.conf, fapi_10_04.conf); otherwise use defaults
# Profiles set BUILD_PRESET and PROFILE_CMAKE_FLAGS. User BUILD_CMAKE_FLAGS is combined afterward.
BUILD_PROFILES_DIR="${SCRIPT_DIR}/cmake-profiles"
PROFILE="${PROFILE:-oai.conf}"
if [[ -n "$PROFILE" ]]; then
    PROFILE_FILE="${BUILD_PROFILES_DIR}/${PROFILE}"
    if [[ -f "$PROFILE_FILE" ]]; then
        set -a
        source "$PROFILE_FILE"
        set +a
    else
        echo "[WARN] Build profile not found: $PROFILE_FILE (PROFILE=$PROFILE); using default BUILD_PRESET/PROFILE_CMAKE_FLAGS" >&2
    fi
fi
BUILD_PRESET="${BUILD_PRESET:-10_02}"
PROFILE_CMAKE_FLAGS="${PROFILE_CMAKE_FLAGS:-}"

# Platform-specific flags based on PLATFORM from versions.sh
PLATFORM_CMAKE_FLAGS=""
case "${PLATFORM:-}" in
    NVIDIA_DGX_Spark_P4242)
        ;;
    *)
        ;;
esac

# Options for build_aerial_sdk.sh before "--" (from versions.sh per PLATFORM)
AERIAL_BUILD_ARGS="${AERIAL_BUILD_FLAGS:-}"
AERIAL_BUILD_ARGS="$(echo "$AERIAL_BUILD_ARGS" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

# Combine: profile + platform + user (BUILD_CMAKE_FLAGS); user can add/override via BUILD_CMAKE_FLAGS
BUILD_CMAKE_FLAGS_COMBINED="${PROFILE_CMAKE_FLAGS} ${PLATFORM_CMAKE_FLAGS} ${BUILD_CMAKE_FLAGS:-}"
BUILD_CMAKE_FLAGS_COMBINED="$(echo "$BUILD_CMAKE_FLAGS_COMBINED" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

# Tag pulled image with a convenient local name
tag_image() {
    local full_image="${AERIAL_REPO}${AERIAL_IMAGE_NAME}:${AERIAL_VERSION_TAG}"
    local local_tag="aerial-cuda-accelerated-ran:latest"
    echo ""
    echo "Tagging image for local convenience..."
    echo "  Source: $full_image"
    echo "  Tag:    $local_tag"
    execute_or_die "docker tag $full_image $local_tag"
}


echo "=============================================="
echo "Aerial SDK Quickstart"
echo "=============================================="
echo "SDK Root (host):      $cuBB_SDK_HOST"
echo "SDK Root (container): $cuBB_SDK"
echo "Container Tag:        $AERIAL_VERSION_TAG"
echo "Build Profile:        ${PROFILE:-oai.conf}"
echo "Build Preset:         $BUILD_PRESET"
echo "Profile CMake flags:  ${PROFILE_CMAKE_FLAGS:-<none>}"
echo "Platform CMake flags: ${PLATFORM_CMAKE_FLAGS:-<none>}"
echo "User BUILD_CMAKE_FLAGS: ${BUILD_CMAKE_FLAGS:-<none>}"
echo "All CMake flags:      $BUILD_CMAKE_FLAGS_COMBINED"
echo "Aerial build args:    ${AERIAL_BUILD_ARGS:-<none>}"
echo "=============================================="
echo ""

# build_aerial_sdk.sh: --preset, then AERIAL_BUILD_FLAGS (script options), then --, then CMake flags
BUILD_CMD="\${cuBB_SDK}/testBenches/phase4_test_scripts/build_aerial_sdk.sh --preset $BUILD_PRESET"
if [[ -n "$AERIAL_BUILD_ARGS" ]]; then
    BUILD_CMD="$BUILD_CMD $AERIAL_BUILD_ARGS"
fi
if [[ -n "$BUILD_CMAKE_FLAGS_COMBINED" ]]; then
    BUILD_CMD="$BUILD_CMD -- $BUILD_CMAKE_FLAGS_COMBINED"
fi

if [[ $TEST_AND_EXIT -eq 1 ]]; then
    echo "Building as configured would run:"
    echo "$BUILD_CMD"
    exit 0
fi

if [[ $TAG_IMAGE -eq 1 ]]; then
    tag_image
    exit
fi

if [[ $BUILD_ONLY_MODE -eq 1 ]]; then
    # Run build directly (assumes we're inside container)
    if [[ ! -f "/.dockerenv" ]]; then
        echo "[ERROR] --build-only requires running inside the container"
        echo "Run without --build-only to start the container first"
        exit 1
    fi
    echo "Running build inside container..."
    execute_or_die "$BUILD_CMD"
    exit 0
else
    # Start container and run build
    echo "Starting Aerial container and building SDK..."
    execute_or_die "$cuBB_SDK_HOST/cuPHY-CP/container/run_aerial.sh $BUILD_CMD"
fi
echo "Exporting NVIPC code"
execute_or_die "$cuBB_SDK_HOST/cuPHY-CP/gt_common_libs/pack_nvipc.sh"
tag_image

echo ""
echo "=============================================="
echo "Quickstart complete!"
echo "=============================================="
