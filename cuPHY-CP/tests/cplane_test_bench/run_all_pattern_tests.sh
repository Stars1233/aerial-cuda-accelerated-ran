#!/bin/bash -e

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

set -o pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cuBB_SDK should be set, or derive it
if [ -z "$cuBB_SDK" ]; then
    cuBB_SDK="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    export cuBB_SDK
fi

BUILD_DIR=${BUILD_DIR:-build.perf.$(uname -m)}
CONFIG_DIR=${CUBB_HOME:-$cuBB_SDK}
TESTVECTOR_DIR=${TESTVECTOR_DIR:-/mnt/cicd_tvs/develop/GPU_test_input}

# Paths
TEST_PATTERNS_4T4R="${SCRIPT_DIR}/test_patterns_4t4r.csv"
TEST_PATTERNS_MMIMO="${SCRIPT_DIR}/test_patterns_mmimo.csv"
TEST_PATTERNS_NRSIM="${SCRIPT_DIR}/test_patterns_nrsim.csv"
CPLANE_TEST_BENCH="${cuBB_SDK}/${BUILD_DIR}/cuPHY-CP/tests/cplane_test_bench/cplane_test_bench"
COPY_TEST_FILES="${cuBB_SDK}/testBenches/phase4_test_scripts/copy_test_files.sh"
SETUP1_DU="${cuBB_SDK}/testBenches/phase4_test_scripts/setup1_DU.sh"
SETUP2_RU="${cuBB_SDK}/testBenches/phase4_test_scripts/setup2_RU.sh"
TEST_CONFIG="${cuBB_SDK}/testBenches/phase4_test_scripts/test_config.sh"
TEST_CONFIG_NRSIM="${cuBB_SDK}/testBenches/phase4_test_scripts/test_config_nrSim.sh"
TEST_CONFIG_SUMMARY="${CONFIG_DIR}/testBenches/phase4_test_scripts/test_config_summary.sh"

# Log file
LOG_FILE="/tmp/cplane_tb_results_$(date +%Y%m%d_%H%M%S).log"

# Config files to restore during cleanup
CONFIG_FILES=(
cuPHY-CP/cuphycontroller/config/cuphycontroller_F08_CG1.yaml
cuPHY-CP/cuphycontroller/config/l2_adapter_config_F08_CG1.yaml
cuPHY-CP/cuphycontroller/config/cuphycontroller_nrSim_SCF_CG1.yaml
cuPHY-CP/cuphycontroller/config/l2_adapter_config_nrSim_SCF_CG1.yaml
cuPHY-CP/ru-emulator/config/config.yaml
cuPHY-CP/testMAC/testMAC/test_mac_config.yaml
cuPHY/nvlog/config/nvlog_config.yaml
testVectors/cuPhyChEstCoeffs.h5
)

# Arrays to track results
declare -a FAILED_TESTS
declare -a PASSED_TESTS

# Flags for controlling test execution
RUN_4T4R=false
RUN_MMIMO=false
RUN_NRSIM=false
DO_BUILD=false
FORCE_CLEANUP=false
SKIP_COPY=false
ENABLE_PCAP=false

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run C-Plane test bench pattern tests for 4T4R, mMIMO, and/or nrSim configurations.

OPTIONS:
    -4, --4t4r          Run only 4T4R pattern tests
    -m, --mmimo         Run only mMIMO pattern tests
    -n, --nrsim         Run only nrSim pattern tests (90xxx cases)
    -b, --build         Build the project before running tests
    -c, --cleanup       Force cleanup before running tests
    --enable-pcap       Enable PCAP capture for each pattern
    --skip-copy         Skip copying test vector files
    -h, --help          Show this help message

EXAMPLES:
    # Run all phases (4T4R + nrSim), with cleanup between phases
    $0

    # Run only 4T4R patterns
    $0 --4t4r

    # Run only mMIMO patterns
    $0 --mmimo

    # Run only nrSim patterns
    $0 --nrsim

    # Build and run all phases with PCAP capture
    $0 --build --enable-pcap

    # Build, cleanup, and run only 4T4R
    $0 --4t4r --build --cleanup

NOTES:
    - If no phase flag is specified, 4T4R, mMIMO and nrSim tests are run in order, with cleanup between.
    - Test patterns are read from:
      * 4T4R:  ${SCRIPT_DIR}/test_patterns_4t4r.csv
      * mMIMO: ${SCRIPT_DIR}/test_patterns_mmimo.csv
      * nrSim: ${SCRIPT_DIR}/test_patterns_nrsim.csv
    - Config generation uses setup1_DU.sh, setup2_RU.sh, test_config.sh (4T4R) /
      test_config_nrSim.sh (nrSim) — same helpers as DLC test bench.

EOF
    exit 0
}

# Logging functions
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; echo "[INFO] $1" >> "$LOG_FILE"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; echo "[PASS] $1" >> "$LOG_FILE"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $1"; echo "[FAIL] $1" >> "$LOG_FILE"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; echo "[WARN] $1" >> "$LOG_FILE"; }

build() {
    cd $cuBB_SDK
    $cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh --preset perf --targets cplane_test_bench
}

# Read patterns from CSV (skip comments and empty lines)
read_patterns() {
    local csv_file="$1"
    if [ ! -f "$csv_file" ]; then
        log_error "Pattern file not found: $csv_file"
        return 1
    fi
    while IFS= read -r pattern || [ -n "$pattern" ]; do
        [[ "$pattern" =~ ^[[:space:]]*# ]] && continue
        pattern="${pattern#"${pattern%%[![:space:]]*}"}"
        pattern="${pattern%"${pattern##*[![:space:]]}"}"
        [[ -z "$pattern" ]] && continue
        echo "$pattern"
    done < "$csv_file"
}

# Run test for a single 4T4R pattern
run_4t4r_pattern_test() {
    local pattern_num="$1"
    local test_name="CPlane_4T4R_${pattern_num}"

    log_info "========================================"
    log_info "Testing 4T4R: Pattern ${pattern_num}"
    log_info "========================================"

    # Step 1: Copy test files
    if [ "$SKIP_COPY" = false ]; then
        log_info "Step 1: Copying test files for pattern ${pattern_num}..."
        if ! "$COPY_TEST_FILES" --src $TESTVECTOR_DIR --dst $CONFIG_DIR/testVectors "$pattern_num" --max_cells 1 >> "$LOG_FILE" 2>&1; then
            log_error "Failed to copy test files for pattern ${pattern_num}"
            FAILED_TESTS+=("$test_name: copy_test_files failed")
            return 1
        fi
    else
        log_info "Step 1: Skipping copy_test_files (--skip-copy)"
    fi

    # Step 2: Setup DU
    log_info "Step 2: Running setup1_DU.sh..."
    if ! "$SETUP1_DU" --config_dir=$CONFIG_DIR --ru-host-type=_CG1 >> "$LOG_FILE" 2>&1; then
        log_error "setup1_DU.sh failed for pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: setup1_DU failed")
        return 1
    fi

    # Step 3: Setup RU
    log_info "Step 3: Running setup2_RU.sh..."
    if ! "$SETUP2_RU" --config_dir=$CONFIG_DIR >> "$LOG_FILE" 2>&1; then
        log_error "setup2_RU.sh failed for pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: setup2_RU failed")
        return 1
    fi

    # Step 4: Test config
    log_info "Step 4: Running test_config.sh..."
    if ! "$TEST_CONFIG" --config_dir=$CONFIG_DIR "$pattern_num" -c 1 --dlc-tb=1 >> "$LOG_FILE" 2>&1; then
        log_error "test_config.sh failed for pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: test_config failed")
        return 1
    fi

    # Step 5: Run C-Plane test bench
    log_info "Step 5: Running cplane_test_bench for pattern ${pattern_num}..."
    local EXTRA_ARGS=""
    if [ "$ENABLE_PCAP" = true ]; then
        EXTRA_ARGS="--enable-pcap --pcap-file cptb_${pattern_num}.pcap"
    fi

    if sudo -E "$CPLANE_TEST_BENCH" -p "$pattern_num" $EXTRA_ARGS >> "$LOG_FILE" 2>&1; then
        log_success "Pattern ${pattern_num} (4T4R) PASSED"
        PASSED_TESTS+=("$test_name")
        return 0
    else
        local ret=$?
        log_error "cplane_test_bench failed for pattern ${pattern_num} (exit code: $ret)"
        FAILED_TESTS+=("$test_name: cplane_test_bench failed (exit code: $ret)")
        return 1
    fi
}

# Run test for a single mMIMO pattern
run_mmimo_pattern_test() {
    local pattern_num="$1"
    local test_name="CPlane_mMIMO_${pattern_num}"

    log_info "========================================"
    log_info "Testing mMIMO: Pattern ${pattern_num}"
    log_info "========================================"

    # Step 1: Setup DU with mMIMO flag (-m 1)
    log_info "Step 1: Running setup1_DU.sh -m 1..."
    if ! "$SETUP1_DU" --config_dir=$CONFIG_DIR -m 1 --ru-host-type=_CG1 >> "$LOG_FILE" 2>&1; then
        log_error "setup1_DU.sh -m 1 failed for pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: setup1_DU failed")
        return 1
    fi

    # Step 2: Setup RU
    log_info "Step 2: Running setup2_RU.sh..."
    if ! "$SETUP2_RU" --config_dir=$CONFIG_DIR >> "$LOG_FILE" 2>&1; then
        log_error "setup2_RU.sh failed for pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: setup2_RU failed")
        return 1
    fi

    # Step 3: Test config (modcomp: -o 4)
    log_info "Step 3: Running test_config.sh (modcomp)..."
    if ! "$TEST_CONFIG" --config_dir=$CONFIG_DIR "$pattern_num" -c 1 -o 4 --dlc-tb=1 >> "$LOG_FILE" 2>&1; then
        log_error "test_config.sh failed for pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: test_config failed")
        return 1
    fi

    # Step 4: Copy test files
    if [ "$SKIP_COPY" = false ]; then
        log_info "Step 4: Copying test files for pattern ${pattern_num}..."
        if ! "$COPY_TEST_FILES" --src $TESTVECTOR_DIR --dst $CONFIG_DIR/testVectors "$pattern_num" --max_cells 1 >> "$LOG_FILE" 2>&1; then
            log_error "Failed to copy test files for pattern ${pattern_num}"
            FAILED_TESTS+=("$test_name: copy_test_files failed")
            return 1
        fi
    else
        log_info "Step 4: Skipping copy_test_files (--skip-copy)"
    fi

    # Step 5: Run C-Plane test bench
    log_info "Step 5: Running cplane_test_bench for mMIMO pattern ${pattern_num}..."
    local EXTRA_ARGS=""
    if [ "$ENABLE_PCAP" = true ]; then
        EXTRA_ARGS="--enable-pcap --pcap-file cptb_mmimo_${pattern_num}.pcap"
    fi

    if sudo -E "$CPLANE_TEST_BENCH" -p "$pattern_num" $EXTRA_ARGS >> "$LOG_FILE" 2>&1; then
        log_success "Pattern ${pattern_num} (mMIMO) PASSED"
        PASSED_TESTS+=("$test_name")
        return 0
    else
        local ret=$?
        log_error "cplane_test_bench failed for mMIMO pattern ${pattern_num} (exit code: $ret)"
        FAILED_TESTS+=("$test_name: cplane_test_bench failed (exit code: $ret)")
        return 1
    fi
}

# Run test for a single nrSim pattern (90xxx)
run_nrsim_pattern_test() {
    local pattern_num="$1"
    local test_name="CPlane_nrSim_${pattern_num}"

    log_info "========================================"
    log_info "Testing nrSim: Pattern ${pattern_num}"
    log_info "========================================"

    # Step 1: Copy test files
    # (Must run before setup1_DU.sh because nrSim's setup1_DU invokes
    # auto_AllConfig.py, which scans $CONFIG_DIR/testVectors/ for the
    # pattern's HDF5 files and exits if none are found.)
    if [ "$SKIP_COPY" = false ]; then
        log_info "Step 1: Copying test files for nrSim pattern ${pattern_num}..."
        if ! "$COPY_TEST_FILES" --src $TESTVECTOR_DIR --dst $CONFIG_DIR/testVectors "$pattern_num" --max_cells 1 >> "$LOG_FILE" 2>&1; then
            log_error "Failed to copy test files for nrSim pattern ${pattern_num}"
            FAILED_TESTS+=("$test_name: copy_test_files failed")
            return 1
        fi
    else
        log_info "Step 1: Skipping copy_test_files (--skip-copy)"
    fi

    # Step 2: Setup DU with nrSim controller mode
    log_info "Step 2: Running setup1_DU.sh -y nrSim_SCF_CG1_${pattern_num} --ru-host-type=_CG1..."
    if ! "$SETUP1_DU" --config_dir=$CONFIG_DIR -y "nrSim_SCF_CG1_${pattern_num}" --ru-host-type=_CG1 >> "$LOG_FILE" 2>&1; then
        log_error "setup1_DU.sh failed for nrSim pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: setup1_DU failed")
        return 1
    fi

    # Step 3: Setup RU
    log_info "Step 3: Running setup2_RU.sh..."
    if ! "$SETUP2_RU" --config_dir=$CONFIG_DIR >> "$LOG_FILE" 2>&1; then
        log_error "setup2_RU.sh failed for nrSim pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: setup2_RU failed")
        return 1
    fi

    # Step 4: Test config (nrSim-specific)
    log_info "Step 4: Running test_config_nrSim.sh..."
    if ! "$TEST_CONFIG_NRSIM" --config_dir=$CONFIG_DIR --dlc-tb=1 >> "$LOG_FILE" 2>&1; then
        log_error "test_config_nrSim.sh failed for pattern ${pattern_num}"
        FAILED_TESTS+=("$test_name: test_config_nrSim failed")
        return 1
    fi

    # Step 5: Run C-Plane test bench
    log_info "Step 5: Running cplane_test_bench for nrSim pattern ${pattern_num}..."
    local EXTRA_ARGS=""
    if [ "$ENABLE_PCAP" = true ]; then
        EXTRA_ARGS="--enable-pcap --pcap-file cptb_nrsim_${pattern_num}.pcap"
    fi

    if sudo -E "$CPLANE_TEST_BENCH" -p "$pattern_num" $EXTRA_ARGS >> "$LOG_FILE" 2>&1; then
        log_success "Pattern ${pattern_num} (nrSim) PASSED"
        PASSED_TESTS+=("$test_name")
        return 0
    else
        local ret=$?
        log_error "cplane_test_bench failed for nrSim pattern ${pattern_num} (exit code: $ret)"
        FAILED_TESTS+=("$test_name: cplane_test_bench failed (exit code: $ret)")
        return 1
    fi
}

# Phase wrappers — read pattern CSV and iterate
run_4t4r_tests() {
    log_info "========================================="
    log_info "PHASE 1: 4T4R Pattern Tests"
    log_info "========================================="

    readarray -t patterns < <(read_patterns "$TEST_PATTERNS_4T4R")
    if [ ${#patterns[@]} -eq 0 ]; then
        log_warning "No 4T4R patterns found in $TEST_PATTERNS_4T4R"
        return
    fi
    log_info "Found ${#patterns[@]} 4T4R patterns: ${patterns[*]}"
    log_info ""
    for pattern in "${patterns[@]}"; do
        run_4t4r_pattern_test "$pattern"
        echo "" ; echo "" >> "$LOG_FILE"
    done
}

run_mmimo_tests() {
    log_info "========================================="
    log_info "PHASE: mMIMO Pattern Tests"
    log_info "========================================="

    readarray -t patterns < <(read_patterns "$TEST_PATTERNS_MMIMO")
    if [ ${#patterns[@]} -eq 0 ]; then
        log_warning "No mMIMO patterns found in $TEST_PATTERNS_MMIMO"
        return
    fi
    log_info "Found ${#patterns[@]} mMIMO patterns: ${patterns[*]}"
    log_info ""
    for pattern in "${patterns[@]}"; do
        run_mmimo_pattern_test "$pattern"
        echo "" ; echo "" >> "$LOG_FILE"
    done
}

run_nrsim_tests() {
    log_info "========================================="
    log_info "PHASE: nrSim Pattern Tests"
    log_info "========================================="

    readarray -t patterns < <(read_patterns "$TEST_PATTERNS_NRSIM")
    if [ ${#patterns[@]} -eq 0 ]; then
        log_warning "No nrSim patterns found in $TEST_PATTERNS_NRSIM"
        return
    fi
    log_info "Found ${#patterns[@]} nrSim patterns: ${patterns[*]}"
    log_info ""
    for pattern in "${patterns[@]}"; do
        run_nrsim_pattern_test "$pattern"
        echo "" ; echo "" >> "$LOG_FILE"
    done
}

# Cleanup: restore config files
perform_cleanup() {
    local cleanup_label="${1:-Cleanup}"

    log_info "========================================="
    log_info "$cleanup_label"
    log_info "========================================="

    if [[ "$CONFIG_DIR" != "$cuBB_SDK" ]]; then
        for file in ${CONFIG_FILES[@]}; do
            mkdir -p $(dirname $CONFIG_DIR/$file)
            cp $cuBB_SDK/$file $CONFIG_DIR/$file
        done
        mkdir -p $(dirname $TEST_CONFIG_SUMMARY)
    else
        log_info "Running git restore..."
        if ! git restore "${CONFIG_FILES[@]/#/$cuBB_SDK/}" >> "$LOG_FILE" 2>&1; then
            log_warning "git restore failed or had warnings"
        fi
    fi

    if [ -f "$TEST_CONFIG_SUMMARY" ]; then
        log_info "Deleting $TEST_CONFIG_SUMMARY..."
        rm -f "$TEST_CONFIG_SUMMARY"
    fi

    log_info "$cleanup_label complete"
    log_info ""
}

# Print summary
print_summary() {
    echo ""
    echo "" >> "$LOG_FILE"
    log_info "========================================"
    log_info "TEST SUMMARY"
    log_info "========================================"

    local total_tests=$((${#PASSED_TESTS[@]} + ${#FAILED_TESTS[@]}))
    log_info "Total tests run: $total_tests"
    log_success "Passed: ${#PASSED_TESTS[@]}"
    log_error "Failed: ${#FAILED_TESTS[@]}"

    if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
        echo ""
        echo "" >> "$LOG_FILE"
        log_success "Passed tests:"
        for test in "${PASSED_TESTS[@]}"; do
            echo -e "  ${GREEN}✓${NC} $test"
            echo "  ✓ $test" >> "$LOG_FILE"
        done
    fi

    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo ""
        echo "" >> "$LOG_FILE"
        log_error "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo -e "  ${RED}✗${NC} $test"
            echo "  ✗ $test" >> "$LOG_FILE"
        done
        echo ""
        echo "" >> "$LOG_FILE"
        log_error "Please check log file for details: $LOG_FILE"
        return 1
    fi

    echo ""
    echo "" >> "$LOG_FILE"
    log_success "All tests passed!"
    return 0
}

# Main
main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -4|--4t4r)        RUN_4T4R=true; shift ;;
            -m|--mmimo)       RUN_MMIMO=true; shift ;;
            -n|--nrsim)       RUN_NRSIM=true; shift ;;
            -b|--build)       DO_BUILD=true; shift ;;
            -c|--cleanup)     FORCE_CLEANUP=true; shift ;;
            --enable-pcap)    ENABLE_PCAP=true; shift ;;
            --skip-copy)      SKIP_COPY=true; shift ;;
            -h|--help)        show_usage ;;
            *)
                echo -e "${RED}Error: Unknown option: $1${NC}"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done

    # If no phase flags specified, run 4T4R, mMIMO and nrSim with cleanup between.
    if [ "$RUN_4T4R" = false ] && [ "$RUN_MMIMO" = false ] && [ "$RUN_NRSIM" = false ]; then
        FORCE_CLEANUP=true
        RUN_4T4R=true
        RUN_MMIMO=true
        RUN_NRSIM=true
    fi

    log_info "========================================="
    log_info "C-Plane Test Bench - Pattern Testing"
    log_info "========================================="
    log_info "Start time: $(date)"
    log_info "cuBB_SDK: $cuBB_SDK"
    log_info "Log file: $LOG_FILE"
    log_info "Test Configuration:"
    log_info "  - Run 4T4R tests:   $RUN_4T4R"
    log_info "  - Run mMIMO tests:  $RUN_MMIMO"
    log_info "  - Run nrSim tests:  $RUN_NRSIM"
    log_info "  - Build project:    $DO_BUILD"
    log_info "  - Force cleanup:    $FORCE_CLEANUP"
    log_info "  - PCAP capture:     $ENABLE_PCAP"
    log_info ""

    if [ "$DO_BUILD" = true ]; then
        if ! build; then
            log_error "build failed"
            exit 1
        fi
    fi

    if [ ! -x "$CPLANE_TEST_BENCH" ]; then
        log_error "cplane_test_bench not found or not executable: $CPLANE_TEST_BENCH"
        exit 1
    fi

    cd "$cuBB_SDK" || exit 1
    log_info "Working directory: $(pwd)"
    log_info ""

    # Phase 1: 4T4R
    if [ "$RUN_4T4R" = true ]; then
        if [ "$FORCE_CLEANUP" = true ]; then
            perform_cleanup "Cleanup before 4T4R tests"
        fi
        run_4t4r_tests
    else
        log_info "========================================="
        log_info "Skipping 4T4R Pattern Tests"
        log_info "========================================="
        log_info ""
    fi

    # Phase 2: mMIMO
    if [ "$RUN_MMIMO" = true ]; then
        if [ "$FORCE_CLEANUP" = true ]; then
            perform_cleanup "Cleanup before mMIMO tests"
        fi
        run_mmimo_tests
    else
        log_info "========================================="
        log_info "Skipping mMIMO Pattern Tests"
        log_info "========================================="
        log_info ""
    fi

    # Phase 3: nrSim
    if [ "$RUN_NRSIM" = true ]; then
        if [ "$FORCE_CLEANUP" = true ]; then
            perform_cleanup "Cleanup before nrSim tests"
        fi
        run_nrsim_tests
    else
        log_info "========================================="
        log_info "Skipping nrSim Pattern Tests"
        log_info "========================================="
        log_info ""
    fi

    print_summary
    local summary_result=$?

    log_info ""
    log_info "End time: $(date)"
    log_info "Log file: $LOG_FILE"

    exit $summary_result
}

main "$@"
