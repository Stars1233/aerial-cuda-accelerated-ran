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

sanitizer_tool_options=("N/A" "memcheck" "racecheck" "synccheck" "initcheck")

# Update the function usage for UNIT_TEST_BLOCK_TO_RUN_FIRST when adding a block to unit_test_blocks
unit_test_blocks=("runPUSCH" "runPDSCH" "runPDCCH" "runPUCCH" "runSSB" "runCSIRS" "runPRACH" "runCHEST" "runCHEQ" "runBWC" "runSRS" "runSRSTX" "runSimplex" "runDeRateMatch" "runComponentTests" "runCSIRSRX")

LOCKFILE="/var/lock/cuphy_unit_test.lock"

LOGFILE="run.log"
if [ -z "$ENABLE_STREAMS" ]; then
    ENABLE_STREAMS=1
fi

EXITCODE=0

function usage {
    cat <<'EOF'
Usage:
  ./cuphy_unit_test.sh <Base_TV_path> <GPU> <PUSCH_CB_ERROR_CHECK> <RUN_COMPONENT_TESTS> <RUN_COMPUTE_SANITIZER> <PYTHON_BIN> <UNIT_TEST_BLOCK_TO_RUN_FIRST> [-b <BUILD_DIR>] [--component-filter <NAME>] [--srs_bfw]

Or with optional flags (recommended):
  ./cuphy_unit_test.sh \
    [-t <Base_TV_path>] \
    [-g <GPU>] \
    [-c <PUSCH_CB_ERROR_CHECK>] \
    [-m <RUN_COMPONENT_TESTS>] \
    [-s <RUN_COMPUTE_SANITIZER>] \
    [-p <PYTHON_BIN>] \
    [-f <UNIT_TEST_BLOCK_TO_RUN_FIRST>] \
    [-b <BUILD_DIR>] \
    [-o <ONLY_BLOCK>] \
    [--component-filter <NAME>] \
    [--srs_bfw]

Parameters                    Default Value   Description
  Base_TV_path                    ./          Path to test vectors
    -t, --tv-path                             Alternative flag for Base_TV_path

  GPU                             0           CUDA device to execute on (sets CUDA_VISIBLE_DEVICES)
    -g, --gpu                                 Alternative flag for GPU

  PUSCH_CB_ERROR_CHECK            1           Enable (1) / disable (0) PUSCH CB error checks
    -c, --cb-error-check                      Alternative flag for PUSCH_CB_ERROR_CHECK

  RUN_COMPONENT_TESTS             0           Enable (1) / disable (0) component-level tests
    -m, --component-tests                     Alternative flag for RUN_COMPONENT_TESTS

  RUN_COMPUTE_SANITIZER           1           Bitmask of compute-sanitizer tools (0–15)
                                               memcheck   = 1
                                               racecheck  = 2
                                               synccheck  = 4
                                               initcheck  = 8
                                              Examples: 0=off, 1=memcheck, 3=memcheck+racecheck, 5=memcheck+synccheck, 15=all
    -s, --sanitizer                           Alternative flag for RUN_COMPUTE_SANITIZER

  PYTHON_BIN                    python3       Python to run helper scripts with
    -p, --python                              Alternative flag for PYTHON_BIN
                                              (Note: mypy/interrogate/pytest may be required by some checks)

  UNIT_TEST_BLOCK_TO_RUN_FIRST    0           Block index to run first (reorders; does NOT limit others)
    -f, --first-block                         Alternative flag for UNIT_TEST_BLOCK_TO_RUN_FIRST

  -b, --build-dir <BUILD_DIR>                 Custom build directory (default: $CUPHY_ROOT/../build/cuPHY)
  -o, --only-block                            Run only the specified block (see indices below)
  --component-filter <NAME>                   Run only selected component test group(s); repeat or comma-separate values
                                              Valid values: pdsch, ldpc, polar, gtests, hdf5
                                              If specified, component tests are enabled automatically
  --srs_bfw                                   Run only SRS+BFW tests

Environment:
  ENABLE_STREAMS                  1           1 = run stream mode where available; 0 = prefer graph-only where coded

Test block indices:
EOF

    # Dynamically print the block map from the array to avoid drift.
    local idx=0
    for name in "${unit_test_blocks[@]}"; do
        printf "  %2d  %s\n" "$idx" "$name"
        idx=$((idx + 1))
    done

    cat <<'EOF'

Examples:
  Run only PUSCH (-o 0) on GPU 0, with memcheck:
    ./cuphy_unit_test.sh -o 0 -g 0 -s 1 -t /path/to/TVs -b /abs/path/build/cuPHY

  Run LDPC and Polar component test groups enabled and all other test blocks enabled (default):
    ./cuphy_unit_test.sh --component-filter ldpc --component-filter polar -t /path/to/TVs

  Run only component tests (LDPC and Polar groups):
    ./cuphy_unit_test.sh -o 14 --component-filter ldpc --component-filter polar -t /path/to/TVs

  Run PDSCH first but keep others:
    ./cuphy_unit_test.sh -f 1 -t ./TVs

Notes:
  - Values for -s/--sanitizer outside 0–15 are invalid.
  - Positional arguments are supported for backward compatibility; flags are preferred.
EOF
    exit "${1:-0}"
}

function add_component_test_filter {
    local candidate="$1"
    local existing=""
    for existing in "${COMPONENT_TEST_FILTERS[@]}"; do
        if [ "$existing" = "$candidate" ]; then
            return
        fi
    done
    COMPONENT_TEST_FILTERS+=("$candidate")
}

function should_run_component_test_group {
    local group="$1"
    local selected=""
    if [ ${#COMPONENT_TEST_FILTERS[@]} -eq 0 ]; then
        return 0
    fi
    for selected in "${COMPONENT_TEST_FILTERS[@]}"; do
        if [ "$selected" = "$group" ]; then
            return 0
        fi
    done
    return 1
}

function ensure_component_test_block_is_selected {
    local block_name=""
    for block_name in "${unit_test_blocks[@]}"; do
        if [ "$block_name" = "runComponentTests" ]; then
            return
        fi
    done
    unit_test_blocks+=("runComponentTests")
}

# Sleep loop while lockfile is present
function checkwaitfile {
    while [ -f "$LOCKFILE" ]; do
        sleep 1
    done
}

function checkBinFile {
    local bin="$1"
    # Pick label either from user arg or caller function name
    local label="${2:-${FUNCNAME[1]:-unknown}}"
    if [[ ! -x "$bin" ]]; then
        echo "SKIP $label: binary not found: $bin"
    fi
}

function checkmem {
    # Check exit code of process that was the input to tee
    pid_errval=${PIPESTATUS[0]}
    if [ $pid_errval -ne 0 ]; then
        echo "Process exited with non-zero exit code $pid_errval"
        return 1
    fi

    # globals on purpose; later code reads them
    errval=0
    warnval=0

    # When sanitizer is run with  --error-exitcode 1  so it returns 1 on an error, this part of the code may not be reached.
    # FIXME some of the regex and other code is unreachable and needs to be checked
    if [ $sanitizer_idx -ne 0 ]; then
        if [ $sanitizer_idx -eq 2 ]; then # racecheck has differnet log format
            #========= RACECHECK SUMMARY: 2 hazards displayed (0 errors, 2 warnings)
            errval=$(grep 'RACECHECK SUMMARY' "$1" | sed -e 's/.*(\([0-9]*\) errors.*/\1/')
            warnval=$(grep 'RACECHECK SUMMARY' "$1" | grep -oP '\d+(?= warnings)')
        else
            errval=$(grep 'ERROR SUMMARY' "$1" | sed -e 's/========= ERROR SUMMARY: \(.*\)error\(.*\)/\1/')
        fi
        if [ -z "$errval" ]; then
            echo "Exit on error found"
            return 1
        fi
        if [ $errval -ne 0 ]; then
            echo "${sanitizer_tool_options[$sanitizer_idx]} error count="$errval
            return $errval
        else
            echo ""
        fi
    fi
}

function checkPuschResults {
    # Check exit code of process that was the input to tee
    pid_errval=${PIPESTATUS[0]}
    if [ $pid_errval -ne 0 ]; then
        echo "Process exited with non-zero exit code $pid_errval"
        return 1
    fi

    # Check for CHEST failure by searching for "ChEst" and "FAILED" in the same line
    if [[ $(grep -Eo 'ChEst.*FAILED' "$1") ]]; then
        chestFailure_string=$(grep -Eo 'ChEst.*FAILED' "$1")
        readarray -t chestFailure_array <<<"$chestFailure_string"
        echo "PUSCH CHEST failure detected, count ${#chestFailure_array[@]}"
        return ${#chestFailure_array[@]}
    elif [[ $(grep -Eo 'ChEst.*PASSED' "$1") ]]; then
        return 0
    fi

    # Check for CHEQCOEFF failure by searching for "ChEqCoef" and "FAILED" in the same line
    if [[ $(grep -Eo 'ChEqCoef.*FAILED' "$1") ]]; then
        chEqCoefFailure_string=$(grep -Eo 'ChEqCoef.*FAILED' "$1")
        readarray -t chEqCoefFailure_array <<<"$chEqCoefFailure_string"
        echo "PUSCH ChEqCoef failure detected, count ${#chEqCoefFailure_array[@]}"
        return ${#chEqCoefFailure_array[@]}
    elif [[ $(grep -Eo 'ChEqCoef.*PASSED' "$1") ]]; then
        return 0
    fi

    # Check for SOFTDEMAPPER failure by searching for "SoftDemapper" and "FAILED" in the same line
    if [[ $(grep -Eo 'SoftDemapper.*FAILED' "$1") ]]; then
        softDemapperFailure_string=$(grep -Eo 'SoftDemapper.*FAILED' "$1")
        readarray -t softDemapperFailure_array <<<"$softDemapperFailure_string"
        echo "PUSCH SoftDemapper failure detected, count ${#softDemapperFailure_array[@]}"
        return ${#softDemapperFailure_array[@]}
    elif [[ $(grep -Eo 'SoftDemapper.*PASSED' "$1") ]]; then
        return 0
    fi

    # Check for non-zero BLER for all TBs in all cells and for UCI too
    errCBs_string=$(grep -Po 'Error CBs\s*\K\d+' "$1")
    #errCBs_string=$(grep -Po '^.*TbIdx.*Error CBs\s*\K\d+' "$1") # Use if you don't want to check UCI BLER too.
    readarray -t errCBs_array <<<"$errCBs_string"
    for errCBs in "${errCBs_array[@]}"; do
        if [ "$errCBs" -ne "0" ]; then
            echo "Non-zero BLER detected for at least one TB in a cell or UCI - Error Code Blocks = "$errCBs
            #return $errCBs
        fi
    done
    # Check for mismatched CBs based on cbErr
    mismatchedCBs_string=$(grep -Po 'Mismatched CBs\s*\K\d+' "$1")
    readarray -t mismatchedCBs_array <<<"$mismatchedCBs_string"
    for mismatchedCBs in "${mismatchedCBs_array[@]}"; do
        if [ "$mismatchedCBs" -ne "0" ]; then
            echo "Mismatch detected for at least one TB in a cell or UCI - Mismatched Code Blocks = "$mismatchedCBs
            return $mismatchedCBs
        fi
    done
    # Check for MismatchedCRC CBs against cbErr
    mismatchedCrcCBs_string=$(grep -Po 'MismatchedCRC CBs\s*\K\d+' "$1")
    readarray -t mismatchedCrcCBs_array <<<"$mismatchedCrcCBs_string"
    for mismatchedCrcCBs in "${mismatchedCrcCBs_array[@]}"; do
        if [ "$mismatchedCrcCBs" -ne "0" ]; then
            echo "MismatchedCRC detected for at least one CB in a cell or UCI - MismatchedCRC Code Blocks = "$mismatchedCrcCBs
            return $mismatchedCrcCBs
        fi
    done
    # Check for MismatchedCRC TBs against tbErr
    mismatchedCrcTBs_string=$(grep -Po 'MismatchedCRC TBs\s*\K\d+' "$1")
    readarray -t mismatchedCrcTBs_array <<<"$mismatchedCrcTBs_string"
    for mismatchedCrcTBs in "${mismatchedCrcTBs_array[@]}"; do
        if [ "$mismatchedCrcTBs" -ne "0" ]; then
            echo "MismatchedCRC detected for at least one TB in a cell or UCI - MismatchedCRC Transport Blocks = "$mismatchedCrcTBs
            return $mismatchedCrcTBs
        fi
    done

    # Check for metric mismatches by searching for "ERROR:" and "mismatch" in the same line
    if [[ $(grep -Eo 'ERROR:.*mismatch' "$1") ]]; then
        metricMismatch_string=$(grep -Eo 'ERROR:.*mismatch' "$1")
        readarray -t metricMismatch_array <<<"$metricMismatch_string"
        echo "PUSCH metric mismatches detected, count ${#metricMismatch_array[@]}"
        return ${#metricMismatch_array[@]}
    fi

    # Check for metric mismatches by searching for "ERR" and "mismatch" in the same line
    if [[ $(grep -Eo 'ERR.*mismatch' "$1") ]]; then
        metricMismatch_string=$(grep -Eo 'ERR.*mismatch' "$1")
        readarray -t metricMismatch_array <<<"$metricMismatch_string"
        echo "PUSCH metric mismatches detected, count ${#metricMismatch_array[@]}"
        return ${#metricMismatch_array[@]}
    fi
}

function checkPucchMismatch {
    # Check for mismatches
    mismatches_string=$(grep -Po 'found\s*\K\d+' "$1")
    readarray -t mismatches_array <<<"$mismatches_string"
    for mismatches in "${mismatches_array[@]}"; do
        if [ "$mismatches" -ne 0 ]; then
            echo "Mismatch detected for PUCCH = "$mismatches
            return $mismatches
        fi
    done
    echo ""
}

function checkSimplexMismatch {
    # Check for mismatches
    mismatches_string=$(grep -Po 'found\s*\K\d+' "$1")
    readarray -t mismatches_array <<<"$mismatches_string"
    for mismatches in "${mismatches_array[@]}"; do
        if [ "$mismatches" -ne 0 ]; then
            echo "Mismatch detected for Simplex code = "$mismatches
            return $mismatches
        fi
    done
    echo ""
}

function checkDeRateMatchMismatch {
    # Check for mismatches
    mismatches_string=$(grep -Po 'detected\s*\K\d+\s+mismatches\s*out\s*of\s*\d+\s+rateMatchedLLRs' "$1")

    readarray -t mismatches_array <<<"$mismatches_string"

    for mismatches in "${mismatches_array[@]}"; do # There can be multiple such lines, one per user
        readarray -d ' ' -t user_mismatches <<<"$mismatches"
        if [ "${user_mismatches[0]}" -ne 0 ]; then
            echo "Mismatch detected for PUSCH deRateMatch = "${user_mismatches[0]}
            return ${user_mismatches[0]}
        fi
    done
    echo ""
}

function checktest {
    # Check exit code of process that was the input to tee
    pid_errval=${PIPESTATUS[0]}
    if [ $pid_errval -ne 0 ]; then
        echo "Process exited with non-zero exit code $pid_errval"
        return 1
    fi
}

# argument 1: test command to run
# argument 2: mask of allowed sanitizers
# argument 3: prefix for test command (e.g. setting environment variables before test)
function runtest {
    test_cmd=$1
    test_prefix=''
    sanitizer_idx=0
    sanitizer_arg=$(($2 & $RUN_COMPUTE_SANITIZER))
    if [ $# -ge 3 ]; then
        test_prefix=$3
    fi
    while [ $sanitizer_arg -gt 0 ]; do
        sanitizer_idx=$((sanitizer_idx + 1))
        if (($sanitizer_arg & 1)); then
            case $sanitizer_idx in
            1)
                compute_sanitizer_cmd_prefix="compute-sanitizer --error-exitcode $EXITCODE --tool memcheck --leak-check full"
                ;;
            2)
                compute_sanitizer_cmd_prefix="compute-sanitizer --error-exitcode $EXITCODE --tool racecheck --racecheck-report all"
                ;;
            3)
                compute_sanitizer_cmd_prefix="compute-sanitizer --error-exitcode $EXITCODE --tool synccheck"
                ;;
            4)
                compute_sanitizer_cmd_prefix="compute-sanitizer --error-exitcode $EXITCODE --tool initcheck"
                ;;
            esac
            checkwaitfile
            full_cmd="$test_prefix $compute_sanitizer_cmd_prefix $test_cmd"
            echo "$full_cmd"
            eval $full_cmd | tee "$LOGFILE"
            checkmem $LOGFILE
            if [ -n "$tv" ]; then
                $PYTHON3 $TESTDIR/test_summary.py -t ${sanitizer_tool_options[$sanitizer_idx]} -v $tv -r $errval
                if [ -n "$warnval" ]; then
                    $PYTHON3 $TESTDIR/test_summary.py -t "${sanitizer_tool_options[$sanitizer_idx]} warnings" -v $tv -r $warnval
                fi
            fi
        fi
        sanitizer_arg=$(($sanitizer_arg >> 1))
    done
    # No compute-sanitizer selected
    if [ $sanitizer_idx -eq 0 ]; then
        checkwaitfile
        full_cmd="$test_prefix $test_cmd"
        echo "$full_cmd"
        eval $full_cmd | tee "$LOGFILE"
        checktest "$LOGFILE"
    fi
}

#function runPyAerial {
#  echo Skipping PyAerial tests ...
#
#  echo Running PyAerial static tests ...
#  $cuBB_SDK/pyaerial/scripts/run_static_tests.sh
#
#  echo Running PyAerial unit tests ...
#  $cuBB_SDK/pyaerial/scripts/run_unit_tests.sh
#}

function runSSB {
    bin_file=$CUPHY_EXAMPLES/ss/testSS
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "*SSB_gNB_CUPHY*.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv})
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
            fi
            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1)
            runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
        done
    fi
}

function runPDCCH {
    bin_file=$CUPHY_EXAMPLES/pdcch/embed_pdcch_tf_signal
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "*PDCCH_gNB_CUPHY*.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv})
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
            fi
            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1)
            runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
        done
    fi
}

function runPDSCH {
    #TODO Not all PDSCH + PDSCH component examples might be needed. Can comment some out.
    #TODO Potentially include PDSCH multi-cell, but it needs a YAML file.
    PDSCH_CMDS=(
        "$CUPHY_EXAMPLES/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 0 0 TB_ALIGNMENT_PLACEHOLDER" #PDSCH in non-AAS mode, streams, different TB alignment per TV
        #"$CUPHY_EXAMPLES/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 1 0" #PDSCH in AAS mode, streams
        "$CUPHY_EXAMPLES/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 0 1 TB_ALIGNMENT_PLACEHOLDER" #PDSCH in non-AAS mode, graphs, different TB alignment per TV
        #"$CUPHY_EXAMPLES/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 1 1" #PDSCH in AAS mode, graphs
        #"$CUPHY_EXAMPLES/dl_rate_matching/dl_rate_matching TV_PLACEHOLDER 2"
        #"$CUPHY_EXAMPLES/modulation_mapper/modulation_mapper TV_PLACEHOLDER 2"
        #"$CUPHY_EXAMPLES/pdsch_dmrs/pdsch_dmrs TV_PLACEHOLDER"
    )

    for ((i = 0; i < "${#PDSCH_CMDS[@]}"; i++)); do
        pdsch_cmd=${PDSCH_CMDS[$i]} # Do not modify this as bin_file (see next line) should come first.
        bin_file=$(echo -n $pdsch_cmd | sed -e "s/ .*$//g")
        checkBinFile "$bin_file"
        if [ -f "${bin_file}" ]; then
            for tv in $(find $BASE_TV_PATH \( -name "*PDSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-DS*.h5" -o -name "TV_cuphy_F01-DS*.h5" -o -name "TV_cuphy_V*-DS*.h5" \)); do
                # Pick different TB alignment per TV to increase test coverage for all TVnr_DLMIX_.*_PDSCH_gNB_CUPHY.*h5 or TVnr_.*_PDSCH_gNB_CUPHY_.*h5 TVs;
                # fall back to 8 byte alignment otherwise

                TB_alignment=8 # default byte alignment, unless overwritten below
                tv_num=$(echo $tv | grep "^.*TVnr[_DLMIX]*_[0-9].*PDSCH_gNB_CUPHY.*h5" | sed -e "s|^.*TVnr[_DLMIX]*_[0]*\([0-9]\+\)_PDSCH_gNB_CUPHY.*h5|\1|g")
                if [ -n "$tv_num" ]; then
                    TB_alignment=$((1 << ($tv_num % 6))) # supported TB alignments in bytes are 1, 2, 4, 8, 16, 32
                fi
                current_pdsch_cmd=$(echo -n "$pdsch_cmd" | sed -e "s|TV_PLACEHOLDER|$tv|g" | sed -e "s|TB_ALIGNMENT_PLACEHOLDER|$TB_alignment|g")
                echo "$current_pdsch_cmd"
                runtest "$current_pdsch_cmd" $RUN_COMPUTE_SANITIZER
            done
        fi
    done
}

function runCSIRS {
    bin_file=$CUPHY_EXAMPLES/csi_rs/nzp_csi_rs_test
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "*CSIRS_gNB_CUPHY*.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv})
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
            fi
            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1)
            runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
        done
    fi
}

function runCSIRSRX {
    bin_file=$CUPHY_EXAMPLES/csirs_rx_multi_cell/cuphy_ex_nzp_csirs_rx_multi_cell
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "*CSIRS_UE_CUPHY*.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv})
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
            fi
            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1)
            runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
        done
    fi
}

function runPRACH {
    bin_file=$CUPHY_EXAMPLES/prach_receiver_multi_cell/prach_receiver_multi_cell
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "*PRACH_gNB_CUPHY*.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd="${bin_file} -i ${tv} -r 1 -k"
                echo "$test_cmd"
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER 'CUDA_MEMCHECK_PATCH_MODULE=1'
            fi
            # Graphs mode
            test_cmd="${bin_file} -i ${tv} -r 1 -k -m 1"
            echo "$test_cmd"
            runtest "$test_cmd" $RUN_COMPUTE_SANITIZER 'CUDA_MEMCHECK_PATCH_MODULE=1'
        done
    fi
}

function runPUCCH {
    bin_file=$CUPHY_EXAMPLES/pucch_rx_pipeline/cuphy_ex_pucch_rx_pipeline
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "TVnr_*PUCCH_F?_gNB_CUPHY_s*.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv})
                runtest "$test_cmd" 15
                checkPucchMismatch $LOGFILE
            fi
            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1)
            runtest "$test_cmd" 15
            checkPucchMismatch $LOGFILE
        done
    fi
}

function runCHEST {
    bin_file=$CUPHY_EXAMPLES/ch_est/cuphy_ex_ch_est
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH \( -name "TVnr_*PUSCH_gNB_CUPHY*s0p*.h5" -o -name "TV_cuphy_F14-US*s0p*.h5" -o -name "TV_cuphy_F01-US*s0p*.h5" -o -name "TV_cuphy_V*-US*s0p*.h5" \)); do

            # ignore forceRxZero PUSCH test cases
            if [[ "$tv" == *"7417"* ]]; then
                continue
            fi

            if [[ "$tv" == *"7531"* ]]; then
                continue
            fi

            # ignore low SNR test case
            if [[ "$tv" == *"7489"* ]]; then
                continue
            fi

            test_cmd=$(echo -n ${bin_file} -i ${tv})
            runtest "$test_cmd" 7

            if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                echo "do checkPuschResults $LOGFILE"
                checkPuschResults $LOGFILE 0
            fi
        done
    fi
}

function runCHEQ {
    bin_file=$CUPHY_EXAMPLES/channel_eq/cuphy_ex_channel_eq
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH \( -name "TVnr_*PUSCH_gNB_CUPHY*s0p*.h5" -o -name "TV_cuphy_F14-US*s0p*.h5" -o -name "TV_cuphy_F01-US*s0p*.h5" -o -name "TV_cuphy_V*-US*s0p*.h5" \)); do

            # ignore forceRxZero PUSCH test cases
            if [[ "$tv" == *"7417"* ]]; then
                continue
            fi

            if [[ "$tv" == *"7531"* ]]; then
                continue
            fi

            # ignore low SNR test case
            if [[ "$tv" == *"7489"* ]]; then
                continue
            fi

            ##################################
            # ignore test cases with large validation error
            if [[ "$tv" == *"7410"* ]]; then #ChEq SNR: 57.907 dB
                continue
            fi
            if [[ "$tv" == *"7419"* ]]; then #ChEq SNR: 53.315 dB
                continue
            fi
            if [[ "$tv" == *"7453"* ]]; then #ChEq SNR: 55.498 dB
                continue
            fi
            if [[ "$tv" == *"7472"* ]]; then #ChEq SNR: 63.266 dB
                continue
            fi
            ###################################

            test_cmd=$(echo -n ${bin_file} -i ${tv})
            runtest "$test_cmd" 7

            if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                echo "do checkPuschResults $LOGFILE"
                checkPuschResults $LOGFILE 0
            fi
        done
    fi
}

function runPUSCH {
    bin_file=$CUPHY_EXAMPLES/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        # Streams mode
        if [ $ENABLE_STREAMS -eq 1 ]; then
            for tv in $(find $BASE_TV_PATH \( -name "TVnr_*PUSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-US*.h5" -o -name "TV_cuphy_F01-US*.h5" -o -name "TV_cuphy_V*-US*.h5" \)); do
                test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1 -m 0)
                runtest "$test_cmd" 7
                if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                    echo "do checkPuschResults $LOGFILE"
                    checkPuschResults $LOGFILE 0
                fi
            done
        fi

        # Streams mode + sub-slot
        if [ $ENABLE_STREAMS -eq 1 ]; then
            for tv in $(find $BASE_TV_PATH \( -name "TVnr_*PUSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-US*.h5" -o -name "TV_cuphy_F01-US*.h5" -o -name "TV_cuphy_V*-US*.h5" \)); do
                test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1 -m 2)
                runtest "$test_cmd" 7
                if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                    echo "do checkPuschResults $LOGFILE"
                    checkPuschResults $LOGFILE 0
                fi
            done
        fi

        # Graphs mode
        for tv in $(find $BASE_TV_PATH \( -name "TVnr_*PUSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-US*.h5" -o -name "TV_cuphy_F01-US*.h5" -o -name "TV_cuphy_V*-US*.h5" \)); do
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1 -r 1)
            runtest "$test_cmd" 7

            if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                echo "do checkPuschResults $LOGFILE"
                checkPuschResults $LOGFILE 0
            fi
        done

        # Graphs mode + sub-slot
        for tv in $(find $BASE_TV_PATH \( -name "TVnr_*PUSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-US*.h5" -o -name "TV_cuphy_F01-US*.h5" -o -name "TV_cuphy_V*-US*.h5" \)); do
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 3 -r 1)
            runtest "$test_cmd" 7

            if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                echo "do checkPuschResults $LOGFILE"
                checkPuschResults $LOGFILE 0
            fi
        done

        # HARQ2 cases
        for tv in $(find $BASE_TV_PATH -name "TVnrPUSCH_HARQ2_*_CUPHY_s0*.h5"); do

            # disable PUSCH TC7356 temporarily
            if [[ "$tv" == *"7356"* ]]; then
                continue
            fi

            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1 -H 2)
                runtest "$test_cmd" 7

                if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                    checkPuschResults $LOGFILE
                fi
            fi

            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1 -r 1 -H 2)
            runtest "$test_cmd" 7

            if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                checkPuschResults $LOGFILE
            fi
        done

        # HARQ4 cases
        for tv in $(find $BASE_TV_PATH -name "TVnrPUSCH_HARQ4_*_CUPHY_s0*.h5"); do

            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1 -H 4)
                runtest "$test_cmd" 7

                if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                    checkPuschResults $LOGFILE
                fi
            fi

            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1 -r 1 -H 4)
            runtest "$test_cmd" 7

            if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                checkPuschResults $LOGFILE
            fi
        done

        # UL TTI Bundling cases
        for tv in $(find $BASE_TV_PATH -name "TVnrPUSCH_HARQ4_7898_PUSCH_gNB_CUPHY_s0p0.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1 -H 7)
                runtest "$test_cmd" 7

                if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                    checkPuschResults $LOGFILE
                fi
            fi

            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -m 1 -r 1 -H 7)
            runtest "$test_cmd" 7

            if [ ${PUSCH_CB_ERROR_CHECK} -ne 0 ]; then
                checkPuschResults $LOGFILE
            fi
        done
    fi
}

function runBWC {
    bin_file=$CUPHY_EXAMPLES/bfc/cuphy_ex_bfc
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "TVnr_*_BFW_gNB_CUPHY*.h5"); do

            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1 -m 0 -c)
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
            fi

            # Graphs mode
            test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1 -m 1 -c)
            runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
        done
    fi
}

function runSRS {
    bin_file=$CUPHY_EXAMPLES/srs_rx_pipeline/cuphy_ex_srs_rx_pipeline
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "TVnr_*_SRS_gNB_CUPHY_*.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1)
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
            else #Graphs mode, since SRS is a single node graph, testing with either stream or graph should be good enough
                test_cmd=$(echo -n ${bin_file} -i ${tv} -r 1 -m 1)
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
            fi
        done
    fi
}

function runSRSTX {
    bin_file=$CUPHY_EXAMPLES/srs_tx/cuphy_ex_srs_tx
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "*SRS_UE*_CUPHY*.h5"); do
            test_cmd=$(echo -n ${bin_file} -i ${tv} -k)
            runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
        done
    fi
}

function runSimplex {
    bin_file=$CUPHY_EXAMPLES/simplex_decoder/cuphy_ex_simplex_decoder
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for tv in $(find $BASE_TV_PATH -name "TVnr_*SIMPLEX_gNB_CUPHY_s*.h5"); do
            # Streams mode
            if [ $ENABLE_STREAMS -eq 1 ]; then
                test_cmd=$(echo -n ${bin_file} -i ${tv})
                runtest "$test_cmd" 15
                checkSimplexMismatch $LOGFILE
            fi
        done
    fi
}

function runDeRateMatch {
    bin_file=$CUPHY_EXAMPLES/pusch_rateMatch/cuphy_ex_pusch_rateMatch
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        #limit to a subset of TCs in favor of test runtime
        for tv in $(find $BASE_TV_PATH -name "TVnr_7*PUSCH_gNB_CUPHY*.h5"); do
            test_cmd=$(echo -n ${bin_file} -i ${tv})
            runtest "$test_cmd" 0 #disable compute sanitizer for Phase 1 test
            checkDeRateMatchMismatch $LOGFILE
        done
    fi
}

function runComponentTests {
    if [ $RUN_COMPONENT_TESTS -eq 1 ]; then

        # PDSCH dl_rate_matching, modulation_mapper, pdsch_dmrs, AAS mode
        if should_run_component_test_group "pdsch"; then
            PDSCH_CMDS=(
                #"$CUPHY_EXAMPLES/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 0 0" #PDSCH in non-AAS mode, streams
                "$CUPHY_EXAMPLES/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 1 0 TB_ALIGNMENT_PLACEHOLDER" #PDSCH in AAS mode, streams. different TB alignment per TV
                #"$CUPHY_EXAMPLES/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 0 1" #PDSCH in non-AAS mode, graphs
                "$CUPHY_EXAMPLES/pdsch_tx/cuphy_ex_pdsch_tx TV_PLACEHOLDER 2 1 1 TB_ALIGNMENT_PLACEHOLDER" #PDSCH in AAS mode, graphs, different TB alignment per TV
                "$CUPHY_EXAMPLES/dl_rate_matching/dl_rate_matching TV_PLACEHOLDER 2"
                "$CUPHY_EXAMPLES/modulation_mapper/modulation_mapper TV_PLACEHOLDER 2"
                "$CUPHY_EXAMPLES/pdsch_dmrs/pdsch_dmrs TV_PLACEHOLDER"
            )

            for ((i = 0; i < "${#PDSCH_CMDS[@]}"; i++)); do
                pdsch_cmd=${PDSCH_CMDS[$i]} # Do not modify this as bin_file (see next line) should come first.
                bin_file=$(echo -n $pdsch_cmd | sed -e "s/ .*$//g")
                checkBinFile "$bin_file"
                if [ -f "${bin_file}" ]; then
                    for tv in $(find $BASE_TV_PATH \( -name "*PDSCH_gNB_CUPHY*.h5" -o -name "TV_cuphy_F14-DS*.h5" -o -name "TV_cuphy_F01-DS*.h5" -o -name "TV_cuphy_V*-DS*.h5" \)); do
                        # Pick different TB alignment per TV to increase test coverage for all TVnr_DLMIX_.*_PDSCH_gNB_CUPHY.*h5 or TVnr_.*_PDSCH_gNB_CUPHY_.*h5 TVs;
                        # fall back to 8 byte alignment otherwise
                        TB_alignment=8 # default byte alignment, unless overwritten below
                        tv_num=$(echo $tv | grep "^.*TVnr[_DLMIX]*_[0-9].*PDSCH_gNB_CUPHY.*h5" | sed -e "s|^.*TVnr[_DLMIX]*_[0]*\([0-9]\+\)_PDSCH_gNB_CUPHY.*h5|\1|g")
                        if [ -n "$tv_num" ]; then
                            TB_alignment=$((1 << ($tv_num % 6))) # supported TB alignments in bytes are 1, 2, 4, 8, 16, 32
                        fi
                        current_pdsch_cmd=$(echo -n "$pdsch_cmd" | sed -e "s|TV_PLACEHOLDER|$tv|g" | sed -e "s|TB_ALIGNMENT_PLACEHOLDER|$TB_alignment|g")
                        echo "$current_pdsch_cmd"
                        runtest "$current_pdsch_cmd" $RUN_COMPUTE_SANITIZER
                    done
                fi
            done
        fi

        # LDPC
        if should_run_component_test_group "ldpc"; then
            bin_file=$CUPHY_EXAMPLES/error_correction/cuphy_ex_ldpc
            checkBinFile "$bin_file"
            if [ -f "${bin_file}" ]; then
                for tv in $(find $BASE_TV_PATH -name "*ldpc_BG1*.h5"); do
                    test_cmd=$(echo -n ${bin_file} -i ${tv} -n 10 -p 8 -f)
                    runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
                done

                for tv in $(find $BASE_TV_PATH -name "*ldpc_BG2*.h5"); do
                    test_cmd=$(echo -n ${bin_file} -i ${tv} -n 10 -p 8 -g 2)
                    runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
                done

                # When no TV is provided, input data is generated. Uses the LDPC encoder to do so.
                test_cmd=$(echo -n ${bin_file})
                runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
            fi
        fi

        # Polar
        if should_run_component_test_group "polar"; then
            bin_file=$CUPHY_EXAMPLES/polar_encoder/cuphy_ex_polar_encoder
            checkBinFile "$bin_file"
            if [ -f "${bin_file}" ]; then
                for tv in $(find $BASE_TV_PATH -name "*TV_polarEnc*.h5"); do
                    test_cmd=$(echo -n ${bin_file} -i ${tv})
                    runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
                done
            fi
        fi

        # Tests for all cuPHY make test cases that do not require a special directory.
        # These tests do not require TV input
        if should_run_component_test_group "gtests"; then
            COMPONENT_TESTS=("$BUILD_DIR/test/cfo_ta_est/test_cfo_ta_est"      #gtest for CFO and TA estimation
                "$BUILD_DIR/test/convert/test_convert"                         #gtest for tensor descriptor conversions
                "$BUILD_DIR/test/crc/crcTest"                                  #gtest for CRC
                "$BUILD_DIR/test/crc/prepareCrcBuffersTest"                    #gtest for prepare CRC buffers (downlink)
                "$BUILD_DIR/test/descrambling/testDescrambling"                #gtest for descrambling
                "$BUILD_DIR/test/dl_rate_matching/testDlRateMatching"          #gtest for rate matching
                "$BUILD_DIR/test/modulation_mapper/testModulationMapper"       #gtest for modulation mapper
                "$BUILD_DIR/test/error_correction/test_ldpc_internal_app_addr" #gtest for LDPC
                "$BUILD_DIR/test/error_correction/test_ldpc_internal_rc"       #gtest for LDPC
                "$BUILD_DIR/test/error_correction/test_ldpc_internal_loader"   #gtest for LDPC
                "$BUILD_DIR/test/soft_demapper/test_soft_demapper_internal"    #gtest for soft demapper
                "$BUILD_DIR/test/rng/test_rng"                                 #gtest for random number generation
                "$BUILD_DIR/test/fill/test_fill"                               #gtest for tensor fill operations
                "$BUILD_DIR/test/tile/test_tile"                               #gtest for tensor tile/repmat operations
                "$BUILD_DIR/test/elementwise/test_elementwise"                 #gtest for tensor elementwise operations
                "$BUILD_DIR/test/reduction/test_reduction"                     #gtest for tensor reduction operations
                "$BUILD_DIR/test/kernelDescr/testKernelDescr"                  #gtest for kernel descriptors
                "$BUILD_DIR/test/tensor_desc_tests/tensor_desc_tests"          #gtest for tensor descriptors
            )

            for ((i = 0; i < "${#COMPONENT_TESTS[@]}"; i++)); do
                bin_file="${COMPONENT_TESTS[$i]}"
                checkBinFile "$bin_file"
                if [ -f "${bin_file}" ]; then
                    checkwaitfile
                    echo "$bin_file"
                    $bin_file | tee $LOGFILE
                    checktest $LOGFILE
                fi
            done
        fi

        # Tests for the make test cases that require a special working directory
        if should_run_component_test_group "hdf5"; then
            HDF5_TESTS=("$BUILD_DIR/test/hdf5/test_hdf5"
                "$BUILD_DIR/test/hdf5/test_hdf5_host"
            )
            cwd=$(pwd)             # save previous working directory
            HDF5_DIR="./test/hdf5" # HDF5 tests should be run from this directory
            if [ -d ${HDF5_DIR} ]; then
                cd ${HDF5_DIR}
            fi
            for ((i = 0; i < "${#HDF5_TESTS[@]}"; i++)); do
                bin_file="${cwd}/${HDF5_TESTS[$i]}"
                checkBinFile "$bin_file"
                if [ -f "${bin_file}" ]; then
                    checkwaitfile
                    echo "$bin_file"
                    $bin_file | tee $LOGFILE
                    checktest $LOGFILE
                fi
            done
            cd ${cwd} # restore working directory
        fi
    fi
}

# Function to check if BFW TV has srsTv field and extract SRS TV number(s)
# Returns space-separated list of SRS TV numbers
function get_srs_tv_from_bfw {
    local bfw_tv=$1
    # Check if srsTv dataset exists
    local srs_tv_data=$(h5dump -d /srsTv "$bfw_tv" 2>/dev/null)
    if [ -z "$srs_tv_data" ]; then
        echo ""
        return
    fi

    # Extract all numbers from the DATA section, handling multiple formats:
    # Single value: (0,0): 21523
    # Multiple values: (0,0): 21643, (0,1): 21653, ...
    # or with DATA wrapper: DATA { (0,0): 21643, (0,1): 21653 }
    local srs_tv_nums=$(echo "$srs_tv_data" | grep -A 20 "DATA" | grep -oE '\([0-9,]+\): [0-9]+' | sed -E 's/.*: ([0-9]+)/\1/' | tr '\n' ' ')

    # Remove trailing space and return
    echo "${srs_tv_nums% }"
}

function runSRSBFW {
    bin_file=$CUPHY_EXAMPLES/srs_bfw/cuphy_ex_srs_bfw
    checkBinFile "$bin_file"
    if [ -f "${bin_file}" ]; then
        for bfw_tv in $(find $BASE_TV_PATH -name "TVnr_*_BFW_gNB_CUPHY_*.h5"); do

            srs_tv_nums=$(get_srs_tv_from_bfw $bfw_tv)
            if [ -n "$srs_tv_nums" ]; then
                echo "Processing BFW TV: $bfw_tv with SRS TV numbers: $srs_tv_nums"

                # Build array of SRS TV files
                srs_tv_files=()
                all_srs_found=true

                for srs_tv_num in $srs_tv_nums; do
                    # Try both naming patterns for SRS TV; 'head -n1' is added to handle multiple files scenario
                    if ([ $srs_tv_num -ge 8000 ] && [ $srs_tv_num -le 8999 ]) || [ $srs_tv_num -le 999 ]; then
                        srs_tv=$(find $BASE_TV_PATH -name "TVnr_$(printf '%04d' $srs_tv_num)_SRS_gNB_CUPHY_*.h5" | head -n1)
                    else
                        srs_tv=$(find $BASE_TV_PATH -name "TVnr_ULMIX_$(printf '%04d' $srs_tv_num)_SRS_gNB_CUPHY_*.h5" | head -n1)
                    fi

                    if [ -n "$srs_tv" ]; then
                        srs_tv_files+=("$srs_tv")
                        echo "  Found SRS TV: $srs_tv for TC number: $srs_tv_num"
                    else
                        echo "  Error: Could not find SRS TV file for TC number: $srs_tv_num"
                        all_srs_found=false
                        break
                    fi
                done

                if [ "$all_srs_found" = true ]; then
                    # Build test command with space-separated SRS files (more concise format)
                    test_cmd="${bin_file} --srs"
                    for srs_file in "${srs_tv_files[@]}"; do
                        test_cmd="${test_cmd} ${srs_file}"
                    done
                    test_cmd="${test_cmd} --bfw ${bfw_tv}"

                    echo "Running test command: $test_cmd"
                    runtest "$test_cmd" $RUN_COMPUTE_SANITIZER
                else
                    echo "Error: Could not find all required SRS TV files for BFW TV: $bfw_tv"
                    exit 1
                fi
            fi
        done
    fi
}

#=======================================================================================================================

if [ -z "$DEBUG" ]; then
    set -e # Exit immediately on non-zero return status.
fi

TESTDIR=$(dirname $(readlink -f $0))
CUPHY_ROOT=$(
    builtin cd $TESTDIR/..
    pwd
)
BUILD_DIR=$CUPHY_ROOT/../build/cuPHY

# Initialize default values
BASE_TV_PATH='./'
GPU_ID=0
PUSCH_CB_ERROR_CHECK=1
RUN_COMPONENT_TESTS=0
RUN_COMPUTE_SANITIZER=1
PYTHON3='python3'
UNIT_TEST_BLOCK_TO_RUN_FIRST=0
CUSTOM_BUILD_DIR=""
RUN_SRS_BFW_ONLY=0
ONLY_BLOCK=""
COMPONENT_TEST_FILTERS=()
VALID_COMPONENT_TEST_GROUPS=("pdsch" "ldpc" "polar" "gtests" "hdf5")

# Check for help flags first
for arg in "$@"; do
    if [[ "$arg" == "-h" ]] || [[ "$arg" == "--help" ]] || [[ "$arg" == *"-?" ]]; then
        usage
    fi
done

# Parse arguments - support both positional and optional flags
args=("$@")
positional_args=()
i=0

while [ $i -lt ${#args[@]} ]; do
    case "${args[i]}" in
    -t | --tv-path)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            BASE_TV_PATH="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    -g | --gpu)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            GPU_ID="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    -c | --cb-error-check)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            PUSCH_CB_ERROR_CHECK="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    -m | --component-tests)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            RUN_COMPONENT_TESTS="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    -s | --sanitizer)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            RUN_COMPUTE_SANITIZER="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    -p | --python)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            PYTHON3="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    -f | --first-block)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            UNIT_TEST_BLOCK_TO_RUN_FIRST="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    -b | --build-dir)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            CUSTOM_BUILD_DIR="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    -o | --only-block)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            ONLY_BLOCK="${args[$((i + 1))]}"
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    --component-filter)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
            IFS=',' read -r -a component_test_values <<<"${args[$((i + 1))]}"
            for component_test in "${component_test_values[@]}"; do
                normalized_component_test=$(echo "$component_test" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
                if [ -n "$normalized_component_test" ]; then
                    add_component_test_filter "$normalized_component_test"
                fi
            done
            i=$((i + 1))
        else
            echo "Error: ${args[i]} option requires an argument"
            usage 1
        fi
        ;;
    --srs_bfw)
        RUN_SRS_BFW_ONLY=1
        ;;
    -*)
        echo "Error: Unknown option ${args[i]}"
        usage 1
        ;;
    *)
        positional_args+=("${args[i]}")
        ;;
    esac
    i=$((i + 1))
done

# Handle positional arguments for backward compatibility
if [ ${#positional_args[@]} -gt 0 ]; then
    BASE_TV_PATH=${positional_args[0]:-$BASE_TV_PATH}
fi
if [ ${#positional_args[@]} -gt 1 ]; then
    GPU_ID=${positional_args[1]:-$GPU_ID}
fi
if [ ${#positional_args[@]} -gt 2 ]; then
    PUSCH_CB_ERROR_CHECK=${positional_args[2]:-$PUSCH_CB_ERROR_CHECK}
fi
if [ ${#positional_args[@]} -gt 3 ]; then
    RUN_COMPONENT_TESTS=${positional_args[3]:-$RUN_COMPONENT_TESTS}
fi
if [ ${#positional_args[@]} -gt 4 ]; then
    RUN_COMPUTE_SANITIZER=${positional_args[4]:-$RUN_COMPUTE_SANITIZER}
fi
if [ ${#positional_args[@]} -gt 5 ]; then
    PYTHON3=${positional_args[5]:-$PYTHON3}
fi
if [ ${#positional_args[@]} -gt 6 ]; then
    UNIT_TEST_BLOCK_TO_RUN_FIRST=${positional_args[6]:-$UNIT_TEST_BLOCK_TO_RUN_FIRST}
fi

# Validate argument count for positional usage
if [ ${#positional_args[@]} -gt 7 ]; then
    echo "Error: Too many positional arguments"
    usage 1
fi

# Set BUILD_DIR to custom path if provided
if [ -n "$CUSTOM_BUILD_DIR" ]; then
    BUILD_DIR="$CUSTOM_BUILD_DIR"
fi
CUPHY_EXAMPLES=$BUILD_DIR/examples
echo "CUPHY_EXAMPLES=${CUPHY_EXAMPLES}"

# hard checks to fail fast on a bad build path
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "ERROR: BUILD_DIR '$BUILD_DIR' does not exist."
    echo "Tip: pass a correct path with:  -b /absolute/path/to/build/cuPHY"
    exit 2
fi

if [[ ! -d "$CUPHY_EXAMPLES" ]]; then
    echo "ERROR: Examples directory '$CUPHY_EXAMPLES' does not exist."
    echo "Tip: is the build complete? (make sure examples were built)"
    exit 2
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "PUSCH_CB_ERROR_CHECK=${PUSCH_CB_ERROR_CHECK}"

echo "RUN_COMPONENT_TESTS=${RUN_COMPONENT_TESTS}"

for selected_component_group in "${COMPONENT_TEST_FILTERS[@]}"; do
    component_group_is_valid=0
    for valid_component_group in "${VALID_COMPONENT_TEST_GROUPS[@]}"; do
        if [ "$selected_component_group" = "$valid_component_group" ]; then
            component_group_is_valid=1
            break
        fi
    done
    if [ $component_group_is_valid -eq 0 ]; then
        echo "invalid value for --component-filter: ${selected_component_group}"
        echo "valid values: ${VALID_COMPONENT_TEST_GROUPS[*]}"
        usage 1
    fi
done

if [ ${#COMPONENT_TEST_FILTERS[@]} -gt 0 ]; then
    echo "COMPONENT_TEST_FILTERS=${COMPONENT_TEST_FILTERS[*]}"
    if [ "$RUN_COMPONENT_TESTS" -ne 1 ]; then
        echo "Enabling component tests because --component-filter was specified"
        RUN_COMPONENT_TESTS=1
    fi
fi

sanitizer_selected=''
sanitizer_idx=0
sanitizer_arg=$RUN_COMPUTE_SANITIZER
if [ $sanitizer_arg -gt 15 ]; then
    usage 1
fi
while [ $sanitizer_arg -ne 0 ]; do
    sanitizer_idx=$((sanitizer_idx + 1))
    if (($sanitizer_arg & 1)); then
        sanitizer_selected=$(echo -n $sanitizer_selected ${sanitizer_tool_options[$sanitizer_idx]})
    fi
    sanitizer_arg=$(($sanitizer_arg >> 1))
done
echo "RUN_COMPUTE_SANITIZER=${RUN_COMPUTE_SANITIZER}, i.e., tools: $sanitizer_selected"

# Python3 with mypy, interrogate, pytest installed required
$PYTHON3 --version

# Validate UNIT_TEST_BLOCK_TO_RUN_FIRST
if [ $UNIT_TEST_BLOCK_TO_RUN_FIRST -lt 0 ] || [ $UNIT_TEST_BLOCK_TO_RUN_FIRST -ge ${#unit_test_blocks[@]} ]; then
    echo "invalid value for UNIT_TEST_BLOCK_TO_RUN_FIRST. Must be between 0 and $((${#unit_test_blocks[@]} - 1))"
    usage 1
fi

# Validate ONLY_BLOCK if specified
if [ -n "$ONLY_BLOCK" ]; then
    if [ $ONLY_BLOCK -lt 0 ] || [ $ONLY_BLOCK -ge ${#unit_test_blocks[@]} ]; then
        echo "invalid value for ONLY_BLOCK. Must be between 0 and $((${#unit_test_blocks[@]} - 1))"
        usage 1
    fi
fi

# If --srs_bfw flag is set, only run SRSBFW tests
if [ $RUN_SRS_BFW_ONLY -eq 1 ]; then
    runSRSBFW
    if [ $? -ne 0 ]; then
        exit 1 # Exit with an error code if runSRSBFW fails
    fi
    exit 0 # Exit with success if runSRSBFW succeeds
fi

# Manipulate unit_test_blocks array based on options
if [ -n "$ONLY_BLOCK" ]; then
    # Run only the specified block
    selected_block="${unit_test_blocks[$ONLY_BLOCK]}"
    unit_test_blocks=("$selected_block")
    echo "Running only test block $ONLY_BLOCK: $selected_block"
elif [ $UNIT_TEST_BLOCK_TO_RUN_FIRST -ne 0 ]; then
    # Reorder array to run specified block first
    first_block="${unit_test_blocks[$UNIT_TEST_BLOCK_TO_RUN_FIRST]}"
    new_blocks=("$first_block")

    # Add remaining blocks (excluding the first block)
    for i in "${!unit_test_blocks[@]}"; do
        if [ $i -ne $UNIT_TEST_BLOCK_TO_RUN_FIRST ]; then
            new_blocks+=("${unit_test_blocks[$i]}")
        fi
    done

    unit_test_blocks=("${new_blocks[@]}")
    echo "Running test block $UNIT_TEST_BLOCK_TO_RUN_FIRST first: $first_block"
fi

if [ ${#COMPONENT_TEST_FILTERS[@]} -gt 0 ]; then
    ensure_component_test_block_is_selected
    echo "Including runComponentTests because --component-filter was specified"
fi

# Run all test blocks in the (possibly modified) array
for test_block in "${unit_test_blocks[@]}"; do
    $test_block
done
