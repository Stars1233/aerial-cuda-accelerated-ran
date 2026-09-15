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

script_name=${0##*/}
usage() {
    echo "Usage: $script_name [options] [cell_num] [alloc_type]"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message and exit"
    echo "  -c, --cell_num        Number of cells (default: 8)"
    echo "  -a, --alloc_type      Allocation type (default: 1)"
    echo "  -r, --renew           Force regeneration even if test vectors already exist"
    echo "  -t, --task <hex>      cuMAC TV task bitmask (default: 0xFFFF, CUMAC_TASK env overrides default)"
    echo "                        bit 0~3: 4T4R, bit 4: PFM SORT, bit 5: UE GROUP"
    echo "  --gpu_share <n>       GPU-share mode: 0=disabled (default), 1=enabled, 2=enabled with cuBB dumped H5 input"
    echo "  --gdb_script <path>   Wrapper script (e.g. gdb_bt.sh) prepended to each TV-generation binary launch."
    echo "                        Pass an empty string to disable wrapping. Default: empty (no wrapping)."
    echo ""
    echo "Positional arguments (for backward compatibility):"
    echo "  cell_num         Number of cells"
    echo "  alloc_type       Allocation type"
    echo ""
    echo "Examples:"
    echo "  $script_name -c 8 -a 1"
    echo "  $script_name --cell_num=8 --alloc_type=1"
    echo "  $script_name 8 1                    # positional arguments"
    echo "  $script_name -c 4 -r                # force regeneration"
    echo "  $script_name -c 4 --task 0x20       # generate/check UE GROUP TVs only"
    echo "  $script_name -c 4 --gdb_script /path/to/gdb_bt.sh"
}

LOG_INFO() {
    LOG_TIME="$(date -u '+%T.%6N')"
    info="$1"
    echo "${LOG_TIME} ${info}"
}

LOG_CMD() {
    LOG_TIME="$(date -u '+%T.%6N')"
    cmd="$1"
    echo "${LOG_TIME} [${cmd}]"
    eval "$cmd"
}

is_screen_running() {
    title=$1
    check=$(screen -ls | grep "\.${title}")
    if [ "$check" != "" ]; then
        echo 1
    else
        echo 0
    fi
}

run_in_screen() {
    title=$1
    cmd=$2
    debug=$3
    if [ "$debug" != "" ]; then
        echo "screen[$title]: $cmd"
    fi
    screen -L -t $title -dmS $title bash -c "$cmd; echo exit_code=\$?";
}

wait_for_screen_exit() {
    title=$1
    timeout_sec=${2:-300}  # Default 5 minutes timeout

    LOG_INFO "Waiting for screen '$title' to exit (timeout: ${timeout_sec}s)..."

    let counter=0
    while [ $(is_screen_running $title) -eq 1 ]; do
        sleep 1
        let counter=counter+1

        if [ $counter -gt $timeout_sec ]; then
            LOG_INFO "Timeout waiting for screen '$title' to exit after ${timeout_sec} seconds"
            return 1
        fi

        if [ $((counter % 10)) -eq 0 ]; then
            LOG_INFO "Still waiting for screen '$title' to exit... (${counter}s elapsed)"
        fi
    done

    LOG_INFO "Screen '$title' has exited after ${counter} seconds"
    return 0
}

# Wait until ANY of the given screens exits, or the timeout elapses.
# Polls once per second; prints a progress line every 10s.
#
# Usage: wait_for_any_screen_exit "title1 title2 ..." [timeout_sec]
# Returns: 0 on first-exit detection (and logs which title exited first),
#          1 on timeout (no screen exited within the budget).
wait_for_any_screen_exit() {
    titles="$1"
    timeout_sec="${2:-300}"

    LOG_INFO "Waiting for any of [${titles}] to exit (timeout: ${timeout_sec}s)..."

    let counter=0
    exited=""
    while [ ${counter} -le ${timeout_sec} ]; do
        for title in ${titles}; do
            if [ $(is_screen_running "$title") -eq 0 ]; then
                exited="$title"
                break 2
            fi
        done
        sleep 1
        let counter=counter+1
        if [ $((counter % 10)) -eq 0 ]; then
            LOG_INFO "Still waiting for any of [${titles}] to exit... (${counter}s elapsed)"
        fi
    done

    if [ -n "${exited}" ]; then
        LOG_INFO "Screen '${exited}' exited first after ${counter}s"
        return 0
    fi
    LOG_INFO "Timeout (${timeout_sec}s) waiting for any of [${titles}] to exit"
    return 1
}

# Initialize default values
cell_num=8
alloc_type=1
force_renew=0
task=""
gpu_share=0   # 0=disabled  1=random SRS  2=CUBB H5 (path from config yaml)
# Default GDB_SCRIPT is empty (no wrapping). The caller (e.g.
# run_cumcp_sa.sh) is responsible for forwarding its own --gdb_script
# value when wrapping the TV-gen binaries is desired.
GDB_SCRIPT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--cell_num)
            if [[ -n "$2" && "$2" != -* ]]; then
                cell_num="$2"
                shift 2
            else
                echo "Error: -c/--cell_num requires a value"
                usage
                exit 1
            fi
            ;;
        --cell_num=*)
            cell_num="${1#*=}"
            shift
            ;;
        -a|--alloc_type)
            if [[ -n "$2" && "$2" != -* ]]; then
                alloc_type="$2"
                shift 2
            else
                echo "Error: -a/--alloc_type requires a value"
                usage
                exit 1
            fi
            ;;
        --alloc_type=*)
            alloc_type="${1#*=}"
            shift
            ;;
        -r|--renew)
            force_renew=1
            shift
            ;;
        --task=*)
            task="${1#*=}"
            shift
            ;;
        -t|--task)
            if [[ -n "$2" && "$2" != -* ]]; then
                task="$2"
                shift 2
            else
                echo "Error: -t/--task requires a value"
                usage
                exit 1
            fi
            ;;
        --gpu_share)
            if [[ -n "$2" && "$2" != -* ]]; then
                gpu_share="$2"
                shift 2
            else
                echo "Error: --gpu_share requires a value (0, 1, or 2)"
                usage
                exit 1
            fi
            ;;
        --gpu_share=*)
            gpu_share="${1#*=}"
            shift
            ;;
        --gdb_script=*)
            GDB_SCRIPT="${1#*=}"
            shift
            ;;
        --gdb_script)
            # Allow explicit empty string ("") to disable wrapping, so
            # accept any next token (including one that looks like a
            # flag) instead of the usual "$2" != -* guard.
            GDB_SCRIPT="$2"
            shift 2
            ;;
        -*)
            echo "Error: Unknown option $1"
            usage
            exit 1
            ;;
        *)
            # Handle positional arguments for backward compatibility
            if [[ -z "$positional_cell_num" ]]; then
                positional_cell_num="$1"
                cell_num="$1"
            elif [[ -z "$positional_alloc_type" ]]; then
                positional_alloc_type="$1"
                alloc_type="$1"
            else
                echo "Error: Too many positional arguments"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate parameters
if ! [[ "${cell_num}" =~ ^[0-9]+$ ]]; then
    echo "Error: cell_num must be a positive integer, got: $cell_num"
    exit 1
fi

if [ "${alloc_type}" != "0" ] && [ "${alloc_type}" != "1" ]; then
    echo "Error: alloc_type must be 0 or 1, got: $alloc_type"
    exit 1
fi

if [ "${gpu_share}" != "0" ] && [ "${gpu_share}" != "1" ] && [ "${gpu_share}" != "2" ]; then
    echo "Error: --gpu_share must be 0, 1, or 2, got: ${gpu_share}"
    exit 1
fi

task_bitmask="0xFFFF"
if [ -n "${CUMAC_TASK}" ]; then
    task_bitmask="${CUMAC_TASK}"
fi
if [ -n "${task}" ]; then
    task_bitmask="${task}"
fi
if ! [[ "${task_bitmask}" =~ ^(0[xX][0-9a-fA-F]+|[0-9]+)$ ]]; then
    echo "Error: task_bitmask must be decimal or hex, got: ${task_bitmask}"
    exit 1
fi
task_bitmask_value=$((task_bitmask))

if [ "$cuBB_SDK" = "" ]; then
    echo "Please set cuBB_SDK first"
    exit 1
fi

# Verify GDB_SCRIPT path (empty string is fine and means "no wrapping").
if [ -n "${GDB_SCRIPT}" ] && [ ! -f "${GDB_SCRIPT}" ]; then
    LOG_INFO "GDB_SCRIPT '${GDB_SCRIPT}' not found, disabling gdb wrapper"
    GDB_SCRIPT=""
fi

LOG_INFO "[${script_name}] Parameters: cell_num=${cell_num} alloc_type=${alloc_type} force_renew=${force_renew} task_bitmask=${task_bitmask} gpu_share=${gpu_share} gdb_script='${GDB_SCRIPT}'"

if [ `whoami` = "root" ];then
    USE_SUDO=""
else
    USE_SUDO="sudo -E"
fi

# TV generation output directory
OUTPUT_DIR=cumac.${cell_num}c.type${alloc_type}.gpu_share${gpu_share}

if [ "${LOG_PATH}" = "" ]; then
    LOG_PATH=${cuBB_SDK}/logs/latest
fi

# Clean old logs if exist. Run "rm -rf" only for known LOG_PATH.
if [ "${LOG_PATH}" = "${cuBB_SDK}/logs/latest" ]; then
    LOG_CMD "${USE_SUDO} rm -rf ${LOG_PATH}/*"
fi

is_4t4r_enabled() {
    (( (task_bitmask_value & 0xF) != 0 ))
}

is_pfm_sort_enabled() {
    (( (task_bitmask_value & 0x10) != 0 ))
}

is_ue_grp_enabled() {
    (( (task_bitmask_value & 0x20) != 0 ))
}

check_4t4r_tv_validity() {
    folder_name=$1
    file_path=${cuBB_SDK}/testVectors/${folder_name}/TV_cumac_F08-MC-CC-${cell_num}PC_DL.h5
    if [ -f "${file_path}" ]; then
        alloc_type_in_tv=$(h5ls -ld ${file_path}/cumacSchedulerParam | grep -o 'allocType=[0-9.]*' | cut -d '=' -f 2)
        cell_num_in_tv=$(h5ls -ld ${file_path}/cumacSchedulerParam | grep -o 'nCell=[0-9.]*' | cut -d '=' -f 2)
        if [ "${cell_num_in_tv}" != "${cell_num}" ] || [ "${alloc_type_in_tv}" != "${alloc_type}" ]; then
            LOG_INFO "4T4R TV at ${cuBB_SDK}/testVectors/${folder_name}/ is invalid: cell_num=${cell_num_in_tv} alloc_type=${alloc_type_in_tv}"
            return 1
        fi
        LOG_INFO "4T4R TV at ${cuBB_SDK}/testVectors/${folder_name}/ is valid"
        return 0
    else
        LOG_INFO "4T4R TV is not found at ${cuBB_SDK}/testVectors/${folder_name}"
        return 1
    fi
}

check_pfm_sort_tv_validity() {
    folder_name=$1
    if compgen -G "${cuBB_SDK}/testVectors/${folder_name}/PFM_SORT*.h5" > /dev/null; then
        LOG_INFO "PFM SORT TV exists at ${cuBB_SDK}/testVectors/${folder_name}"
        return 0
    fi
    LOG_INFO "PFM SORT TV is not found at ${cuBB_SDK}/testVectors/${folder_name}"
    return 1
}

check_ue_grp_tv_validity() {
    folder_name=$1
    if compgen -G "${cuBB_SDK}/testVectors/${folder_name}/muUePairTV_*.h5" > /dev/null; then
        LOG_INFO "UE GROUP TV exists at ${cuBB_SDK}/testVectors/${folder_name}"
        return 0
    fi
    LOG_INFO "UE GROUP TV is not found at ${cuBB_SDK}/testVectors/${folder_name}"
    return 1
}

# Generate TV
LOG_INFO "======================================"
LOG_INFO "Generate TV for cell_num=${cell_num} alloc_type=${alloc_type} ..."

if [ "${BUILD}" = "" ]; then
    export BUILD=build.$(arch)
fi

# Set BUILD_DIR to same with BUILD for build_aerial_sdk.sh
export BUILD_DIR=${BUILD}

sed_set_value() {
    name="$1"
    value="$2"
    file="$3"
    # Replace only the value token, preserving any trailing inline comment (# ...).
    sed_cmd="sed -i 's/\(${name}[ ]*\)[^ #]*/\1${value}/' ${file}"
    LOG_INFO "${sed_cmd}"
    eval "${sed_cmd}"
}

if [ "${force_renew}" = "1" ]; then
    LOG_CMD "rm -rf ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
fi

# Remove old TV link or folder if exist
LOG_CMD "rm -rf ${cuBB_SDK}/testVectors/cumac"
LOG_CMD "mkdir -p ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
LOG_CMD "ls -ld ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"

remove_4t4r_tv() {
    LOG_INFO "Remove old 4T4R TVs from ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
    LOG_CMD "${USE_SUDO} rm -f ${cuBB_SDK}/testVectors/${OUTPUT_DIR}/TV_cumac_F08-MC-CC-${cell_num}PC_DL.h5"
    LOG_CMD "${USE_SUDO} rm -f ${cuBB_SDK}/testVectors/${OUTPUT_DIR}/TV_F08_MAC_SCH_CONFIG_REQUEST_cell_*.h5"
    LOG_CMD "${USE_SUDO} rm -f ${cuBB_SDK}/testVectors/${OUTPUT_DIR}/TV_F08_MAC_SCH_TTI_REQUEST_cell_*.h5"
}

remove_pfm_sort_tv() {
    LOG_INFO "Remove old PFM SORT TVs from ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
    LOG_CMD "${USE_SUDO} rm -f ${cuBB_SDK}/testVectors/${OUTPUT_DIR}/PFM_SORT*.h5"
}

remove_ue_grp_tv() {
    LOG_INFO "Remove old UE GROUP TVs from ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
    LOG_CMD "${USE_SUDO} rm -f ${cuBB_SDK}/testVectors/${OUTPUT_DIR}/muUePairTV_*.h5"
}

########################################################
# Generate 4T4R TV
########################################################
gen_4t4r_tv() {
    # Set parameters. Values are loaded at runtime from parameters.yaml
    # (see cuMAC/examples/parameters.cpp); editing the YAML needs no rebuild.
    PARAMS_YAML=${cuBB_SDK}/cuMAC/examples/parameters.yaml
    sed_set_value "numCellConst:" "${cell_num}" ${PARAMS_YAML}
    sed_set_value "gpuDeviceIdx:" "0" ${PARAMS_YAML}
    # sed_set_value "cpuGpuPerfGapSumRConst:" 0.03 ${PARAMS_YAML}
    # sed_set_value "cpuGpuPerfGapPerUeConst:" 0.01 ${PARAMS_YAML}

    sed_set_value "gpuAllocTypeConst:" "${alloc_type}" ${PARAMS_YAML}
    sed_set_value "cpuAllocTypeConst:" "${alloc_type}" ${PARAMS_YAML}

    # Parameter values are applied at runtime, so no rebuild is needed after
    # changing them. Only build multiCellSchedulerUeSelection if it is missing.
    TV_BINARY="${cuBB_SDK}/${BUILD}/cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection"
    if [ ! -x "${TV_BINARY}" ]; then
        LOCAL_CMD="cd ${cuBB_SDK} && testBenches/phase4_test_scripts/build_aerial_sdk.sh --targets multiCellSchedulerUeSelection > ${LOG_PATH}/build_targets.log 2>&1"
        LOG_INFO "${LOCAL_CMD}"
        build_start=$(date +%s)
        eval "${LOCAL_CMD}"
        return_code=$?
        build_end=$(date +%s)
        LOG_INFO "[${LOCAL_CMD}] time cost: $((build_end - build_start))s"
        if [ ${return_code} -ne 0 ]; then
            LOG_INFO "[${LOCAL_CMD}] failed"
            exit 1
        fi
    fi
    LOG_INFO "======================================"

    # Run test vector generation. Point the binary at the edited parameters.yaml
    # via CUMAC_PARAMS_YAML (it runs from the TV output dir, not examples/);
    # `env` keeps the variable across sudo's environment filtering.
    LOCAL_CMD="cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -t 3 > ${LOG_PATH}/tv_gen_4t4r.log 2>&1"
    LOCAL_CMD="${GDB_SCRIPT} ${cuBB_SDK}/${BUILD}/${LOCAL_CMD}"
    LOCAL_CMD="env CUMAC_PARAMS_YAML=${PARAMS_YAML} timeout -s 9 600 ${LOCAL_CMD}" # 10 minutes timeout
    LOCAL_CMD="cd ${cuBB_SDK}/testVectors/${OUTPUT_DIR} && ${USE_SUDO} ${LOCAL_CMD}"
    LOG_INFO "${LOCAL_CMD}"
    tv_start=$(date +%s)
    eval "${LOCAL_CMD}"
    return_code=$?
    tv_end=$(date +%s)
    LOG_INFO "TV generation [multiCellSchedulerUeSelection -t 3] time cost: $((tv_end - tv_start))s"
    if [ ${return_code} -ne 0 ]; then
        LOG_INFO "4T4R TV generation failed"
    fi
    LOG_INFO "======================================"
}

########################################################
# Generate PFM sorting TV
########################################################
gen_pfm_sort_tv() {
    # Set cell_num for PFM sorting TV generation
    sed_set_value "NUM_CELL:" "${cell_num}" "${cuBB_SDK}/cuMAC/examples/pfmSort/config.yaml"
    LOCAL_CMD="cuMAC/examples/pfmSort/pfmSortTest -t 2 > ${LOG_PATH}/tv_gen_pfm_sort.log 2>&1"
    LOCAL_CMD="${GDB_SCRIPT} ${cuBB_SDK}/${BUILD}/${LOCAL_CMD}"
    LOCAL_CMD="timeout -s 9 600 ${LOCAL_CMD}" # 10 minutes timeout
    LOCAL_CMD="cd ${cuBB_SDK}/testVectors/${OUTPUT_DIR} && ${USE_SUDO} ${LOCAL_CMD}"
    LOG_INFO "${LOCAL_CMD}"
    tv_start=$(date +%s)
    eval "${LOCAL_CMD}"
    return_code=$?
    tv_end=$(date +%s)
    LOG_INFO "TV generation [pfmSortTest -t 2] time cost: $((tv_end - tv_start))s"
    if [ ${return_code} -ne 0 ]; then
        LOG_INFO "PFM sorting TV generation failed"
    fi
    LOG_INFO "======================================"
}
########################################################
# Generate MU-MIMO UE Grouping (UE Pairing) TV
########################################################
gen_ue_grp_tv() {
    LOG_INFO "======================================"
    LOG_INFO "Generating MU-MIMO UE Grouping TV..."

    # Set config for TV generation
    MU_UE_GRP_CONFIG="${cuBB_SDK}/cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/config.yaml"

    # Update config to save all slots and set cell number
    sed_set_value "TV_SAVE_ALL_SLOTS:" "true" "${MU_UE_GRP_CONFIG}"
    sed_set_value "NUM_CELL:" "${cell_num}" "${MU_UE_GRP_CONFIG}"
    # sed_set_value "NUM_TIME_SLOTS:" "20" "${MU_UE_GRP_CONFIG}"
    sed_set_value "ENABLE_TV_TEST_MODE:" "true" "${MU_UE_GRP_CONFIG}"

    if [[ "${gpu_share}" = "0" || "${gpu_share}" = "" ]]; then
        sed_set_value "ENABLE_L1_L2_MEM_SHARING:" "false" "${MU_UE_GRP_CONFIG}"
        sed_set_value "ENABLE_CUBB_TV_INPUT:" "false" "${MU_UE_GRP_CONFIG}"
    elif [ "${gpu_share}" = "1" ]; then
        sed_set_value "ENABLE_L1_L2_MEM_SHARING:" "true" "${MU_UE_GRP_CONFIG}"
        sed_set_value "ENABLE_CUBB_TV_INPUT:" "false" "${MU_UE_GRP_CONFIG}"
    elif [ "${gpu_share}" = "2" ]; then
        sed_set_value "ENABLE_L1_L2_MEM_SHARING:" "true" "${MU_UE_GRP_CONFIG}"
        sed_set_value "ENABLE_CUBB_TV_INPUT:" "true" "${MU_UE_GRP_CONFIG}"
    else
        LOG_INFO "Unsupported gpu_share: ${gpu_share}"
    fi

    TMP_TV_DIR="${cuBB_SDK}/tv"
    LOG_CMD "${USE_SUDO} rm -rf ${TMP_TV_DIR}/*"

    # Kill any existing test processes
    LOG_CMD "${USE_SUDO} killall -9 -q l1_main cumac_main l2_main 2>/dev/null"
    sleep 3 # Wait for processes to be killed

    # Run the TV generation with modified config
    MU_UE_GRP_DIR="${cuBB_SDK}/${BUILD}/cuMAC/examples/muMimoUeGrpL2Integration"
    LOG_INFO "Running MU-MIMO UE Grouping TV generation..."
    tv_start=$(date +%s)

    # Per-cell TV generation cost scales roughly linearly with cell_num,
    # so size the timeout proportionally (50s per cell) instead of a
    # one-size-fits-all literal. Used both as the `timeout` SIGKILL
    # window and as the wait_for_screen_exit poll budget so the latter
    # cannot prematurely return success/timeout while the binary is
    # still running.
    ue_grp_timeout_sec=$((cell_num * 50))
    LOG_INFO "MU-MIMO UE Grouping per-process timeout: ${ue_grp_timeout_sec}s (cell_num=${cell_num} * 50)"

    # Run from cuBB_SDK directory so config files can be found
    # Output will be saved in testVectors/${OUTPUT_DIR}
    cd "${LOG_PATH}" || { LOG_INFO "Error: cannot cd to LOG_PATH: ${LOG_PATH}"; return 1; }

    # Start l1_muUeGrp_test (PRIMARY - must start first to create shared resources).
    # ${GDB_SCRIPT} (empty by default) prepends gdb_bt.sh so we capture a
    # backtrace on crash; the binary takes no positional args, so the
    # `--args $*` forwarding in gdb_bt.sh is a no-op for these launches.
    run_in_screen "l1_main" "cd ${cuBB_SDK} && ${USE_SUDO} timeout -s 9 ${ue_grp_timeout_sec} ${GDB_SCRIPT} ${MU_UE_GRP_DIR}/l1_muUeGrp_test" 1
    sleep 2

    # Start cumac_muUeGrp_test (SECONDARY - attaches to L1's resources)
    run_in_screen "cumac_main" "cd ${cuBB_SDK} && ${USE_SUDO} timeout -s 9 ${ue_grp_timeout_sec} ${GDB_SCRIPT} ${MU_UE_GRP_DIR}/cumac_muUeGrp_test" 1
    sleep 2

    # Start l2_muUeGrp_test
    run_in_screen "l2_main" "cd ${cuBB_SDK} && ${USE_SUDO} timeout -s 9 ${ue_grp_timeout_sec} ${GDB_SCRIPT} ${MU_UE_GRP_DIR}/l2_muUeGrp_test" 1

    # Wait until ANY one of the three cooperating screens exits. The
    # three binaries are coupled through shared-memory IPC, so once one
    # finishes (success path) or crashes, the other two are typically
    # blocked on a dead peer and would otherwise sit until the
    # per-process `timeout -s 9 ${ue_grp_timeout_sec}` SIGKILL fires --
    # wasting up to ue_grp_timeout_sec of wall-clock per remaining
    # screen. Match the wait budget to the per-process timeout so the
    # helper only gives up if every screen is somehow still alive past
    # that deadline.
    if wait_for_any_screen_exit "l1_main cumac_main l2_main" "${ue_grp_timeout_sec}"; then
        LOG_INFO "Sleeping 1s before force-killing the remaining screens"
        sleep 1
        screen -ls
    else
        LOG_INFO "Force-killing all three screens after wait timeout"
    fi

    # Force-kill the underlying binaries. killall is idempotent with -q
    # so the process that already exited normally is a silent no-op.
    LOG_CMD "${USE_SUDO} killall -9 -q l1_muUeGrp_test cumac_muUeGrp_test l2_muUeGrp_test 2>/dev/null || true"

    # Quit any still-attached screen sessions so the TV-count step below
    # sees a clean slate even when `screen -L` is slow to flush.
    for title in l1_main cumac_main l2_main; do
        if [ $(is_screen_running $title) -eq 1 ]; then
            screen -S "$title" -X quit 2>/dev/null || true
        fi
    done

    tv_end=$(date +%s)
    LOG_INFO "MU-MIMO UE Grouping TV generation time cost: $((tv_end - tv_start))s"

    TV_COUNT=$(ls -1 ${TMP_TV_DIR}/muUePairTV_*.h5 2>/dev/null | wc -l)
    if [ ${TV_COUNT} -eq 0 ]; then
        LOG_INFO "No MU-MIMO UE Grouping TV files found after generation"
    else
        LOG_CMD "${USE_SUDO} mv ${TMP_TV_DIR}/muUePairTV_*.h5 ${cuBB_SDK}/testVectors/${OUTPUT_DIR}/ 2>/dev/null || true"
    fi

    LOG_INFO "Generated ${TV_COUNT} MU-MIMO UE Grouping TV files"
    LOG_INFO "======================================"
}

########################################################
# Main function
########################################################
enabled_task_count=0

if is_4t4r_enabled; then
    enabled_task_count=$((enabled_task_count + 1))
    if [ "${force_renew}" = "1" ]; then
        remove_4t4r_tv
        gen_4t4r_tv
    elif check_4t4r_tv_validity "${OUTPUT_DIR}"; then
        LOG_INFO "Skip 4T4R TV generation: existing TV is valid"
    else
        gen_4t4r_tv
    fi
else
    LOG_INFO "Skip 4T4R TV generation: task_bitmask=${task_bitmask}"
fi

if is_pfm_sort_enabled; then
    enabled_task_count=$((enabled_task_count + 1))
    if [ "${force_renew}" = "1" ]; then
        remove_pfm_sort_tv
        gen_pfm_sort_tv
    elif check_pfm_sort_tv_validity "${OUTPUT_DIR}"; then
        LOG_INFO "Skip PFM SORT TV generation: existing TV found"
    else
        gen_pfm_sort_tv
    fi
else
    LOG_INFO "Skip PFM SORT TV generation: task_bitmask=${task_bitmask}"
fi

if is_ue_grp_enabled; then
    enabled_task_count=$((enabled_task_count + 1))
    if [ "${force_renew}" = "1" ]; then
        remove_ue_grp_tv
        gen_ue_grp_tv
    elif check_ue_grp_tv_validity "${OUTPUT_DIR}"; then
        LOG_INFO "Skip UE GROUP TV generation: existing TV found"
    else
        gen_ue_grp_tv
    fi
else
    LOG_INFO "Skip UE GROUP TV generation: task_bitmask=${task_bitmask}"
fi

if [ ${enabled_task_count} -eq 0 ]; then
    LOG_INFO "No TV generation task is enabled by task_bitmask=${task_bitmask}"
fi

# Create link to target TV folder
cd ${cuBB_SDK}/testVectors
rm -rf cumac
ln -s ${OUTPUT_DIR} cumac

LOG_INFO "TV generation succeeded: ${cuBB_SDK}/testVectors/cumac -> ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
LOG_INFO "======================================"

exit 0
