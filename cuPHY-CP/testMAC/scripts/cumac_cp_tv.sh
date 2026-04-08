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
function usage {
    echo "Usage: $script_name [options] [cell_num] [alloc_type]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message and exit"
    echo "  -c, --cell_num   Number of cells (default: 8)"
    echo "  -a, --alloc_type Allocation type (default: 1)"
    echo "  -r, --renew      Force regeneration even if test vectors already exist"
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
}

function LOG_INFO {
    LOG_TIME="$(date -u '+%T.%6N')"
    info="$1"
    echo "${LOG_TIME} ${info}"
}

# Initialize default values
cell_num=8
alloc_type=1
force_renew=0

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

if [ "$cuBB_SDK" = "" ]; then
    echo "Please set cuBB_SDK first"
    exit 1
fi

LOG_INFO "Parameters: cell_num=${cell_num} alloc_type=${alloc_type} force_renew=${force_renew}"

if [ `whoami` = "root" ];then
    USE_SUDO=""
else
    USE_SUDO="sudo -E"
fi

# TV generation output directory
OUTPUT_DIR=cumac.type${alloc_type}.${cell_num}c

function check_tv_validity {
    folder_name=$1
    file_path=${cuBB_SDK}/testVectors/${folder_name}/TV_cumac_F08-MC-CC-${cell_num}PC_DL.h5
    if [ -f ${file_path} ]; then
        alloc_type_in_tv=$(h5ls -ld ${file_path}/cumacSchedulerParam | grep -o 'allocType=[0-9.]*' | cut -d '=' -f 2)
        cell_num_in_tv=$(h5ls -ld ${file_path}/cumacSchedulerParam | grep -o 'nCell=[0-9.]*' | cut -d '=' -f 2)
        if [ "${cell_num_in_tv}" != "${cell_num}" ] || [ "${alloc_type_in_tv}" != "${alloc_type}" ]; then
            LOG_INFO "TV at ${cuBB_SDK}/testVectors/${folder_name}/ is invalid: cell_num=${cell_num_in_tv} alloc_type=${alloc_type_in_tv}"
            return 1
        fi
        LOG_INFO "TV at ${cuBB_SDK}/testVectors/${folder_name}/ is valid"
        return 0
    else
        LOG_INFO "TV is not found at ${cuBB_SDK}/testVectors/${folder_name}"
        return 1
    fi
}

if [ "${force_renew}" = "0" ]; then
    # Check if TV is valid at $cuBB_SDK/testVectors/cumac
    check_tv_validity "cumac"
    return_code=$?
    if [ ${return_code} -eq 0 ]; then
        LOG_INFO "Found TV for cell_num=${cell_num} alloc_type=${alloc_type} at ${cuBB_SDK}/testVectors/cumac"
        exit 0
    fi

    # Check if TV is valid at $cuBB_SDK/testVectors/${OUTPUT_DIR}
    check_tv_validity "${OUTPUT_DIR}"
    return_code=$?
    if [ ${return_code} -eq 0 ]; then
        LOG_INFO "Found TV for cell_num=${cell_num} alloc_type=${alloc_type} at ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
        LOG_INFO "Update link to ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
        cd ${cuBB_SDK}/testVectors
        rm -rf cumac
        ln -s ${OUTPUT_DIR} cumac
        exit 0
    fi
fi

# Generate TV
LOG_INFO "======================================"
LOG_INFO "Generate TV for cell_num=${cell_num} alloc_type=${alloc_type} ..."

if [ "${BUILD}" = "" ]; then
    export BUILD=build.$(arch)
fi

# Set BUILD_DIR to same with BUILD for build_aerial_sdk.sh
export BUILD_DIR=${BUILD}

function sed_set_value {
    name="$1"
    value="$2"
    file="$3"
    # sed_cmd="sed -i 's/${name}[ ]*:.*/${name}: ${value}/g' ${file}"
    sed_cmd="sed -i 's/${name}[ ]*.*/${name} ${value}/g' ${file}"
    LOG_INFO "${sed_cmd}"
    eval "${sed_cmd}"
}

# Set parameters
sed_set_value "#define numCellConst" "${cell_num}" ${cuBB_SDK}/cuMAC/examples/parameters.h
sed_set_value "#define gpuDeviceIdx" "0" ${cuBB_SDK}/cuMAC/examples/parameters.h
# sed_set_value "#define cpuGpuPerfGapSumRConst" 0.03 ${cuBB_SDK}/cuMAC/examples/parameters.h
# sed_set_value "#define cpuGpuPerfGapPerUeConst" 0.01 ${cuBB_SDK}/cuMAC/examples/parameters.h

sed_set_value "#define gpuAllocTypeConst" "${alloc_type}" ${cuBB_SDK}/cuMAC/examples/parameters.h
sed_set_value "#define cpuAllocTypeConst" "${alloc_type}" ${cuBB_SDK}/cuMAC/examples/parameters.h

# Build
LOG_INFO "cd ${cuBB_SDK} && testBenches/phase4_test_scripts/build_aerial_sdk.sh --targets multiCellSchedulerUeSelection pfmSortTest cumac_cp test_mac > build_aerial_sdk.log 2>&1"
build_start=$(date +%s)
cd ${cuBB_SDK} && testBenches/phase4_test_scripts/build_aerial_sdk.sh --targets multiCellSchedulerUeSelection pfmSortTest cumac_cp test_mac > build_aerial_sdk.log 2>&1
return_code=$?
build_end=$(date +%s)
LOG_INFO "build_aerial_sdk.sh time cost: $((build_end - build_start))s"
if [ ${return_code} -ne 0 ]; then
    LOG_INFO "$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh --targets multiCellSchedulerUeSelection pfmSortTest cumac_cp test_mac > build_aerial_sdk.log 2>&1 failed"
    exit 1
fi
LOG_INFO "======================================"

# Remove old TV link or folder if exist
rm -rf ${cuBB_SDK}/testVectors/cumac
rm -rf ${cuBB_SDK}/testVectors/${OUTPUT_DIR}
mkdir -p ${cuBB_SDK}/testVectors/${OUTPUT_DIR}
ls -ld ${cuBB_SDK}/testVectors/${OUTPUT_DIR}

# Run test vector generation
LOCAL_CMD="cuMAC/examples/multiCellSchedulerUeSelection/multiCellSchedulerUeSelection -t 3 > cumac_cp_tv.log 2>&1"
LOCAL_CMD="${cuBB_SDK}/${BUILD}/${LOCAL_CMD}"
LOCAL_CMD="timeout -s 9 600 ${LOCAL_CMD}" # 10 minutes timeout
LOCAL_CMD="cd ${cuBB_SDK}/testVectors/${OUTPUT_DIR} && ${USE_SUDO} ${LOCAL_CMD}"
LOG_INFO "${LOCAL_CMD}"
tv_start=$(date +%s)
eval "${LOCAL_CMD}"
return_code=$?
tv_end=$(date +%s)
LOG_INFO "TV generation [multiCellSchedulerUeSelection -t 3] time cost: $((tv_end - tv_start))s"
if [ ${return_code} -ne 0 ]; then
    LOG_INFO "4T4R TV generation failed"
    exit 1
fi
LOG_INFO "======================================"

########################################################
# Generate PFM sorting TV
########################################################
# Set cell_num for PFM sorting TV generation
sed_set_value "NUM_CELL:" "${cell_num}" "${cuBB_SDK}/cuMAC/examples/pfmSort/config.yaml"
LOCAL_CMD="cuMAC/examples/pfmSort/pfmSortTest -t 2 >> cumac_cp_tv.log 2>&1"
LOCAL_CMD="${cuBB_SDK}/${BUILD}/${LOCAL_CMD}"
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
    exit 1
fi

# Create link to target TV folder
cd ${cuBB_SDK}/testVectors
rm -rf cumac
ln -s ${OUTPUT_DIR} cumac

LOG_INFO "TV generation succeeded: ${cuBB_SDK}/testVectors/cumac -> ${cuBB_SDK}/testVectors/${OUTPUT_DIR}"
LOG_INFO "======================================"
exit 0