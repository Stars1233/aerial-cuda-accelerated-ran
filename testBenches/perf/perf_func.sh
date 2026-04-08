#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# usage: ./perf_func.sh <test_vectors_folder> <cuphy_build_folder> <gpu_ID=0> <run compute-sanitizer=0> <Python_version=python3>

set -e # Exit immediately on non-zero return status.

SCRIPT_PATH=`realpath $0`
SRC_TEST_FOLDER=`dirname $SCRIPT_PATH` # Used to determine location of perf_yaml directory and update_yaml.py
echo "SRC test/ folder is: $SRC_TEST_FOLDER"

VECTORS_FOLDER=$1
BUILD_FOLDER=$2

GPU_ID=${3:-0} #Default is 0
echo "GPU_ID=${GPU_ID}"

RUN_COMPUTE_SANITIZER=${4:-0} # flag to enable or disable compute-sanitizer. Disabled by default until we fix existing mem. leaks. FIXME
echo "RUN_COMPUTE_SANITIZER=${RUN_COMPUTE_SANITIZER}"

if [[ "$RUN_COMPUTE_SANITIZER" == 1 ]]; then
   COMPUTE_SANITIZER_EXEC="compute-sanitizer" #Modify this if you want to try a different compute-sanitizer version
   # Print version used
   echo -n "compute-sanitizer: "
   $COMPUTE_SANITIZER_EXEC -v  | grep "Version"
   compute_sanitizer_cmd_prefix="${COMPUTE_SANITIZER_EXEC} --error-exitcode 1 --tool memcheck --leak-check full"
else
   compute_sanitizer_cmd_prefix=""
fi

# Python3 with pyyaml installed required
PYTHON3=${5:-'python3'}
$PYTHON3 --version

YAML_FOLDER="$SRC_TEST_FOLDER/perf_yaml"
REF_CHECKS="-k --k -b --c PUSCH,PDSCH,PDCCH,PUCCH,SSB,DLBFW,ULBFW,CSIRS,PRACH,SRS" # -k enables reference checks for PDSCH, --k for PDCCH and -b prints BLER for PUSCH and all other channels with -c. Set to empty string (see below) to disable them
#REF_CHECKS=""

# update the directory that have TVs
# TODO: currenlty cuMAC TVs are in aerial_sdk/cuMAC/testVectors; will update once CICD can generate cuMAC TVs, see TODO in update_yaml.py
$PYTHON3 "$SRC_TEST_FOLDER/update_yaml.py" --yaml $YAML_FOLDER --folder $VECTORS_FOLDER

function closeMPS {
    echo quit | CUDA_VISIBLE_DEVICES={$GPU_ID} CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control
}

function checkmem {
    # Check exit code of process that was the input to tee
    pid_errval=${PIPESTATUS[0]}
    if [ $pid_errval -ne 0 ]; then
       echo "Process exited with non-zero exit code $pid_errval"
       closeMPS
       return 1
    fi

    # Check for memory errors reported by compute-sanitizer
    if [[ "$RUN_COMPUTE_SANITIZER" == 1 ]]; then
       errval=$(grep 'ERROR SUMMARY' $1 | sed -e 's/========= ERROR SUMMARY: \(.*\)error\(.*\)/\1/')
       if [ -z "$errval" ]; then
           echo "Exit on error found"
           closeMPS
           return 1
       fi
       if [ "$errval" -ne "0" ]; then
           echo "memcheck error count="$errval
           closeMPS
           return $errval
       else
           echo ""
       fi
    fi

    # Check for functional correctness for PUSCH, if ref. checks is enabled.
    # Currently PDSCH reference checks failure will cause cubb_gpu_test_bench to exit with a non zero code.
    # But for PUSCH the BLER values are printed, so they need to be post-processed
    if [[ "$REF_CHECKS" == *" -b"* ]]; then
        # Check for non-zero BLER for all TBs in all cells and for UCI too
        errCBs_string=$(grep -Po 'Error CBs\s*\K\d+' $1)
        #errCBs_string=$(grep -Po '^.*TbIdx.*Error CBs\s*\K\d+' $1) # Use if you don't want to check UCI BLER too.
        readarray -t errCBs_array <<< "$errCBs_string"
        for errCBs in "${errCBs_array[@]}"; do
            if [ "$errCBs" -ne "0" ]; then
                echo "Non-zero BLER detected for at least one TB in a cell or UCI - Error Code Blocks = "$errCBs
                #closeMPS
                #return $errCBs
            fi
        done
        # Check for mismatched CBs based on cbErr for PUSCH
        mismatchedCBs_string=$(grep -Po 'Mismatched CBs\s*\K\d+' $1)
        readarray -t mismatchedCBs_array <<< "$mismatchedCBs_string"
        for mismatchedCBs in "${mismatchedCBs_array[@]}"; do
            if [ "$mismatchedCBs" -ne "0" ]; then
                echo "Mismatch detected for at least one TB in a cell or UCI - Mismatched Code Blocks = "$mismatchedCBs
                closeMPS
                return $mismatchedCBs
            fi
        done
        # Check for MismatchedCRC CBs against cbErr for PUSCH
        mismatchedCrcCBs_string=$(grep -Po 'MismatchedCRC CBs\s*\K\d+' $1)
        readarray -t mismatchedCrcCBs_array <<< "$mismatchedCrcCBs_string"
        for mismatchedCrcCBs in "${mismatchedCrcCBs_array[@]}"; do
            if [ "$mismatchedCrcCBs" -ne "0" ]; then
                echo "MismatchedCRC detected for at least one CB in a cell or UCI - MismatchedCRC Code Blocks = "$mismatchedCrcCBs
                closeMPS
                return $mismatchedCrcCBs
            fi
        done
        # Check for MismatchedCRC TBs against tbErr for PUSCH
        mismatchedCrcTBs_string=$(grep -Po 'MismatchedCRC TBs\s*\K\d+' $1)
        readarray -t mismatchedCrcTBs_array <<< "$mismatchedCrcTBs_string"
        for mismatchedCrcTBs in "${mismatchedCrcTBs_array[@]}"; do
            if [ "$mismatchedCrcTBs" -ne "0" ]; then
                echo "MismatchedCRC detected for at least one TB in a cell or UCI - MismatchedCRC Transport Blocks = "$mismatchedCrcTBs
                closeMPS
                return $mismatchedCrcTBs
            fi
        done
        echo ""
    fi
}

OLD_RUN_COMPUTE_SANITIZER=$RUN_COMPUTE_SANITIZER
RUN_COMPUTE_SANITIZER=0;

# Leave this green context test commented out; functionality tested in the subsequent one
# example with green contexts (-n). Do not start MPS yet. CUDA_DEVICE_MAX_CONNECTIONS will be explicitly set (also to 12) from within the test bench in green contexts mode.
#full_cmd="CUDA_DEVICE_MAX_CONNECTIONS=12 $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors-example-phyOnly-4t4r.yaml -r 1 -w 50000 -u 5 -d 0 -m 0 -B --P --Q --X  --M 2,36,4,96,82,14 -L -a --G --b --g $REF_CHECKS -n"
#echo $full_cmd
#eval $full_cmd | tee run.log
#checkmem run.log

# F08 cuPHY + cuMAC, 4T4R 10 peak, 24 slots with green contexts (-n) Do not start MPS yet. Modified SM counts
full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=8 $compute_sanitizer_cmd_prefix $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors-example-macPhyShare-4t4r.yaml -r 1 -w 50000 -u 5 -d 0 -m 1 -B -K 1 --P --Q --X --G --b --g --M 8,24,16,104,68,16,102 -L -a --T --U --D --B $REF_CHECKS -n"
echo $full_cmd
eval $full_cmd | tee run.log
checkmem run.log

# Raise MPS
CUDA_VISIBLE_DEVICES=$GPU_ID CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. nvidia-cuda-mps-control -d

# example 4T4R with MPS
full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=8 $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors-example-phyOnly-4t4r.yaml -r 1 -w 50000 -u 5 -d 0 -m 0 -B --P --Q --X  --M 2,36,4,96,82,14 -L -a --G --b --g --U --D --B $REF_CHECKS"
echo $full_cmd
eval $full_cmd | tee run.log
checkmem run.log
RUN_COMPUTE_SANITIZER=$OLD_RUN_COMPUTE_SANITIZER

# F08 cuPHY + cuMAC, 4T4R 10 peak, 24 slots, use PDSCH timer
full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=8 $compute_sanitizer_cmd_prefix $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors-example-macPhyShare-4t4r.yaml -r 1 -w 50000 -u 5 -d 0 -m 1 -B -K 1 --P --Q --X --G --b --g --M 8,16,16,92,108,16,108 -L -a --T --U --D --B $REF_CHECKS"
echo $full_cmd
eval $full_cmd | tee run.log
checkmem run.log

# F08 cuPHY + cuMAC, 4T4R 10 peak, 24 slots, use MAC internal timer
full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=8 $compute_sanitizer_cmd_prefix $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors-example-macPhyShare-4t4r.yaml -r 1 -w 50000 -u 5 -d 0 -m 1 -B -K 1 --P --Q --X --G --b --g --M 8,16,16,92,108,16,108 -L -a --T --U --D --R --B $REF_CHECKS"
echo $full_cmd
eval $full_cmd | tee run.log
checkmem run.log

# F14 cuPHY, 64TR 2 peak, 30 slots, new mMIMO pattern
# full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=8 $compute_sanitizer_cmd_prefix $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors-example-phyOnly-64tr.yaml -r 1 -w 50000 -u 6 -d 0 -m 1 -B -K 1 --P --Q --X --G --b --g --M 2,12,20,102,82,16 -L -a --Z --U --D $REF_CHECKS"
# echo $full_cmd
# eval $full_cmd | tee run.log
# checkmem run.log

# F01 TCTM
full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=37 CUDA_DEVICE_MAX_CONNECTIONS=16 $compute_sanitizer_cmd_prefix $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors-00.yaml -r 1 -w 30000 -u 4 -d 0 -m 1 -C 4 -S 3 -I 1 --U --D --B $REF_CHECKS"
echo $full_cmd
eval $full_cmd | tee run.log
checkmem run.log


# F01 TCTM Cell-groups

full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=37 CUDA_DEVICE_MAX_CONNECTIONS=16 $compute_sanitizer_cmd_prefix $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors-00.yaml -r 1 -w 30000 -u 4 -d 0 -m 1 -C 4 -S 3 -I 1 --G --b --g --U --D --B $REF_CHECKS"
echo $full_cmd
eval $full_cmd | tee run.log
checkmem run.log

# Currently a memory leak is observed, likely from a library, when PRACH is run.
# So, if RUN_COMPUTE_SANITIZER is enabled, run the F14 testcases twice: once with the sanitizer but with a *noPrach.yaml and
# once without the sanitizer but with the default yaml that contains PRACH

# The prach_workaround* arrays provide 3 options:
# (i) when compute-sanitizer is disable run default yaml
# (ii) compute-sanitizer with special noPrach yaml
# (iii) no compute-sanitizer with default yaml

prach_workaround_cmd=("$compute_sanitizer_cmd_prefix" "$compute_sanitizer_cmd_prefix" "")
prach_workaround_yaml=("" "_noPrach" "")
prach_workaround_ctx=("--P" "" "--P")
prach_workaround_SMs=("2," "" "2,")

# Specify which of  of the above combinations should be executed
if [[ "$RUN_COMPUTE_SANITIZER" == 1 ]]; then
    cmd="1 2"
else
    cmd="0"
fi

OLD_RUN_COMPUTE_SANITIZER=$RUN_COMPUTE_SANITIZER

for cfg in $cmd; do

    new_cmd=${prach_workaround_cmd[$cfg]}
    yaml_suffix=${prach_workaround_yaml[$cfg]}
    prach_ctx=${prach_workaround_ctx[$cfg]}
    prach_SMs=${prach_workaround_SMs[$cfg]}
    #echo "Running new_cmd $new_cmd with yaml_suffix $yaml_suffix, prach_ctx=$prach_ctx and prach_SMs=$prach_SMs"

    #Necessary overwrite as checkmem relies on RUN_COMPUTE_SANITIZER value
    if [ "$new_cmd" == "" ]; then
        RUN_COMPUTE_SANITIZER=0
    else
        RUN_COMPUTE_SANITIZER=$OLD_COMPUTE_SANITIZER
    fi

    # F14 TCTM DDDSU

    full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=16 $new_cmd $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors_F14-02$yaml_suffix.yaml -r 1 -w 30000 -u 3 -d 0 -m 0 --Q --X $prach_ctx --M $prach_SMs\28,16,108,108,108 -L -a --S --U --D --B $REF_CHECKS"
    echo $full_cmd
    eval $full_cmd | tee run.log
    checkmem run.log

    # F14 TCTM DDDSU Cell-groups

    full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=16 $new_cmd $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors_F14_cell_groups-02$yaml_suffix.yaml -r 1 -w 30000 -u 3 -d 0 -m 0 --Q --X $prach_ctx --M $prach_SMs\28,16,108,108,108 -L -a --G --b --g --U --D --B $REF_CHECKS"
    echo $full_cmd
    eval $full_cmd | tee run.log
    checkmem run.log


    # F14 TCTM DDDSUUDDDD
    full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=16 $new_cmd $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors_F14-01$yaml_suffix.yaml -r 1 -w 30000 -u 5 -d 0 -m 0 -B --Q --X $prach_ctx --M $prach_SMs\28,16,108,108,108 -L -a --S --U --D --B $REF_CHECKS"
    echo $full_cmd
    eval $full_cmd | tee run.log
    checkmem run.log


    # F14 TCTM DDDSUUDDDD Cell-groups
    full_cmd="CUDA_MPS_PIPE_DIRECTORY=. CUDA_LOG_DIRECTORY=. CUDA_DEVICE_MAX_CONNECTIONS=16 $new_cmd $BUILD_FOLDER/cubb_gpu_test_bench/cubb_gpu_test_bench -i $YAML_FOLDER/vectors_F14_cell_groups-01$yaml_suffix.yaml -r 1 -w 30000 -u 5 -d 0 -m 0 -B --Q --X $prach_ctx --M $prach_SMs\28,16,108,108,108 -L -a --G --b --g --U --D --B $REF_CHECKS"
    echo $full_cmd
    eval $full_cmd | tee run.log
    checkmem run.log

done

# Close MPS
closeMPS
