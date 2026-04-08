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

script_name="cumac_tests.sh"
test=""
gpu=""
log=""
smoke="false"

# add -m smoke test option, when true, will run smoke tests (not exhaustive tests) on 4t4r and 64tr tests
while getopts ":t:T:l:L:g:G:m:M:" opt; do
  case $opt in
    t|T)
      test=$OPTARG
      ;;
    l|L)
      log=${OPTARG%/}  # Remove trailing slash if present
      ;;
    g|G)
      gpu=$OPTARG
      ;;
    m|M)
      smoke=$OPTARG
      ;;
    \?)
      echo "Unknown parameter: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$cuBB_SDK" ]]; then
   echo "Error: cuBB_SDK environment variable is not set" >&2
   exit 1
fi

script_folder="${cuBB_SDK}/cuMAC/scripts"

if [[ -z "$test" || -z "$log" ]]; then
   echo "test and log are required parameters."
   echo "Usage: $script_name -t <test> -l <log> -g <gpu> -m <true|false>"
   echo "   test: 4t4r, tdl, cdl, drl, srs, 64tr, 4t4r_fast_fading, pfmsort"
   echo "   log: log folder"
   echo "   gpu: gpu device id, if not provided,then will be 0"
   echo "   smoke (-m/-M): enable smoke test, true|false, if not provided,then will be false, which runs exhaustive sanity tests, but will consume more time. Used only for 4t4r and 64tr tests"
   exit 1
fi

if [ -z "$gpu" ]; then
   test_cmd="python3 ${script_folder}/run_cumac_test.py -s ${cuBB_SDK}"
else
   test_cmd="python3 ${script_folder}/run_cumac_test.py -g ${gpu} -s ${cuBB_SDK}"
fi
echo "test cmd: ${test_cmd}"
mkdir -p ${log}/${test}
echo "log folder: ${log}/${test}"

if [[ "$test" == "4t4r" || "$test" == "4t4r_fast_fading" ]]; then
   ant="4"
   # DL is always supported, but those are the only TVs that support both DL and UL tests
   tvs="0004 0008 0012 0016 0020 0024 0028 0032" 
   # 4t4r test will take 3.2 hours when numSimChnRlz = 2000 on tvs "0008 0016 0020 0024 0028 0032"  
   # tvs_for_smoke can even be reduced from "0008 0016 0020 0024 0028 0032" to  "0016 0028 0032" 
   tvs_for_smoke="0008 0016 0020 0024 0028 0032"  

   for dir in "DL" "UL"; do
      generate_cmd="python3 ${script_folder}/generate_tv.py -d ${dir} -l ${log}/${test} -a ${ant} -s ${cuBB_SDK}"

      if [[ "$smoke" == "true" ]]; then
            for idx in $tvs_for_smoke; do
               gen_cmd="${generate_cmd} -i ${idx}"
               echo "Start to generate 4T4R ${dir} smoke TV ${idx} ..."
               cd "$script_folder" && eval "$gen_cmd"
            done
      else 
         if [ "$test" == "4t4r_fast_fading" ]; then
            generate_cmd="${generate_cmd} -o 4"
         fi

         echo "Start to generate 4T4R ${dir} TVs ..."
         cd "$script_folder" && eval "$generate_cmd"
      fi 

      # Temporarily commented out test cases
      for case in "UE_selection" "PRG_allocation" "Layer_selection" "MCS_selection" "Scheduler_pipeline"; do
         echo "Start to run cumac ${case} for 4T4R ${dir} all tests ..."

         if [ "$smoke" == "false" ]; then
            if [ "$dir" == "UL" ]; then
               for tv in $tvs; do
                  cd "$script_folder" && eval "$test_cmd -d ${dir} -l ${log}/${test} -i ${tv} -a ${ant} -t ${case}" # run all TVs that support UL for non-smoke test
               done
            else
               cd "$script_folder" && eval "$test_cmd -d ${dir} -l ${log}/${test} -i all -a ${ant} -t ${case}"  # run all TVs for DL tests for non-smoke test
            fi
         else
            for tv in $tvs_for_smoke; do
               cd "$script_folder" && eval "$test_cmd -d ${dir} -l ${log}/${test} -i ${tv} -a ${ant} -t ${case}" # In smoke test, only run these TVs for both DL and UL tests
            done
         fi
      done
   done

elif [[ "$test" == "tdl" || "$test" == "cdl" ]]; then
   cases="f1 f2"
   [ "$test" == "cdl" ] && cases="f3 f4"

   for tv in {0001..0016}; do
      for case in $cases; do
         echo "Start to run cumac ${case} test ..."
         if [ "$smoke" == "true" ]; then
            cd "$script_folder" && eval "$test_cmd -o ${case} -l ${log}/${test} -i ${tv} --allow-bigger-cpu-gpu-gap"
         else
            cd "$script_folder" && eval "$test_cmd -o ${case} -l ${log}/${test} -i ${tv}"
         fi
      done
   done
elif [[ "$test" == "drl" ]];then
   tv_folder="${cuBB_SDK}/cuMAC/examples/ml/testVectors/drlMcsSelection"
   case="0001"
   cd "$script_folder" && $test_cmd -o "$test" -l "${log}/${test}" -i "$case" -tv "$tv_folder"

elif [[ "$test" == "64tr" ]];then
   echo "Start to run cumac $test test ..."
   if [ "$smoke" == "true" ]; then
      cd "$script_folder" && python3 cumac_64tr_test.py --execute --log-dir ${log}/${test} --smoke
   else
      cd "$script_folder" && python3 cumac_64tr_test.py --execute --log-dir ${log}/${test}
   fi

elif [[ "$test" == "srs" ]];then
   echo "Start to run cumac $test test with default config file ..."
   cd $script_folder && $test_cmd -o $test -l ${log}/${test} -i all

elif [[ "$test" == "pfmsort" ]];then
   echo "Start to run cumac $test test ..."
   if [[ "$smoke" == "true" ]]; then
      # with smoke, seeds are reduced to [0, 1, 2]
      cd "$script_folder" && $test_cmd -o $test -l ${log}/${test} -i all --smoke
   else
      cd "$script_folder" && $test_cmd -o $test -l ${log}/${test} -i all
   fi
fi
