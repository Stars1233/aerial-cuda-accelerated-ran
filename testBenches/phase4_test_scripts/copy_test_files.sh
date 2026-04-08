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

# if DST is not set explicitly, TVs will be copies to DST=$cuBB_SDK/testVectors

# Identify SCRIPT_DIR
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)

cuBB_SDK=${cuBB_SDK:-$(realpath $SCRIPT_DIR/../..)}

# Source the valid patterns file
source "$SCRIPT_DIR/valid_perf_patterns.sh"

DIR=$PWD
SRC=""
DST="$cuBB_SDK/testVectors"
MAX_CELLS="20"

UUID=$(${cuBB_SDK}/5GModel/get_uuid.sh)

show_usage() {
  echo "Usage: $0 <pattern> [options]"
  echo
  echo "Arguments:"
  echo "  pattern_name                    Name of the pattern (required)"
  echo "  Please provide one of the following patterns as the argument:"
  echo "  ${valid_perf_patterns[*]}"
  echo
  echo "Options:"
  echo "  --src <source_directory>        Specify the full path for the source directory (default: auto-detected using get_uuid.sh)"
  echo "  --dst <destination_directory>   Specify the full path for the destination directory (default: $DST)"
  echo "  --max_cells <max_cells>         Specify the maximum number of cells you will run (default: $MAX_CELLS)"
  echo "  -h, --help                      Show this help message and exit"
  echo
  echo "Examples:"
  echo "  # Copy pattern 46 test vectors using default paths:"
  echo "  $0 46"
  echo
  echo "  # Copy nrSim pattern 90601 test vectors using default paths:"
  echo "  $0 90601"
  echo
  echo "  # Launch pattern for nrSim pattern 3337 does not exist, "
  echo "    use TVnr_3337_gNB_FAPI_s*.h5 to generate launch pattern and copy files using default paths:"
  echo "  $0 3337"
  echo
  echo "  # Copy pattern 59 test vectors with custom source and destination:"
  echo "  $0 59 --src /custom/source/path --dst /custom/dest/path"
  echo
  echo "  # Copy pattern 60a test vectors for 10 cells:"
  echo "  $0 60a --max_cells 10"
  echo
  echo "  # Copy pattern 67a test vectors with all custom options:"
  echo "  $0 67a --src /data/test_vectors --dst /app/vectors --max_cells 15"
  exit 1
}

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --src)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --src option"
                exit 1
            fi
            SRC="$2"
            if [ ! -d "$SRC" ]; then
              echo "Error: Unable to access $SRC"
              exit 1
            fi
            shift 2
            ;;
        --dst)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --dst option"
                exit 1
            fi
            DST="$2"
            if [ ! -d "$DST" ]; then
              echo "Error: Unable to access $DST"
              exit 1
            fi
            shift 2
            ;;
        --max_cells)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "Error: Missing value for --max_cells option"
                exit 1
            fi
            MAX_CELLS="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            # Assume the first non-option argument is the pattern name
            pattern_name=$1 # Do not convert the pattern name to lowercase as it might not work for nrSim cases
            shift
            ;;
    esac
done

if [[ -z "${pattern_name}" ]]; then
  echo "Error: pattern argument missing"
  show_usage
  exit 1
fi

if [[ -z "${SRC}" ]]; then
  echo "--src option is not specified. Using get_uuid.sh to automatically set the source path for copying TVs ..."
  # What test vector set to use?
  #   Check the UUID using get_uuid.sh
  #   If the full version of TVs has been generated - use it
  #   If full is not ready, use compact
  #   If neither full nor compact are there - use full version of develop
  if [[ -e /mnt/cicd_tvs/$UUID/.5gmodel_tv_generated ]]; then
      SRC=/mnt/cicd_tvs/$UUID/GPU_test_input
  elif [[ -e /mnt/cicd_tvs/$UUID/.5gmodel_tv_compact_generated ]]; then
      SRC=/mnt/cicd_tvs/$UUID/compact/GPU_test_input
  else
      SRC=/mnt/cicd_tvs/develop/GPU_test_input
  fi
fi

if cd "$SRC"; then
  echo -n "Copying pattern TVs from: "$SRC" "
else
  echo "Error: Unable to access source directory "$SRC""
  exit 1
fi

if [ -d "$DST" ]; then
  echo "to "$DST""
  if [ ! -d "$DST/multi-cell" ]; then
    mkdir "$DST/multi-cell"
  fi
else
  echo
  echo "Error: Unable to access destination directory "$DST""
  exit 1
fi

#-------------------------------------------------------------------------
display_progress() {
  local total_files="$1"
  local current_file="$2"
  printf "\r[%-50s] %d%%" "$(printf '#%.0s' $(seq 1 $((current_file * 50 / total_files))))" $((current_file * 100 / total_files))
}

# Function to handle copying files based on tv_ids array
copy_files() {
  local pattern_name="$1"
  local current_file=0
  local total_files=0
  local temp_launch_pattern=""

  # Check if pattern_name is valid by either:
  # 1. Being in the predefined valid_patterns array, or
  # 2. Having a matching nrSim pattern file in the source directory
  if [[ ! " ${valid_perf_patterns[*]} " =~ " $pattern_name " ]]; then
     # If not in valid_patterns, check if there's a matching nrSim pattern file
     if ! ls "$SRC"/launch_pattern_nrSim_"$pattern_name".yaml 1> /dev/null 2>&1; then
         # Try to generate the launch pattern using auto_lp.py directly to DST/multi-cell
         echo "No existing launch pattern found. Attempting to generate one..."
         python3 "$cuBB_SDK"/cubb_scripts/auto_lp.py -i "$SRC" --test_case "$pattern_name" -o "$DST"/multi-cell
         if [ $? -ne 0 ]; then
             echo "Error: Invalid pattern name '$pattern_name'."
             echo "Pattern must either:"
             echo "  1. Be one of: ${valid_perf_patterns[*]}"
             echo "  2. Have a matching file: $SRC/launch_pattern_nrSim_$pattern_name.yaml"
             echo "  3. Have matching test vector: $SRC/TVnr_${pattern_name}_gNB_FAPI_s*.h5"
             exit 1
         fi
         temp_launch_pattern="$DST/multi-cell/launch_pattern_nrSim_$pattern_name.yaml"
     fi
  fi

  pattern="[[:alnum:]_]+\.h5";
  
  # First check if it's an nrSim pattern (either existing or temp)
  if [ -n "$temp_launch_pattern" ]; then
      # Get list of TVs from generated pattern file
      total_files_list=`grep -Eo $pattern "$temp_launch_pattern" | sort -u`
      total_files=`echo $total_files_list | wc -w`
      
      echo "Copying test vectors from generated launch pattern file to "$DST"/ ..."
  
  elif ls "$SRC"/launch_pattern_nrSim_"$pattern_name".yaml 1> /dev/null 2>&1; then
      # Copy the existing nrSim launch pattern YAML file
      cp "$SRC"/launch_pattern_nrSim_"$pattern_name".yaml "$DST"/multi-cell/.

      # Get list of TVs from nrSim pattern file
      total_files_list=`grep -Eo $pattern "$SRC"/launch_pattern_nrSim_"$pattern_name".yaml | sort -u`
      total_files=`echo $total_files_list | wc -w`
      
      echo "Copying test vectors from nrSim launch pattern file launch_pattern_nrSim_$pattern_name.yaml to "$DST"/ ..."
  
  # If not nrSim, check for F08 pattern
  elif [ -e "$SRC"/launch_pattern_F08_"$MAX_CELLS"C_"$pattern_name".yaml ]; then
      # Copy the F08 launch pattern YAML file(s) for all cell counts
      cp "$SRC"/launch_pattern_F08*_"$pattern_name".yaml "$DST"/multi-cell/.

      # Get list of TVs from the launch pattern file with the max. cell count
      total_files_list=`grep -Eo $pattern "$SRC"/launch_pattern_F08_"$MAX_CELLS"C_"$pattern_name".yaml | sort -u`
      total_files=`echo $total_files_list | wc -w`
      
      echo "Copying test vectors from F08 launch pattern file launch_pattern_F08_$MAX_CELLS""C_$pattern_name.yaml to "$DST"/ ..."
  
  else
      echo "Error: No matching launch pattern file found for pattern '$pattern_name'"
      echo "Looked for:"
      echo "  - launch_pattern_nrSim_$pattern_name.yaml"
      echo "  - launch_pattern_F08_$MAX_CELLS""C_$pattern_name.yaml"
      echo "Consider using the --max_cells option with a different value (current value: $MAX_CELLS)."
      show_usage
  fi

  # Copy the test vectors
  for file in $total_files_list; do
      cp "$SRC"/"$file" "$DST"/; #  a message will be printed if file does not exist
      ((current_file++))
      display_progress "$total_files" "$current_file"
  done
}

#-------------------------------------------------------------------------
cp cuPhyChEstCoeffs.h5 "$DST"/.

copy_files "$pattern_name"

cd $DIR
echo ""
