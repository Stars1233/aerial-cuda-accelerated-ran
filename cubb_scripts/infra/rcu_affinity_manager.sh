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

# Define colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BRIGHT_CYAN='\033[1;36m'  # Bright/Bold Cyan
BRIGHT_MAGENTA='\033[1;35m'  # Bright/Bold Magenta
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'  # Add red for warnings/errors
NC='\033[0m' # No Color

# Change the default to summarize
SUMMARIZE_OFFLOAD=true
VERBOSE_OUTPUT=false
WRITE_MODE=false
CORE_LIST=""

# Function to display usage information
show_help() {
    echo "Usage: $0 [-v] [-h] [-w -c CORE_LIST]"
    echo "  -v  Verbose mode: show each RCU offload process individually"
    echo "  -h  Display this help message"
    echo "  -w  Write mode: attempt to set CPU affinity mask for processes"
    echo "  -c  Core list: comma-separated list of cores to pin processes to (required with -w)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Default: summarize RCU processes"
    echo "  $0 -v                 # Show verbose RCU process information"
    echo "  $0 -w -c 0,1,2,3      # Set RCU processes to only run on cores 0-3"
    exit 0
}

# Parse command line arguments
while getopts "vhwc:" opt; do
  case $opt in
    v) SUMMARIZE_OFFLOAD=false ; VERBOSE_OUTPUT=true ;;  # -v sets verbose mode
    h) show_help ;;                                      # -h shows help and exits
    w) WRITE_MODE=true ;;                                # -w enables write mode
    c) CORE_LIST="$OPTARG" ;;                            # -c sets the core list
    *) echo "Usage: $0 [-v] [-h] [-w -c CORE_LIST]" >&2
       echo "Use -h for detailed help" >&2
       exit 1 ;;
  esac
done

# Validate that write mode has core list specified
if [ "$WRITE_MODE" = true ] && [ -z "$CORE_LIST" ]; then
    echo -e "${RED}Error: Write mode (-w) requires a core list (-c)${NC}" >&2
    echo "Example: $0 -w -c 0,1,2,3" >&2
    exit 1
fi

# If core list is specified without write mode, warn user
if [ "$WRITE_MODE" = false ] && [ -n "$CORE_LIST" ]; then
    echo -e "${YELLOW}Warning: Core list (-c) specified without write mode (-w)${NC}" >&2
    echo "Core list will be ignored. Use -w flag to set affinities." >&2
fi

# If we're in write mode, automatically enable verbose mode for RCU offload processes
if [ "$WRITE_MODE" = true ] && [ -n "$CORE_LIST" ]; then
    SUMMARIZE_OFFLOAD=false
    echo -e "${GREEN}Note: Enabling verbose mode to show write results for all processes${NC}"
fi

# Get total number of CPUs for formatting the mask properly
TOTAL_CPUS=$(nproc)
MAX_CPU=$((TOTAL_CPUS - 1))
MASK_WIDTH=$(((MAX_CPU + 4) / 4))  # Hex digits needed to represent all CPUs

# Function to convert hex mask to a more readable format showing CPU numbers
convert_mask_to_cpus() {
    local mask=$1
    local cpus=""
    
    # Convert hex to binary
    mask_bin=$(echo "obase=2; ibase=16; ${mask^^}" | bc | sed 's/^0*//')
    
    # Pad with leading zeros to match CPU count
    while [ ${#mask_bin} -lt $TOTAL_CPUS ]; do
        mask_bin="0$mask_bin"
    done
    
    # Reverse the string to match CPU numbering (LSB = CPU 0)
    mask_bin=$(echo $mask_bin | rev)
    
    # Find which CPUs are enabled (1 in the mask)
    for i in $(seq 0 $MAX_CPU); do
        if [ "${mask_bin:$i:1}" = "1" ]; then
            cpus="$cpus $i"
        fi
    done
    
    echo $cpus
}

# Function to print process affinity information
print_process_affinity() {
    local pid=$1
    local proc_name=$2
    local color=$3
    
    # Get current CPU the process is running on
    current_cpu=$(ps -o psr= -p $pid)
    
    # Get affinity mask
    affinity_info=$(taskset -p $pid 2>/dev/null)
    if [ $? -eq 0 ]; then
        mask=$(echo "$affinity_info" | grep -oP 'current affinity mask: \K.*')
        
        # Pad the mask with leading zeros to ensure consistent width
        padded_mask=$(printf "%0${MASK_WIDTH}s" "$mask" | tr ' ' '0')
        
        # Convert mask to CPU list
        cpu_list=$(convert_mask_to_cpus $mask)
        
        # Truncate CPU list if it's too long
        if [ ${#cpu_list} -gt 30 ]; then
            cpu_list_short="${cpu_list:0:27}..."
        else
            cpu_list_short="$cpu_list"
        fi
        
        # If in write mode, try to set the affinity
        if [ "$WRITE_MODE" = true ]; then
            # Store original values
            original_mask=$padded_mask
            original_cpu_list=$cpu_list_short
            
            # Try to set affinity
            if taskset -pc $CORE_LIST $pid > /dev/null 2>&1; then
                # Get new affinity
                new_affinity_info=$(taskset -p $pid 2>/dev/null)
                new_mask=$(echo "$new_affinity_info" | grep -oP 'current affinity mask: \K.*')
                new_padded_mask=$(printf "%0${MASK_WIDTH}s" "$new_mask" | tr ' ' '0')
                new_cpu_list=$(convert_mask_to_cpus $new_mask)
                
                if [ ${#new_cpu_list} -gt 30 ]; then
                    new_cpu_list_short="${new_cpu_list:0:27}..."
                else
                    new_cpu_list_short="$new_cpu_list"
                fi
                
                # Print with all columns - status last
                printf "${color}%-30s %-8s %-${MASK_WIDTH}s %-${MASK_WIDTH}s %s ${GREEN}[SUCCESS]${NC}\n" \
                    "$proc_name" "$pid" "$original_mask" "$new_padded_mask" \
                    "$new_cpu_list_short (on CPU $current_cpu)"
            else
                # Print with failed status and original values
                printf "${color}%-30s %-8s %-${MASK_WIDTH}s %-${MASK_WIDTH}s %s ${RED}[FAILED]${NC}\n" \
                    "$proc_name" "$pid" "$original_mask" "$original_mask" \
                    "$original_cpu_list (on CPU $current_cpu)"
            fi
        else
            # Just display information (no write mode)
            printf "${color}%-30s %-8s %-${MASK_WIDTH}s %s (on CPU %d)${NC}\n" \
                "$proc_name" "$pid" "$padded_mask" "$cpu_list_short" "$current_cpu"
        fi
    else
        # Could not get affinity
        if [ "$WRITE_MODE" = true ]; then
            printf "${YELLOW}%-30s %-8s %-${MASK_WIDTH}s %-${MASK_WIDTH}s %s ${RED}[FAILED]${NC}\n" \
                "$proc_name" "$pid" "ERROR" "ERROR" "Could not get affinity"
        else
            printf "${YELLOW}%-30s %-8s %-${MASK_WIDTH}s %s${NC}\n" \
                "$proc_name" "$pid" "ERROR" "Could not get affinity"
        fi
    fi
}

# Print header function
print_header() {
    local title=$1
    echo
    echo -e "${GREEN}=== $title ===${NC}"
    echo
    
    if [ "$WRITE_MODE" = true ]; then
        # Header with columns for before/after and status at the end
        printf "%-30s %-8s %-${MASK_WIDTH}s %-${MASK_WIDTH}s %s\n" \
            "Process Name" "PID" "Old Affinity" "New Affinity" "CPUs and Status"
        printf "%-30s %-8s %-${MASK_WIDTH}s %-${MASK_WIDTH}s %s\n" \
            "$(printf '%0.s-' {1..30})" "$(printf '%0.s-' {1..8})" \
            "$(printf '%0.s-' {1..${MASK_WIDTH}})" "$(printf '%0.s-' {1..${MASK_WIDTH}})" \
            "$(printf '%0.s-' {1..50})"
    else
        # Original header
        printf "%-30s %-8s %-${MASK_WIDTH}s %s\n" "Process Name" "PID" "Affinity Mask" "CPUs"
        printf "%-30s %-8s %-${MASK_WIDTH}s %s\n" \
            "$(printf '%0.s-' {1..30})" "$(printf '%0.s-' {1..8})" \
            "$(printf '%0.s-' {1..${MASK_WIDTH}})" "$(printf '%0.s-' {1..40})"
    fi
}

# Main script starts here
echo -e "${GREEN}RCU Process Affinity Information${NC}"

# 1. Core RCU Processes
print_header "Core RCU Processes"

# Define the core RCU processes to check
CORE_RCU_PROCESSES=(
    "rcu_gp"
    "rcu_par_gp"
    "rcu_preempt"
    "rcu_tasks_kthread"
    "rcu_tasks_rude_kthread"
    "rcu_tasks_trace_kthread"
)

# Check each core RCU process
for proc in "${CORE_RCU_PROCESSES[@]}"; do
    # Find PIDs for this process
    pids=$(ps -eo pid,comm | grep -E "^[[:space:]]*[0-9]+ ${proc}$" | awk '{print $1}')
    
    if [ -z "$pids" ]; then
        printf "${YELLOW}%-30s %-8s %-${MASK_WIDTH}s %s${NC}\n" "$proc" "N/A" "N/A" "Process not found"
        continue
    fi
    
    for pid in $pids; do
        print_process_affinity $pid "$proc" $BRIGHT_CYAN
    done
done

# 2. RCU-related kworker threads
print_header "RCU-related kworker Threads"

# Find kworker threads handling RCU tasks
kworker_pids=$(ps -eo pid,comm | grep "kworker/.*rcu\|kworker/R-rcu" | awk '{print $1}')

if [ -z "$kworker_pids" ]; then
    printf "${YELLOW}%-30s %-8s %-${MASK_WIDTH}s %s${NC}\n" "kworker-rcu" "N/A" "N/A" "No RCU kworker threads found"
else
    for pid in $kworker_pids; do
        # Get the full command name
        comm=$(ps -p $pid -o comm=)
        # Use the same process_affinity function for consistent formatting
        print_process_affinity $pid "$comm" $CYAN
    done
fi

# 3. RCU Offload Processes
print_header "RCU Offload Processes"

# Define the RCU offload processes to check
OFFLOAD_RCU_PROCESSES=(
    "rcuop"
    "rcuog"
)

if [ "$SUMMARIZE_OFFLOAD" = true ]; then
    # Summarized version - group by affinity mask
    for proc_prefix in "${OFFLOAD_RCU_PROCESSES[@]}"; do
        # Find all processes with this prefix
        pids=$(ps -eo pid,comm | grep -E "^[[:space:]]*[0-9]+ ${proc_prefix}[0-9/]+" | awk '{print $1}')
        
        if [ -z "$pids" ]; then
            printf "${YELLOW}%-30s %-8s %-${MASK_WIDTH}s %s${NC}\n" "${proc_prefix}*" "N/A" "N/A" "No ${proc_prefix} processes found"
            continue
        fi
        
        # Create associative array to store mask -> process list mapping
        declare -A mask_to_procs
        declare -A mask_to_cpulist
        
        # Group processes by affinity mask
        for pid in $pids; do
            comm=$(ps -p $pid -o comm=)
            
            # Get affinity mask
            affinity_info=$(taskset -p $pid 2>/dev/null)
            if [ $? -eq 0 ]; then
                mask=$(echo "$affinity_info" | grep -oP 'current affinity mask: \K.*')
                padded_mask=$(printf "%0${MASK_WIDTH}s" "$mask" | tr ' ' '0')
                
                # Add to the list for this mask
                if [ -z "${mask_to_procs[$padded_mask]}" ]; then
                    mask_to_procs[$padded_mask]="$comm"
                    # Store CPU list for this mask
                    cpu_list=$(convert_mask_to_cpus $mask)
                    if [ ${#cpu_list} -gt 30 ]; then
                        mask_to_cpulist[$padded_mask]="${cpu_list:0:27}..."
                    else
                        mask_to_cpulist[$padded_mask]="$cpu_list"
                    fi
                else
                    mask_to_procs[$padded_mask]="${mask_to_procs[$padded_mask]}, $comm"
                fi
            fi
        done
        
        # Print the summary
        for mask in "${!mask_to_procs[@]}"; do
            # Count the number of processes
            proc_count=$(echo "${mask_to_procs[$mask]}" | tr ',' '\n' | wc -l)
            
            if [ "$WRITE_MODE" = true ]; then
                printf "${CYAN}%-30s %-8s %-${MASK_WIDTH}s %-${MASK_WIDTH}s %s ${YELLOW}[SUMMARY]${NC}\n" \
                    "${proc_prefix}* ($proc_count processes)" "multiple" "$mask" "$mask" "${mask_to_cpulist[$mask]}"
            else
                printf "${CYAN}%-30s %-8s %-${MASK_WIDTH}s %s${NC}\n" \
                    "${proc_prefix}* ($proc_count processes)" "multiple" "$mask" "${mask_to_cpulist[$mask]}"
            fi
            
            # Print the first few process names as a sample
            sample=$(echo "${mask_to_procs[$mask]}" | tr ',' '\n' | head -5 | tr '\n' ',' | sed 's/,/, /g' | sed 's/, $//')
            if [ $proc_count -gt 5 ]; then
                sample="$sample, ..."
            fi
            printf "${CYAN}  Sample: %s${NC}\n" "$sample"
        done
        
        # Clean up
        unset mask_to_procs
        unset mask_to_cpulist
    done
else
    # Detailed version - show each process
    for proc_prefix in "${OFFLOAD_RCU_PROCESSES[@]}"; do
        # Find PIDs for processes starting with this prefix
        pids=$(ps -eo pid,comm | grep -E "^[[:space:]]*[0-9]+ ${proc_prefix}[0-9/]+" | awk '{print $1}')
        
        if [ -z "$pids" ]; then
            printf "${YELLOW}%-30s %-8s %-${MASK_WIDTH}s %s${NC}\n" "${proc_prefix}*" "N/A" "N/A" "No ${proc_prefix} processes found"
            continue
        fi
        
        for pid in $pids; do
            # Get the full command name
            comm=$(ps -p $pid -o comm=)
            # Use the process_affinity function for consistent formatting
            print_process_affinity $pid "$comm" $CYAN
        done
    done
fi

# 4. Summary statistics
print_header "RCU Process Distribution Summary"

# Get all RCU processes and their current CPUs
if [ "$WRITE_MODE" = true ]; then
    echo -e "${BRIGHT_MAGENTA}CPU distribution of RCU processes after affinity changes:${NC}"
else
    echo -e "${BRIGHT_MAGENTA}CPU distribution of RCU processes:${NC}"
fi
echo -e "${BRIGHT_MAGENTA}--------------------------------${NC}"

# Get all RCU processes (core + kworkers)
ps -eo pid,psr,comm | grep -E "(^|\s)rcu|kworker.*rcu" | sort -n -k2 | while read pid cpu comm; do
    # Count processes per CPU
    if [[ ! -z "$cpu" && "$cpu" =~ ^[0-9]+$ ]]; then
        echo "CPU $cpu: $comm (PID $pid)"
    fi
done | awk '{
    cpu[$2]++; 
    if (!seen[$2]++) {
        cpus[count++] = $2
    }
}
END {
    for (i=0; i<count; i++) {
        c = cpus[i]
        printf "'"${BRIGHT_MAGENTA}"'CPU %-3s: %d RCU processes'"${NC}"'\n", c, cpu[c]
    }
}'

echo
echo -e "${GREEN}=== End of RCU Process Affinity Information ===${NC}"
