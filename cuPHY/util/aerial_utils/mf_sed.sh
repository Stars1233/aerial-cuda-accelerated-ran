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

usage() {
    cat << EOF
Usage: $(basename "$0") [-r] TARGET_DIR [TARGET_DIR ...]

Description:
    This script replaces CUDA and DOCA memory allocation function names with
    MF_ prefixed versions across C/C++/CUDA source files (.cpp, .hpp, .c, .h, .cu, .cuh).

    The following functions are replaced:
    - cuMemAlloc*, cudaMalloc*, cudaHostAlloc
    - doca_gpu_mem_alloc, rte_gpu_mem_alloc
    - cuCtxCreate*, cuGraphInstantiate, cudaGraphInstantiate
    - cudaStreamCreate*, doca_ctx_start
    - cudaFree*, cuMemFree*, cudaFreeHost, cuMemFreeHost
    - doca_gpu_mem_free, rte_gpu_mem_free

Required:
    cuBB_SDK environment variable must be set

Arguments:
    TARGET_DIR    Required. One or more directories to process. Use "all" to process
                  all default directories:
                  - cuPHY/src
                  - cuPHY/examples/common
                  - cuPHY-CP/aerial-fh-driver/lib
                  - cuPHY-CP/cuphydriver

Options:
    -r, --reverse Replace MF_ macros back to original APIs
    -h, --help    Display this help message

Examples:
    # Process all default directories
    $(basename "$0") all

    # Process specific directory
    $(basename "$0") cuPHY/src

    # Process multiple directories
    $(basename "$0") cuPHY/src cuPHY-CP/cuphydriver

    # Reverse MF_ macros in all default directories
    $(basename "$0") -r all

    # Reverse MF_ macros in specific directory
    $(basename "$0") -r cuPHY/src

    # Reverse MF_ macros in multiple directories
    $(basename "$0") -r cuPHY/src cuPHY-CP/cuphydriver

EOF
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

if [ "$cuBB_SDK" = "" ]; then
    echo "Error: Please set cuBB_SDK environment variable first"
    echo ""
    usage
    exit 1
fi

# Parse options
reverse_mode=false
targets=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--reverse)
            reverse_mode=true
            shift
            ;;
        *)
            targets+=("$1")
            shift
            ;;
    esac
done

# Check if at least one TARGET_DIR is provided
if [ ${#targets[@]} -eq 0 ]; then
    echo "Error: At least one TARGET_DIR is required"
    echo ""
    usage
    exit 1
fi

echo "cuBB_SDK=${cuBB_SDK}"

# Function to sed replace in a directory (forward: replace APIs with MF_ macros)
function sed_in_dir_forward {
    dir_path="$1"
    echo "Replacing APIs with MF_ macros in ${dir_path} ..."
    src_files=$(find ${dir_path} -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \))
    for file in ${src_files}; do
        sed -i 's/\bcuMemAlloc\b/MF_CU_MEM_ALLOC/g' ${file}
        sed -i 's/\bcuMemAllocHost\b/MF_CU_MEM_ALLOC_HOST/g' ${file}
        sed -i 's/\bcuMemAllocPitch\b/MF_CU_MEM_ALLOC_PITCH/g' ${file}
        sed -i 's/\bcuMemAllocAsync\b/MF_CU_MEM_ALLOC_ASYNC/g' ${file}
        sed -i 's/\bcuMemAllocFromPoolAsync\b/MF_CU_MEM_ALLOC_FROM_POOL_ASYNC/g' ${file}
        sed -i 's/\bcudaHostAlloc\b/MF_CUDA_HOST_ALLOC/g' ${file}
        sed -i 's/\bcudaMalloc\b/MF_CUDA_MALLOC/g' ${file}
        sed -i 's/\bcudaMalloc3D\b/MF_CUDA_MALLOC_3D/g' ${file}
        sed -i 's/\bcudaMalloc3DArray\b/MF_CUDA_MALLOC_3D_ARRAY/g' ${file}
        sed -i 's/\bcudaMallocArray\b/MF_CUDA_MALLOC_ARRAY/g' ${file}
        sed -i 's/\bcudaMallocHost\b/MF_CUDA_MALLOC_HOST/g' ${file}
        sed -i 's/\bcudaMallocManaged\b/MF_CUDA_MALLOC_MANAGED/g' ${file}
        sed -i 's/\bcudaMallocMipmappedArray\b/MF_CUDA_MALLOC_MIPMAPPED_ARRAY/g' ${file}
        sed -i 's/\bcudaMallocPitch\b/MF_CUDA_MALLOC_PITCH/g' ${file}
        sed -i 's/\bdoca_gpu_mem_alloc\b/MF_DOCA_GPU_MEM_ALLOC/g' ${file}
        sed -i 's/\brte_gpu_mem_alloc\b/MF_RTE_GPU_MEM_ALLOC/g' ${file}
        sed -i 's/\bcuCtxCreate\b/MF_CU_CTX_CREATE/g' ${file}
        sed -i 's/\bcuCtxCreate_v3\b/MF_CU_CTX_CREATE_V3/g' ${file}
        sed -i 's/\bcuGreenCtxCreate\b/MF_CU_GREEN_CTX_CREATE/g' ${file}
        sed -i 's/\bcuGraphInstantiate\b/MF_CU_GRAPH_INSTANTIATE/g' ${file}
        sed -i 's/\bcudaGraphInstantiate\b/MF_CUDA_GRAPH_INSTANTIATE/g' ${file}
        sed -i 's/\bcudaStreamCreateWithFlags\b/MF_CUDA_STREAM_CREATE_WITH_FLAGS/g' ${file}
        sed -i 's/\bcudaStreamCreateWithPriority\b/MF_CUDA_STREAM_CREATE_WITH_PRIORITY/g' ${file}
        sed -i 's/\bdoca_ctx_start\b/MF_DOCA_CTX_START/g' ${file}
        sed -i 's/\bdoca_buf_arr_start\b/MF_DOCA_BUF_ARR_START/g' ${file}
        # GPU memory free functions
        sed -i 's/\bcudaFree\b/MF_CUDA_FREE/g' ${file}
        sed -i 's/\bcudaFreeArray\b/MF_CUDA_FREE_ARRAY/g' ${file}
        sed -i 's/\bcudaFreeMipmappedArray\b/MF_CUDA_FREE_MIPMAPPED_ARRAY/g' ${file}
        sed -i 's/\bcudaFreeAsync\b/MF_CUDA_FREE_ASYNC/g' ${file}
        sed -i 's/\bcuMemFree\b/MF_CU_MEM_FREE/g' ${file}
        sed -i 's/\bcuMemFreeHost\b/MF_CU_MEM_FREE_HOST/g' ${file}
        sed -i 's/\bcuMemFreeAsync\b/MF_CU_MEM_FREE_ASYNC/g' ${file}
        # Host pinned memory free functions
        sed -i 's/\bcudaFreeHost\b/MF_CUDA_FREE_HOST/g' ${file}
        # DOCA and RTE free functions
        sed -i 's/\bdoca_gpu_mem_free\b/MF_DOCA_GPU_MEM_FREE/g' ${file}
        sed -i 's/\brte_gpu_mem_free\b/MF_RTE_GPU_MEM_FREE/g' ${file}
    done
    cd $cuBB_SDK
}

# Function to sed replace in a directory (reverse: replace MF_ macros back to original APIs)
function sed_in_dir_reverse {
    dir_path="$1"
    echo "Replacing MF_ macros back to original APIs in ${dir_path} ..."
    src_files=$(find ${dir_path} -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \))
    for file in ${src_files}; do
        # Replace longer patterns first to avoid partial matches
        # cuMemAlloc* functions
        sed -i 's/\bMF_CU_MEM_ALLOC_FROM_POOL_ASYNC\b/cuMemAllocFromPoolAsync/g' ${file}
        sed -i 's/\bMF_CU_MEM_ALLOC_ASYNC\b/cuMemAllocAsync/g' ${file}
        sed -i 's/\bMF_CU_MEM_ALLOC_PITCH\b/cuMemAllocPitch/g' ${file}
        sed -i 's/\bMF_CU_MEM_ALLOC_HOST\b/cuMemAllocHost/g' ${file}
        sed -i 's/\bMF_CU_MEM_ALLOC\b/cuMemAlloc/g' ${file}
        # cudaMalloc* functions
        sed -i 's/\bMF_CUDA_MALLOC_MIPMAPPED_ARRAY\b/cudaMallocMipmappedArray/g' ${file}
        sed -i 's/\bMF_CUDA_MALLOC_3D_ARRAY\b/cudaMalloc3DArray/g' ${file}
        sed -i 's/\bMF_CUDA_MALLOC_3D\b/cudaMalloc3D/g' ${file}
        sed -i 's/\bMF_CUDA_MALLOC_PITCH\b/cudaMallocPitch/g' ${file}
        sed -i 's/\bMF_CUDA_MALLOC_MANAGED\b/cudaMallocManaged/g' ${file}
        sed -i 's/\bMF_CUDA_MALLOC_HOST\b/cudaMallocHost/g' ${file}
        sed -i 's/\bMF_CUDA_MALLOC_ARRAY\b/cudaMallocArray/g' ${file}
        sed -i 's/\bMF_CUDA_MALLOC\b/cudaMalloc/g' ${file}
        # Other allocation functions
        sed -i 's/\bMF_CUDA_HOST_ALLOC\b/cudaHostAlloc/g' ${file}
        sed -i 's/\bMF_DOCA_GPU_MEM_ALLOC\b/doca_gpu_mem_alloc/g' ${file}
        sed -i 's/\bMF_RTE_GPU_MEM_ALLOC\b/rte_gpu_mem_alloc/g' ${file}
        # Context and graph functions
        sed -i 's/\bMF_CU_GREEN_CTX_CREATE\b/cuGreenCtxCreate/g' ${file}
        sed -i 's/\bMF_CU_CTX_CREATE_V3\b/cuCtxCreate_v3/g' ${file}
        sed -i 's/\bMF_CU_CTX_CREATE\b/cuCtxCreate/g' ${file}
        sed -i 's/\bMF_CU_GRAPH_INSTANTIATE\b/cuGraphInstantiate/g' ${file}
        sed -i 's/\bMF_CUDA_GRAPH_INSTANTIATE\b/cudaGraphInstantiate/g' ${file}
        sed -i 's/\bMF_CUDA_STREAM_CREATE_WITH_PRIORITY\b/cudaStreamCreateWithPriority/g' ${file}
        sed -i 's/\bMF_CUDA_STREAM_CREATE_WITH_FLAGS\b/cudaStreamCreateWithFlags/g' ${file}
        sed -i 's/\bMF_DOCA_CTX_START\b/doca_ctx_start/g' ${file}
        sed -i 's/\bMF_DOCA_BUF_ARR_START\b/doca_buf_arr_start/g' ${file}
        # GPU memory free functions
        sed -i 's/\bMF_CUDA_FREE_MIPMAPPED_ARRAY\b/cudaFreeMipmappedArray/g' ${file}
        sed -i 's/\bMF_CUDA_FREE_ARRAY\b/cudaFreeArray/g' ${file}
        sed -i 's/\bMF_CUDA_FREE_ASYNC\b/cudaFreeAsync/g' ${file}
        sed -i 's/\bMF_CUDA_FREE_HOST\b/cudaFreeHost/g' ${file}
        sed -i 's/\bMF_CUDA_FREE\b/cudaFree/g' ${file}
        sed -i 's/\bMF_CU_MEM_FREE_ASYNC\b/cuMemFreeAsync/g' ${file}
        sed -i 's/\bMF_CU_MEM_FREE_HOST\b/cuMemFreeHost/g' ${file}
        sed -i 's/\bMF_CU_MEM_FREE\b/cuMemFree/g' ${file}
        # DOCA and RTE free functions
        sed -i 's/\bMF_DOCA_GPU_MEM_FREE\b/doca_gpu_mem_free/g' ${file}
        sed -i 's/\bMF_RTE_GPU_MEM_FREE\b/rte_gpu_mem_free/g' ${file}
    done
    cd $cuBB_SDK
}

# Function wrapper that calls the appropriate function based on mode
function sed_in_dir {
    if [ "$reverse_mode" = true ]; then
        sed_in_dir_reverse "$1"
    else
        sed_in_dir_forward "$1"
    fi
}

# Change to cuBB_SDK directory
cd $cuBB_SDK

# Process target directories
for target in "${targets[@]}"; do
    if [ "$target" = "all" ]; then
        # Process all default directories
        sed_in_dir cuPHY/src
        sed_in_dir cuPHY/examples/common
        sed_in_dir cuPHY-CP/aerial-fh-driver/lib
        sed_in_dir cuPHY-CP/cuphydriver
    else
        # Process specific directory
        sed_in_dir "$target"
    fi
done