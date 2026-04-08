/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MEMFOOT_GLOBAL_H
#define MEMFOOT_GLOBAL_H

#include <stdint.h>
#include <stddef.h>
#include <libgen.h>
#include <string.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum {
    MF_TAG_CUPHY_OTHER,
    MF_TAG_PHYDRV_OTHER,
    MF_TAG_FH_DOCA_RX,
    MF_TAG_FH_DOCA_TX,
    MF_TAG_FH_GPU_COMM,
    MF_TAG_FH_PEER,
    MF_TAG_OTHER,
    MF_TAG_MAX_NUM
} memfoot_tag_t;

typedef enum {
    MF_MODULE_CUPHY,
    MF_MODULE_PHYDRV,
    MF_MODULE_FH,
    MF_MODULE_OTHER,
    MF_MODULE_MAX_NUM
} memfoot_module_t;

void memfoot_add_gpu_size(memfoot_tag_t tag_id, int64_t size);

void memfoot_global_get_mem_info(const char* file, int line, int log_tag, const char* func_caller);
void memfoot_global_gpu_size_check(const char* file, int line, int log_tag, const char* func_caller);

int memfoot_global_gpu_size_add(const char* file, int line, int log_tag, const char* func_caller, const char* func_alloc, size_t alloc_size, int err_code);
int memfoot_global_cpu_size_add(const char* file, int line, int log_tag, const char* func_caller, const char* func_alloc, size_t alloc_size, int err_code);

/*
* GPU memory deallocation tracking functions. Since no memory size is provided
* as parameter, these functions check the actual freed memory size by cudaMemGetInfo.
*/
int memfoot_global_gpu_size_sub(const char* file, int line, int log_tag, const char* func_caller, const char* func_free, int err_code);
int memfoot_global_cpu_size_sub(const char* file, int line, int log_tag, const char* func_caller, const char* func_free, int err_code);

void memfoot_global_print_all();

/* For debugging, add this macro before and after the function which was
 * suspected to allocate GPU memory, will see "untracked" size in the log
 * if there's untracked memory allocated.
 */
#define MF_GPU_MEMFOOT_CHECK() memfoot_global_gpu_size_check(__FILE__, __LINE__, __MF_LOG_TAG__, __func__)

// Helper macro to get log_tag (TAG if defined, otherwise 0)
#ifdef TAG
#define __MF_LOG_TAG__ TAG
#else
#define __MF_LOG_TAG__ 0
#endif

/* Common macros for tracking the memory allocation */
// cuda prefixed functions
#define MF_CUDA_DEVICE_ALLOC_TRACK(func, size, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (cudaError_t)memfoot_global_gpu_size_add(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, size, (int)func(__VA_ARGS__)))
#define MF_CUDA_HOST_ALLOC_TRACK(func, size, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (cudaError_t)memfoot_global_cpu_size_add(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, size, (int)func(__VA_ARGS__)))
// cu prefixed functions
#define MF_CU_DEVICE_ALLOC_TRACK(func, size, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (CUresult)memfoot_global_gpu_size_add(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, size, (int)func(__VA_ARGS__)))
#define MF_CU_HOST_ALLOC_TRACK(func, size, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (CUresult)memfoot_global_cpu_size_add(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, size, (int)func(__VA_ARGS__)))
// doca allocation functions
#define MF_DOCA_DEVICE_ALLOC_TRACK(func, size, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (doca_error_t)memfoot_global_gpu_size_add(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, size, (int)func(__VA_ARGS__)))
// rte allocation functions
#define MF_RTE_DEVICE_ALLOC_TRACK(func, size, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (rte_error_t)memfoot_global_gpu_size_add(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, size, (int)func(__VA_ARGS__)))

/* Common macros for tracking the memory deallocation */
// cuda prefixed functions
#define MF_CUDA_DEVICE_FREE_TRACK(func, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (cudaError_t)memfoot_global_gpu_size_sub(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, (int)func(__VA_ARGS__)))
#define MF_CUDA_HOST_FREE_TRACK(func, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (cudaError_t)memfoot_global_cpu_size_sub(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, (int)func(__VA_ARGS__)))
// cu prefixed functions
#define MF_CU_DEVICE_FREE_TRACK(func, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (CUresult)memfoot_global_gpu_size_sub(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, (int)func(__VA_ARGS__)))
#define MF_CU_HOST_FREE_TRACK(func, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (CUresult)memfoot_global_cpu_size_sub(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, (int)func(__VA_ARGS__)))
// doca free functions
#define MF_DOCA_DEVICE_FREE_TRACK(func, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (doca_error_t)memfoot_global_gpu_size_sub(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, (int)func(__VA_ARGS__)))
// rte free functions
#define MF_RTE_DEVICE_FREE_TRACK(func, ...) (memfoot_global_get_mem_info(__FILE__, __LINE__, __MF_LOG_TAG__, __func__), (rte_error_t)memfoot_global_gpu_size_sub(__FILE__, __LINE__, __MF_LOG_TAG__, __func__, #func, (int)func(__VA_ARGS__)))

/* Helper macros for calculating the size of the memory allocation */
#define MF_SIZEOF_EXTENT(extent) ((extent.width == 0 ? 1 : extent.width) * (extent.height == 0 ? 1 : extent.height) * (extent.depth == 0 ? 1 : extent.depth))
#define MF_SIZEOF_W_H(width, height) ((width == 0 ? 1 : width) * (height == 0 ? 1 : height))

/* Implicitly GPU memory allocation functions */

// doca_ctx_start
#define MF_DOCA_CTX_START(...) MF_DOCA_DEVICE_ALLOC_TRACK(doca_ctx_start, 0, ##__VA_ARGS__)
// doca_buf_arr_start
#define MF_DOCA_BUF_ARR_START(...) MF_DOCA_DEVICE_ALLOC_TRACK(doca_buf_arr_start, 0, ##__VA_ARGS__)

// cudaStreamCreateWithFlags
#define MF_CUDA_STREAM_CREATE_WITH_FLAGS(...) MF_CUDA_DEVICE_ALLOC_TRACK(cudaStreamCreateWithFlags, 0, ##__VA_ARGS__)
// cudaStreamCreateWithPriority
#define MF_CUDA_STREAM_CREATE_WITH_PRIORITY(...) MF_CUDA_DEVICE_ALLOC_TRACK(cudaStreamCreateWithPriority, 0, ##__VA_ARGS__)

// cuGraphInstantiate
#define MF_CU_GRAPH_INSTANTIATE(...) MF_CU_DEVICE_ALLOC_TRACK(cuGraphInstantiate, 0, ##__VA_ARGS__)
// cudaGraphInstantiate
#define MF_CUDA_GRAPH_INSTANTIATE(...) MF_CUDA_DEVICE_ALLOC_TRACK(cudaGraphInstantiate, 0, ##__VA_ARGS__)

// cuCtxCreate
#define MF_CU_CTX_CREATE(...) MF_CU_DEVICE_ALLOC_TRACK(cuCtxCreate, 0, ##__VA_ARGS__)
// cuCtxCreate_v3
#define MF_CU_CTX_CREATE_V3(...) MF_CU_DEVICE_ALLOC_TRACK(cuCtxCreate_v3, 0, ##__VA_ARGS__)
// cuGreenCtxCreate
#define MF_CU_GREEN_CTX_CREATE(...) MF_CU_DEVICE_ALLOC_TRACK(cuGreenCtxCreate, 0, ##__VA_ARGS__)

/* Explicitly GPU memory allocation functions */

// cuMemAlloc
#define MF_CU_MEM_ALLOC(ptr, size) MF_CU_DEVICE_ALLOC_TRACK(cuMemAlloc, size, ptr, size)
// cuMemAllocPitch
#define MF_CU_MEM_ALLOC_PITCH(ptr, pitch, width, height, elemSize) MF_CU_DEVICE_ALLOC_TRACK(cuMemAllocPitch, MF_SIZEOF_W_H(width, height) * elemSize, ptr, pitch, width, height, elemSize)
// cuMemAllocAsync
#define MF_CU_MEM_ALLOC_ASYNC(ptr, size, stream) MF_CU_DEVICE_ALLOC_TRACK(cuMemAllocAsync, size, ptr, size, stream)
// cuMemAllocFromPoolAsync
#define MF_CU_MEM_ALLOC_FROM_POOL_ASYNC(ptr, size, pool, stream) MF_CU_DEVICE_ALLOC_TRACK(cuMemAllocFromPoolAsync, size, ptr, size, pool, stream)

// cudaMalloc
#define MF_CUDA_MALLOC(ptr, size) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMalloc, size, ptr, size)
// cudaMalloc3D
#define MF_CUDA_MALLOC_3D(ptr, extent) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMalloc3D, MF_SIZEOF_EXTENT(extent), ptr, extent)
// cudaMalloc3DArray
#define MF_CUDA_MALLOC_3D_ARRAY(ptr, desc, extent, ...) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMalloc3DArray, MF_SIZEOF_EXTENT(extent), ptr, desc, extent, ##__VA_ARGS__)
// cudaMallocArray
#define MF_CUDA_MALLOC_ARRAY(ptr, desc, width, height, flags) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMallocArray, MF_SIZEOF_W_H(width, height), ptr, desc, width, height, flags)
// cudaMallocManaged
#define MF_CUDA_MALLOC_MANAGED(ptr, size, ...) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMallocManaged, size, ptr, size, ##__VA_ARGS__)
// cudaMallocMipmappedArray
#define MF_CUDA_MALLOC_MIPMAPPED_ARRAY(ptr, desc, extent, numLevels, ...) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMallocMipmappedArray, MF_SIZEOF_EXTENT(extent), ptr, desc, extent, numLevels, ##__VA_ARGS__)
// cudaMallocPitch
#define MF_CUDA_MALLOC_PITCH(ptr, pitch, width, height) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMallocPitch, MF_SIZEOF_W_H(width, height), ptr, pitch, width, height)
// cudaMallocAsync
#define MF_CUDA_MALLOC_ASYNC(ptr, size, stream) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMallocAsync, size, ptr, size, stream)
// cudaMallocFromPoolAsync
#define MF_CUDA_MALLOC_FROM_POOL_ASYNC(ptr, size, pool, stream) MF_CUDA_DEVICE_ALLOC_TRACK(cudaMallocFromPoolAsync, size, ptr, size, pool, stream)

// doca_gpu_mem_alloc
#define MF_DOCA_GPU_MEM_ALLOC(gpu_dev, size, alignment, mtype, memptr_gpu, memptr_cpu) MF_DOCA_DEVICE_ALLOC_TRACK(doca_gpu_mem_alloc, size, gpu_dev, size, alignment, mtype, memptr_gpu, memptr_cpu)
// rte_gpu_mem_alloc
#define MF_RTE_GPU_MEM_ALLOC(gpu_dpdk_id, buffer_size, pageSizeAlign) MF_RTE_DEVICE_ALLOC_TRACK(rte_gpu_mem_alloc, buffer_size, gpu_dpdk_id, buffer_size, pageSizeAlign)

/* Host pinned memory allocation functions */

// cudaHostAlloc
#define MF_CUDA_HOST_ALLOC(ptr, size, flags) MF_CUDA_HOST_ALLOC_TRACK(cudaHostAlloc, size, ptr, size, flags)
// cudaMallocHost
#define MF_CUDA_MALLOC_HOST(ptr, size) MF_CUDA_HOST_ALLOC_TRACK(cudaMallocHost, size, ptr, size)
// cuMemAllocHost
#define MF_CU_MEM_ALLOC_HOST(ptr, size) MF_CU_HOST_ALLOC_TRACK(cuMemAllocHost, size, ptr, size)

/* GPU memory deallocation functions */

// cudaFree
#define MF_CUDA_FREE(ptr) MF_CUDA_DEVICE_FREE_TRACK(cudaFree, ptr)
// cudaFreeArray
#define MF_CUDA_FREE_ARRAY(ptr) MF_CUDA_DEVICE_FREE_TRACK(cudaFreeArray, ptr)
// cudaFreeMipmappedArray
#define MF_CUDA_FREE_MIPMAPPED_ARRAY(ptr) MF_CUDA_DEVICE_FREE_TRACK(cudaFreeMipmappedArray, ptr)
// cudaFreeAsync
#define MF_CUDA_FREE_ASYNC(ptr, stream) MF_CUDA_DEVICE_FREE_TRACK(cudaFreeAsync, ptr, stream)
// cuMemFree
#define MF_CU_MEM_FREE(ptr) MF_CU_DEVICE_FREE_TRACK(cuMemFree, ptr)
// cuMemFreeAsync
#define MF_CU_MEM_FREE_ASYNC(ptr, stream) MF_CU_DEVICE_FREE_TRACK(cuMemFreeAsync, ptr, stream)

// cudaFreeHost
#define MF_CUDA_FREE_HOST(ptr) MF_CUDA_HOST_FREE_TRACK(cudaFreeHost, ptr)
// cuMemFreeHost
#define MF_CU_MEM_FREE_HOST(ptr) MF_CU_HOST_FREE_TRACK(cuMemFreeHost, ptr)

// doca_gpu_mem_free
#define MF_DOCA_GPU_MEM_FREE(gpu_dev, memptr_gpu) MF_DOCA_DEVICE_FREE_TRACK(doca_gpu_mem_free, gpu_dev, memptr_gpu)
// rte_gpu_mem_free
#define MF_RTE_GPU_MEM_FREE(gpu_dpdk_id, memptr_gpu) MF_RTE_DEVICE_FREE_TRACK(rte_gpu_mem_free, gpu_dpdk_id, memptr_gpu)

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* MEMFOOT_GLOBAL_H */
