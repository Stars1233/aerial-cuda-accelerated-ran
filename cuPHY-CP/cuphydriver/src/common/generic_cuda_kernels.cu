/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 6) // "DRV.GEN_CUDA"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "constant.hpp"
#include "cuphydriver_api.hpp"
#include "phychannel.hpp"
#include "gpu_blockFP.h" //Compression Decompression repo
#include "gpu_fixed.h"
#include "comp_kernel.cuh"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void print_complex_fp16(__half2* addr, int offset, int num_samples)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        for(int k = 0; k < num_samples; k++)
        {
            printf("[%06d] %f + %fj\n", offset + k, static_cast<float>(addr[k + offset].x), static_cast<float>(addr[k + offset].y));
        }
    }
}
__global__ void print_hexbytes(uint8_t* addr, int offset, int num_bytes)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
    {
        for(int k = 0; k < num_bytes; k++)
        {
            printf("%02X ", addr[k + offset]);
            // printf("[%08d] %02X\n", k + offset, addr[k + offset]);
        }
        printf("\n");
    }
}

void launch_kernel_print_hex(cudaStream_t stream, uint8_t* addr, int offset, int num_bytes)
{
    cudaError_t result = cudaSuccess;

    if(!addr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "addr is NULL");
        return;
    }

    print_hexbytes<<<1, 1, 0, stream>>>(addr, offset, num_bytes);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void kernel_write(uint32_t* addr, uint32_t value)
{
    ACCESS_ONCE(*addr) = value;
}

void launch_kernel_write(cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    cudaError_t result = cudaSuccess;

    if(!addr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "addr is NULL");
        return;
    }
    MemtraceDisableScope md;

    kernel_write<<<1, 1, 0, stream>>>(addr, value);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void kernel_read(uint8_t* addr)
{
    printf("0) %x - 1) %x - 2) %x\n", addr[0], addr[1], addr[2]);
    printf("2048) %x - 2049) %x - 2050) %x\n", addr[0], addr[1], addr[2]);
}

void launch_kernel_read(cudaStream_t stream, uint8_t* addr)
{
    cudaError_t result = cudaSuccess;

    if(!addr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "addr is NULL");
        return;
    }

    kernel_read<<<1, 1, 0, stream>>>(addr);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}
__global__ void warmup_kernel()
{
    __threadfence();
}

void launch_kernel_warmup(cudaStream_t stream)
{
    cudaError_t result = cudaSuccess;

    warmup_kernel<<<1, 512, 0, stream>>>();

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void kernel_wait_update(uint32_t* addr, uint32_t expected, uint32_t updated)
{
    while(ACCESS_ONCE(*addr) != expected)
        ;
    ACCESS_ONCE(*addr) = updated;

    __threadfence();
}

void launch_kernel_wait_update(cudaStream_t stream, uint32_t* addr, uint32_t expected, uint32_t updated)
{
    cudaError_t result = cudaSuccess;

    kernel_wait_update<<<1, 1, 0, stream>>>(addr, expected, updated);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void kernel_wait_eq(uint32_t* addr, uint32_t value)
{
    while(ACCESS_ONCE(*addr) != value);
    __threadfence();
}

void launch_kernel_wait_eq(cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    cudaError_t result = cudaSuccess;

    kernel_wait_eq<<<1, 1, 0, stream>>>(addr, value);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void kernel_wait_neq(uint32_t* addr, uint32_t value)
{
    while(ACCESS_ONCE(*addr) == value);
    __threadfence();
}

void launch_kernel_wait_neq(cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    cudaError_t result = cudaSuccess;

    kernel_wait_neq<<<1, 1, 0, stream>>>(addr, value);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void kernel_wait_geq(uint32_t* addr, uint32_t value)
{
    while(ACCESS_ONCE(*addr) < value);
    __threadfence();
}

void launch_kernel_wait_geq(cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    cudaError_t result = cudaSuccess;

    kernel_wait_geq<<<1, 1, 0, stream>>>(addr, value);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void kernel_compare(uint8_t* addr1, uint8_t* addr2, int size)
{
    int i = threadIdx.x;
    // int count = 0;
    for(i = threadIdx.x; i < size; i += blockDim.x)
    {
        // printf("Compare %d: addr1 = %x, addr2=%x\n", i, addr1[i], addr2[i]);

        if(addr1[i] != addr2[i])
            printf("Difference in %d: addr1 = %x, addr2=%x\n", i, addr1[i], addr2[i]);
    }

    // printf("ORDER KERNEL DIFFERENCE IS %d\n", count);

    __syncthreads();
    __threadfence();
}

void launch_kernel_compare(cudaStream_t stream, uint8_t* addr1, uint8_t* addr2, int size)
{
    cudaError_t result = cudaSuccess;

    kernel_compare<<<1, 512, 0, stream>>>(addr1, addr2, size);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CRC error count on the GPU
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CRC_THREADS 512

__global__ void kernel_check_crc(const uint32_t* i_buf, size_t i_elems, uint32_t* out)
{
    __shared__ uint32_t out_sh[1];
    if(threadIdx.x == 0)
        out_sh[0] = 0;
    __syncthreads();
    for(int i = threadIdx.x; i < (int)i_elems; i += CRC_THREADS)
    {
        if(i_buf[i] != 0)
            atomicAdd(out_sh, 1);
    }
    __syncthreads();
    __threadfence_block();
    if(threadIdx.x == 0)
    {
        *out = out_sh[0];
    }
}

void launch_kernel_check_crc(cudaStream_t stream, const uint32_t* i_buf, size_t i_elems, uint32_t* out)
{
    cudaError_t result = cudaSuccess;
    result             = cudaGetLastError();

    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));

    kernel_check_crc<<<1, CRC_THREADS, sizeof(uint32_t) * 1, stream>>>(i_buf, i_elems, out);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

void launch_kernel_compression(
    cudaStream_t stream,
    const std::array<compression_params, NUM_USER_DATA_COMPRESSION_METHODS>& cparams_array)
{
    cudaError_t result = cudaSuccess;
    MemtraceDisableScope md;

    // Process each compression method that has cells
    for(std::size_t comp_method = 0; comp_method < NUM_USER_DATA_COMPRESSION_METHODS; ++comp_method)
    {
        const compression_params& params = cparams_array[comp_method];
        
        // Skip if no cells for this compression method
        if(params.num_cells == 0)
            continue;

        // PRBs can vary by threads, so launch enough to cover the worst case
        auto max_antennas = std::max_element(params.num_antennas, params.num_antennas + params.num_cells);

        switch (static_cast<aerial_fh::UserDataCompressionMethod>(comp_method))
        {
            case aerial_fh::UserDataCompressionMethod::NO_COMPRESSION:
            case aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT:
            {
                dim3 blocks(*max_antennas, params.num_cells, SLOT_NUM_SYMS);

                // If all cells have the same compression bit_width, we can specialize the kernel
                const auto first_bfp = params.bit_width[0];
                const bool const_bfp = std::all_of(
                    params.bit_width,
                    params.bit_width + params.num_cells,
                    [first_bfp](decltype(first_bfp) bw) { return bw == first_bfp; }
                );
            
                if(const_bfp && first_bfp == 9)
                {
                    kernel_compress<9><<<blocks, COMPRESSION_THREADS, 0, stream>>>(params);
                }
                else if(const_bfp && first_bfp == 14)
                {
                    kernel_compress<14><<<blocks, COMPRESSION_THREADS, 0, stream>>>(params);
                }
                else if(const_bfp && first_bfp == 16)
                {
                    kernel_compress<16><<<blocks, COMPRESSION_THREADS, 0, stream>>>(params);
                }
                else // Otherwise use the non-specialized kernel
                {
                    kernel_compress<0><<<blocks, COMPRESSION_THREADS, 0, stream>>>(params);
                }
                break;
            }
            case aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION:
            {
                const int nwarps = 2;
                dim3 threads(32, nwarps, SLOT_NUM_SYMS);
                dim3 blocks_mod((MAX_SECTIONS_PER_UPLANE_SYMBOL + nwarps - 1)/nwarps , *max_antennas, params.num_cells);
                kernel_mod_compression<QAM_Comp><<<blocks_mod, threads, 0, stream>>>(params);
                break;
            }
            default:
                // Other compression methods not yet implemented
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Compression method {} not implemented", comp_method);
                break;
        }

        result = cudaGetLastError();
        if(cudaSuccess != result)
            NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} for compression method {}", 
                __FILE__, __LINE__, cudaGetErrorString(result), comp_method);
    }
}

#define COPY_BLOCKS 4
#define COPY_THREADS 512

__global__ void kernel_copy(uint8_t* input_buffer, uint8_t* output_buffer, int bytes)
{
    int tid = (threadIdx.x+blockIdx.x*blockDim.x);

    while(tid < bytes)
    {
        output_buffer[tid] = input_buffer[tid];
        tid += (blockDim.x * gridDim.x);
    }
}

void launch_kernel_copy(cudaStream_t stream, uint8_t* input_buffer, uint8_t* output_buffer, int bytes)
{
    cudaError_t result = cudaSuccess;

    kernel_copy<<<COPY_BLOCKS, COPY_THREADS, 0, stream>>>(input_buffer, output_buffer, bytes);

    result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

__global__ void memset_kernel(void* d_buffers) {

    const uint4 val = {0, 0, 0, 0};
    int cell = blockIdx.y;

    CleanupDlBufInfo* d_buffer_addr_buf = (CleanupDlBufInfo*)d_buffers + cell;
    uint4* d_buffer_addr = d_buffer_addr_buf->d_buf_addr;
    size_t d_buffer_size = d_buffer_addr_buf->buf_size;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < (d_buffer_size >> 4)) { // 4 is to divide by sizeof(uint4) as d_buffer_size is in bytes
        d_buffer_addr[tid] = val;
    }

    //Handle leftover bytes
    int leftover_bytes = d_buffer_size & 0xF; // modulo sizeof(uint4) = 16
    if (tid < leftover_bytes) {
        uint8_t* d_buffer_byte_addr = (uint8_t*)d_buffer_addr + (d_buffer_size - leftover_bytes);
        d_buffer_byte_addr[tid] = 0;
    }
}

void launch_memset_kernel(void* d_buffers_addr, int num_cells, size_t max_buffer_size, cudaStream_t strm) {

    int num_threads = 1024;
    // max_buffer_size is in bytes
    int blocks = (max_buffer_size + sizeof(uint4)*num_threads - 1) / (sizeof(uint4)*num_threads);
    memset_kernel<<<dim3(blocks, num_cells), num_threads, 0, strm>>>(d_buffers_addr);
    cudaError_t result = cudaGetLastError();
    if(cudaSuccess != result)
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cuda failed with {} ", __FILE__, __LINE__, cudaGetErrorString(result));
}

void force_loading_generic_cuda_kernels()
{
    std::array<void*, 13> generic_cuda_functions = {
     (void*)print_complex_fp16,
     (void*)print_hexbytes,
     (void*)kernel_write,
     (void*)kernel_read,
     (void*)warmup_kernel,
     (void*)kernel_wait_update,
     (void*)kernel_wait_eq,
     (void*)kernel_wait_neq,
     (void*)kernel_wait_geq,
     (void*)kernel_compare,
     (void*)kernel_check_crc,
     (void*)kernel_copy,
     (void*)memset_kernel};

     for(auto& generic_cuda_function:generic_cuda_functions)
     {
         cudaFuncAttributes attr;
         cudaError_t e = cudaFuncGetAttributes(&attr, static_cast<const void*>(generic_cuda_function));
         if(cudaSuccess != e)
         {
             NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cudaFuncGetAttributes call failed with {} ", __FILE__, __LINE__, cudaGetErrorString(e));
         }
     }

}

#ifdef __cplusplus
}
#endif
