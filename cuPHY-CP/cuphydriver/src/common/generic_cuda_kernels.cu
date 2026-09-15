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
#include "compression_types.hpp"       // compression_params, mod_compression_params, CleanupDlBufInfo
#include "aerial-fh-driver/api.hpp"    // aerial_fh::UserDataCompressionMethod
#include "cuda_driver_utils/cuda_driver_utils.hpp"  // CUDA_DRIVER_CHECK
#include "gpu_blockFP.h" //Compression Decompression repo
#include "gpu_fixed.h"
#include "comp_kernel.cuh"
#include "generic_kernel_functions.hpp"
#include "cuda_driver_utils/cuda_kernel_utils.cuh"  // resolve_kernel_func (must be before extern "C" block)

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

void launch_kernel_print_hex(const CUfunction func, const cudaStream_t stream, uint8_t* addr, int offset, int num_bytes)
{
    if(!addr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "addr is NULL");
        return;
    }

    void* args[] = {&addr, &offset, &num_bytes};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
}

__global__ void kernel_write(uint32_t* addr, uint32_t value)
{
    ACCESS_ONCE(*addr) = value;
}

void launch_kernel_write(const CUfunction func, const cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    if(!addr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "addr is NULL");
        return;
    }
    MemtraceDisableScope md;

    void* args[] = {&addr, &value};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
}

__global__ void kernel_read(uint8_t* addr)
{
    printf("0) %x - 1) %x - 2) %x\n", addr[0], addr[1], addr[2]);
    printf("2048) %x - 2049) %x - 2050) %x\n", addr[0], addr[1], addr[2]);
}

void launch_kernel_read(const CUfunction func, const cudaStream_t stream, uint8_t* addr)
{
    if(!addr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "addr is NULL");
        return;
    }

    void* args[] = {&addr};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
}
__global__ void warmup_kernel()
{
    __threadfence();
}

void launch_kernel_warmup(const CUfunction func, const cudaStream_t stream)
{
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 512, 1, 1, 0, stream, nullptr, nullptr));
}

__global__ void kernel_wait_update(uint32_t* addr, uint32_t expected, uint32_t updated)
{
    while(ACCESS_ONCE(*addr) != expected)
        ;
    ACCESS_ONCE(*addr) = updated;

    __threadfence();
}

void launch_kernel_wait_update(const CUfunction func, const cudaStream_t stream, uint32_t* addr, uint32_t expected, uint32_t updated)
{
    void* args[] = {&addr, &expected, &updated};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
}

__global__ void kernel_wait_eq(uint32_t* addr, uint32_t value)
{
    while(ACCESS_ONCE(*addr) != value);
    __threadfence();
}

void launch_kernel_wait_eq(const CUfunction func, const cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    void* args[] = {&addr, &value};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
}

__global__ void kernel_wait_neq(uint32_t* addr, uint32_t value)
{
    while(ACCESS_ONCE(*addr) == value);
    __threadfence();
}

void launch_kernel_wait_neq(const CUfunction func, const cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    void* args[] = {&addr, &value};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
}

__global__ void kernel_wait_geq(uint32_t* addr, uint32_t value)
{
    while(ACCESS_ONCE(*addr) < value);
    __threadfence();
}

void launch_kernel_wait_geq(const CUfunction func, const cudaStream_t stream, uint32_t* addr, uint32_t value)
{
    void* args[] = {&addr, &value};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));
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

void launch_kernel_compare(const CUfunction func, const cudaStream_t stream, uint8_t* addr1, uint8_t* addr2, int size)
{
    void* args[] = {&addr1, &addr2, &size};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, 512, 1, 1, 0, stream, args, nullptr));
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

void launch_kernel_check_crc(const CUfunction func, const cudaStream_t stream, const uint32_t* i_buf, size_t i_elems, uint32_t* out)
{
    void* args[] = {&i_buf, &i_elems, &out};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, 1, 1, 1, CRC_THREADS, 1, 1, sizeof(uint32_t) * 1, stream, args, nullptr));
}

void launch_kernel_compression(
    const CompressionKernelFunctions& comp,
    cudaStream_t stream,
    const std::array<compression_params, NUM_USER_DATA_COMPRESSION_METHODS>& cparams_array)
{
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
                dim3 grid(*max_antennas, params.num_cells, SLOT_NUM_SYMS);

                const auto first_bfp = params.bit_width[0];
                const bool const_bfp = std::all_of(
                    params.bit_width,
                    params.bit_width + params.num_cells,
                    [first_bfp](decltype(first_bfp) bw) { return bw == first_bfp; }
                );

                // If all cells have the same compression bit_width, we can specialize the kernel
                CUfunction chosen{};
                if(const_bfp && first_bfp == 9)
                    chosen = comp.compress_9;
                else if(const_bfp && first_bfp == 14)
                    chosen = comp.compress_14;
                else if(const_bfp && first_bfp == 16)
                    chosen = comp.compress_16;
                else // Otherwise use the non-specialized kernel
                    chosen = comp.compress_0;

                void* args[] = {const_cast<compression_params*>(&params)};
                CUDA_DRIVER_CHECK(cuLaunchKernel(chosen, grid.x, grid.y, grid.z, COMPRESSION_THREADS, 1, 1, 0, stream, args, nullptr));
                break;
            }
            case aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION:
            {
                const int nwarps = 2;
                dim3 grid((MAX_SECTIONS_PER_UPLANE_SYMBOL + nwarps - 1) / nwarps, *max_antennas, params.num_cells);
                dim3 block(32, nwarps, SLOT_NUM_SYMS);
                void* args[] = {const_cast<compression_params*>(&params)};
                CUDA_DRIVER_CHECK(cuLaunchKernel(comp.mod_compression_qam, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, stream, args, nullptr));
                break;
            }
            default:
                // Other compression methods not yet implemented
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Compression method {} not implemented", comp_method);
                break;
        }
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

void launch_kernel_copy(const CUfunction func, const cudaStream_t stream, uint8_t* input_buffer, uint8_t* output_buffer, int bytes)
{
    void* args[] = {&input_buffer, &output_buffer, &bytes};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, COPY_BLOCKS, 1, 1, COPY_THREADS, 1, 1, 0, stream, args, nullptr));
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

void launch_memset_kernel(const CUfunction func, void* d_buffers_addr, int num_cells, size_t max_buffer_size, const cudaStream_t strm) {

    int num_threads = 1024;
    // max_buffer_size is in bytes
    int blocks = (max_buffer_size + sizeof(uint4)*num_threads - 1) / (sizeof(uint4)*num_threads);
    void* args[] = {&d_buffers_addr};
    CUDA_DRIVER_CHECK(cuLaunchKernel(func, blocks, num_cells, 1, num_threads, 1, 1, 0, strm, args, nullptr));
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

bool resolve_warmup_kernel_handle(CUfunction* out)
{
    return resolve_kernel_func<TAG>(out, reinterpret_cast<const void*>(warmup_kernel), "warmup_kernel");
}

bool resolve_kernel_write_handle(CUfunction* out)
{
    return resolve_kernel_func<TAG>(out, reinterpret_cast<const void*>(kernel_write), "kernel_write");
}

bool resolve_kernel_wait_eq_handle(CUfunction* out)
{
    return resolve_kernel_func<TAG>(out, reinterpret_cast<const void*>(kernel_wait_eq), "kernel_wait_eq");
}

bool resolve_memset_kernel_handle(CUfunction* out)
{
    return resolve_kernel_func<TAG>(out, reinterpret_cast<const void*>(memset_kernel), "memset_kernel");
}

bool resolve_compression_kernel_handles(CompressionKernelFunctions& out)
{
    bool ok = true;
    ok &= resolve_kernel_func<TAG>(&out.compress_0,              reinterpret_cast<const void*>(kernel_compress<0>),              "kernel_compress<0>");
    ok &= resolve_kernel_func<TAG>(&out.compress_9,              reinterpret_cast<const void*>(kernel_compress<9>),              "kernel_compress<9>");
    ok &= resolve_kernel_func<TAG>(&out.compress_14,             reinterpret_cast<const void*>(kernel_compress<14>),             "kernel_compress<14>");
    ok &= resolve_kernel_func<TAG>(&out.compress_16,             reinterpret_cast<const void*>(kernel_compress<16>),             "kernel_compress<16>");
    ok &= resolve_kernel_func<TAG>(&out.mod_compression_qam,     reinterpret_cast<const void*>(kernel_mod_compression<QAM_Comp>), "kernel_mod_compression<QAM_Comp>");
    return ok;
}

#ifdef __cplusplus
}
#endif

bool init_generic_cuda_kernel_functions(GenericCudaKernelFunctions& funcs)
{
    bool ok = true;
    ok &= resolve_kernel_func<TAG>(&funcs.print_complex_fp16, reinterpret_cast<const void*>(print_complex_fp16), "print_complex_fp16");
    ok &= resolve_kernel_func<TAG>(&funcs.print_hexbytes,     reinterpret_cast<const void*>(print_hexbytes),     "print_hexbytes");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_write,       reinterpret_cast<const void*>(kernel_write),       "kernel_write");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_read,        reinterpret_cast<const void*>(kernel_read),        "kernel_read");
    ok &= resolve_kernel_func<TAG>(&funcs.warmup_kernel,      reinterpret_cast<const void*>(warmup_kernel),      "warmup_kernel");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_wait_update, reinterpret_cast<const void*>(kernel_wait_update), "kernel_wait_update");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_wait_eq,     reinterpret_cast<const void*>(kernel_wait_eq),     "kernel_wait_eq");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_wait_neq,    reinterpret_cast<const void*>(kernel_wait_neq),    "kernel_wait_neq");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_wait_geq,    reinterpret_cast<const void*>(kernel_wait_geq),    "kernel_wait_geq");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_compare,     reinterpret_cast<const void*>(kernel_compare),     "kernel_compare");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_check_crc,   reinterpret_cast<const void*>(kernel_check_crc),   "kernel_check_crc");
    ok &= resolve_kernel_func<TAG>(&funcs.kernel_copy,        reinterpret_cast<const void*>(kernel_copy),        "kernel_copy");
    ok &= resolve_kernel_func<TAG>(&funcs.memset_kernel,      reinterpret_cast<const void*>(memset_kernel),      "memset_kernel");
    ok &= resolve_kernel_func<TAG>(&funcs.compression.compress_0,  reinterpret_cast<const void*>(kernel_compress<0>),  "kernel_compress<0>");
    ok &= resolve_kernel_func<TAG>(&funcs.compression.compress_9,  reinterpret_cast<const void*>(kernel_compress<9>),  "kernel_compress<9>");
    ok &= resolve_kernel_func<TAG>(&funcs.compression.compress_14, reinterpret_cast<const void*>(kernel_compress<14>), "kernel_compress<14>");
    ok &= resolve_kernel_func<TAG>(&funcs.compression.compress_16, reinterpret_cast<const void*>(kernel_compress<16>), "kernel_compress<16>");
    ok &= resolve_kernel_func<TAG>(&funcs.compression.mod_compression_qam, reinterpret_cast<const void*>(kernel_mod_compression<QAM_Comp>), "kernel_mod_compression<QAM_Comp>");
    return ok;
}

