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

#if !defined(CUPHY_KERNEL_UTIL_CUH_INCLUDED_)
#define CUPHY_KERNEL_UTIL_CUH_INCLUDED_

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "cuphy_internal.h"

// clang-format off
#if CUPHY_DEBUG

#define KERNEL_PRINT_BLOCK_ONCE(...) do { if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z)) { printf(__VA_ARGS__); } } while(0)
#define KERNEL_PRINT_GRID_ONCE(...) do \
    {                                  \
        if((0 == threadIdx.x) &&       \
           (0 == threadIdx.y) &&       \
           (0 == threadIdx.z) &&       \
           (0 == blockIdx.x)  &&       \
           (0 == blockIdx.y)  &&       \
           (0 == blockIdx.z))          \
        {                              \
            printf(__VA_ARGS__);       \
        }                              \
    } while(0)
#define KERNEL_PRINT_GRID_ONCE_AT_X(threadX, blockX, ...) do { if((threadX == threadIdx.x) && (blockX == blockIdx.x)) { printf(__VA_ARGS__); } } while(0)
#define KERNEL_PRINT_IF(cond, ...) do { if(cond) { printf(__VA_ARGS__); } } while(0)
#define KERNEL_PRINT(...) do { printf(__VA_ARGS__); } while(0)

#else

#define KERNEL_PRINT_BLOCK_ONCE(...)
#define KERNEL_PRINT_GRID_ONCE(...)
#define KERNEL_PRINT_GRID_ONCE_AT_X(threadX, blockX, ...)
#define KERNEL_PRINT_IF(cond, ...)
#define KERNEL_PRINT(...)

#endif

// clang-format on

////////////////////////////////////////////////////////////////////////
// See the simpleTemplates CUDA example
// clang-format off
template <typename T> struct shared_mem_t                { __device__ T*             addr() { extern __device__ void error(); error();  return nullptr; } };
template <>           struct shared_mem_t<float>         { __device__ float*         addr() { extern __shared__ float s_float[];        return s_float; } };
template <>           struct shared_mem_t<struct __half> { __device__ struct __half* addr() { extern __shared__ struct __half s_half[]; return s_half;  } };
template <>           struct shared_mem_t<int>           { __device__ int*           addr() { extern __shared__ int s_int[];            return s_int;   } };
template <>           struct shared_mem_t<char>          { __device__ char*          addr() { extern __shared__ char s_char[];          return s_char;  } };
// clang-format on

////////////////////////////////////////////////////////////////////////
// kernel_mat
// Simple wrapper to allow (row, col) addressing from an address within
// a kernel
template <typename T>
class kernel_mat //
{
public:
    __device__ kernel_mat(T* d, int lda) :
        data_(d),
        lda_(lda) {}
    __device__ T&    operator()(int i, int j) { return data_[(j * lda_) + i]; }
    __device__ const T& operator()(int i, int j) const { return data_[(j * lda_) + i]; }

private:
    T*  data_;
    int lda_;
};

////////////////////////////////////////////////////////////////////////
// grid_copy_x()
template <typename T>
__device__ void grid_copy_x(T* dst, const T* src, size_t szElements)
{
    for(size_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < szElements; i += (blockDim.x * gridDim.x))
    {
        dst[i] = src[i];
    }
}

////////////////////////////////////////////////////////////////////////
// block_copy_N_sync()
template <typename T, int N_PER_THREAD>
__device__ void block_copy_N_sync(T* dst, const T* src)
{
    #pragma unroll
    for(int i = static_cast<int>(threadIdx.x); i < N_PER_THREAD; i += static_cast<int>(blockDim.x))
    {
        dst[i] = src[i];
    }
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////
// block_copy_N_sync_check()
// Copies data from src to dst for 1D geometries. If the source address
// is greater than or equal to srcEnd, zero is written to the destination.
template <typename T, int N_PER_THREAD>
__device__ void block_copy_N_sync_check(T* dst, const T* src, const T* srcEnd)
{
    #pragma unroll
    for(int i = 0; i < N_PER_THREAD; ++i)
    {
        int idx = threadIdx.x + (i * blockDim.x);
        dst[idx] = ((src + idx) >= srcEnd) ? 0 : src[idx];
    }
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////
// block_copy_sync
template <typename T>
__device__ void block_copy_sync(T* dst, const T* src, int szElements)
{
    for(int i = static_cast<int>(threadIdx.x); i < szElements; i += static_cast<int>(blockDim.x))
    {
        dst[i] = src[i];
    }
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////
// block_copy_sync_2D
template <typename T>
__device__ void block_copy_sync_2D(T* dst, const T* src, int szElements)
{
    for(int i = static_cast<int>(threadIdx.x + (threadIdx.y * blockDim.x));
        i < szElements;
        i += static_cast<int>(blockDim.x * blockDim.y))
    {
        dst[i] = src[i];
    }
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////
// block_zero_sync
template <typename T>
__device__ void block_zero_sync(T* dst, int szElements)
{
    for(int i = static_cast<int>(threadIdx.x);
        i < szElements;
        i += static_cast<int>(blockDim.x))
    {
        dst[i] = 0;
    }
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////
// block_zero_sync_2D
template <typename T>
__device__ void block_zero_sync_2D(T* dst, int szElements)
{
    for(int i = static_cast<int>(threadIdx.x + (threadIdx.y * blockDim.x));
        i < szElements;
        i += static_cast<int>(blockDim.x * blockDim.y))
    {
        dst[i] = static_cast<T>(0);
    }
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////
// block_copy_pair_sync
template <typename T>
__device__ void block_copy_pair_sync(T* dst0, T* dst1, const T* devSource, int sz)
{
    for(int i = static_cast<int>(threadIdx.x); i < sz; i += static_cast<int>(blockDim.x))
    {
        dst0[i] = devSource[i];
        dst1[i] = devSource[i];
    }
    __syncthreads();
}

// clang-format off
template <typename T> struct printer;
template <>           struct printer<int32_t>       { static CUDA_INLINE void print(const int32_t& i)       { KERNEL_PRINT("%4i ", static_cast<int>(i)); } };
template <>           struct printer<int16_t>       { static CUDA_INLINE void print(const int16_t& i)       { KERNEL_PRINT("%4i ", static_cast<int>(i)); } };
template <>           struct printer<uint32_t>      { static CUDA_INLINE void print(const uint32_t& u)      { KERNEL_PRINT("%4u ", static_cast<unsigned int>(u)); } };
template <>           struct printer<uint16_t>      { static CUDA_INLINE void print(const uint16_t& u)      { KERNEL_PRINT("%4u ", static_cast<unsigned int>(u)); } };
template <>           struct printer<int8_t>        { static CUDA_INLINE void print(const int8_t& i)        { KERNEL_PRINT("%4i ", static_cast<int>(i)); } };
template <>           struct printer<float>         { static CUDA_INLINE void print(const float& f)         { KERNEL_PRINT("%.4f ", f);                  } };
template <>           struct printer<__half>        { static CUDA_INLINE void print(const __half& h)        { KERNEL_PRINT("%.4f ", __half2float(h));    } };
template <>           struct printer<__half2>       { static CUDA_INLINE void print(const __half2& h)       { KERNEL_PRINT("[%.4f, %.4f] ", __low2float(h), __high2float(h)); } };
template <>           struct printer<__nv_fp8_e5m2> { static CUDA_INLINE void print(const __nv_fp8_e5m2& q) { KERNEL_PRINT("%.4f ", static_cast<float>(q)); } };
template <>           struct printer<__nv_fp8_e4m3> { static CUDA_INLINE void print(const __nv_fp8_e4m3& q) { KERNEL_PRINT("%.4f ", static_cast<float>(q)); } };
// clang-format on

template <typename T>
__device__ void print_array_sync(const char* desc, const T* shmem, int N)
{
    __syncthreads();
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        KERNEL_PRINT("%s:\n", desc);
        for(int i = 0; i < N; ++i)
        {
            KERNEL_PRINT("%4i: ", i);
            printer<T>::print(shmem[i]);
            KERNEL_PRINT("\n");
        }
    }
    __syncthreads();
}

template <typename T>
__device__ void print_array_sync(const char* desc, const char* fmt, const T* shmem, int N)
{
    __syncthreads();
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        KERNEL_PRINT("%s:\n", desc);
        for(int i = 0; i < N; ++i)
        {
            KERNEL_PRINT("%4i: ", i);
            KERNEL_PRINT(fmt, shmem[i]);
            KERNEL_PRINT("\n");
        }
    }
    __syncthreads();
}

template <typename T, int M, int N>
__device__ void print_matrix(const char* desc, const T (&m)[M][N])
{
    __syncthreads();
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        KERNEL_PRINT("%s (%i x %i):\n", desc, M, N);
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                printer<T>::print(m[i][j]);
            }
            KERNEL_PRINT("\n");
        }
    }
}

template <typename T>
__device__ void print_kernel_mat(const char* desc, const kernel_mat<T>& a, int M, int N)
{
    __syncthreads();
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        KERNEL_PRINT("%s (%i x %i):\n", desc, M, N);
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                printer<T>::print(a(i, j));
            }
            KERNEL_PRINT("\n");
        }
    }
}

////////////////////////////////////////////////////////////////////////
// uint32_permute()
template <unsigned int PRMT_BYTES>
__device__ inline
uint32_t uint32_permute(uint32_t a, uint32_t b)
{
    uint32_t wOut;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(wOut) : "r"(a), "r"(b), "n"(PRMT_BYTES));
    return wOut;
}

////////////////////////////////////////////////////////////////////////
// tex_1D_ptx()(float result)
inline
__device__ void tex_1D_ptx(cuphy_i::tex_result_v4<float>& res, cudaTextureObject_t texObj, float u)
{
    asm volatile("tex.1d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5}];"
                 : "=f"(res.x), "=f"(res.y), "=f"(res.z), "=f"(res.w)
                 : "l"(texObj), "f"(u));
}
////////////////////////////////////////////////////////////////////////
// tex_1D_ptx() (__half result)
inline
__device__ void tex_1D_ptx(cuphy_i::tex_result_v4<__half>& res, cudaTextureObject_t texObj, float u)
{
    asm volatile("tex.1d.v2.f16x2.f32 {%0, %1}, [%2, {%3}];"
                 : "=r"(res.a.u32), "=r"(res.b.u32)
                 : "l"(texObj), "f"(u));
}

////////////////////////////////////////////////////////////////////////
// tex_1D_lod_ptx() (float result)
inline
__device__ void tex_1D_lod_ptx(cuphy_i::tex_result_v4<float>& res, cudaTextureObject_t texObj, float u, float lvl)
{
    asm volatile("tex.level.1d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5}], %6;"
                 : "=f"(res.x), "=f"(res.y), "=f"(res.z), "=f"(res.w)
                 : "l"(texObj), "f"(u), "f"(lvl));
}

////////////////////////////////////////////////////////////////////////
// tex_1D_lod_ptx() (__half result)
inline
__device__ void tex_1D_lod_ptx(cuphy_i::tex_result_v4<__half>& res, cudaTextureObject_t texObj, float u, float lvl)
{
    asm volatile("tex.level.1d.v2.f16x2.f32 {%0, %1}, [%2, {%3}], %4;"
                 : "=r"(res.a.u32), "=r"(res.b.u32)
                 : "l"(texObj), "f"(u), "f"(lvl));
}

////////////////////////////////////////////////////////////////////////
// Grid stride indexing
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
// Usage:
// __global__ void loop_kernel(float*   p,
//                             unsigned int xmax,
//                             unsigned int xstride,
//                             unsigned int ymax,
//                             unsigned int ystride,
//                             unsigned int zmax,
//                             unsigned int zstride)
//{
//
//    for(grid_stride_index<2> it2; it2 < zmax; it2.next())
//    {
//        for(grid_stride_index<1> it1; it1 < ymax; it1.next())
//        {
//            for(grid_stride_index<0> it0; it0 < xmax; it0.next())
//            {
//                unsigned int idx = (it0.value * xstride) +
//                                   (it1.value * ystride) +
//                                   (it2.value * zstride);
//                p[idx] = do_something();
//            } // it0
//        }     // it1
//    }         // it2
//}


// clang-format off
template <unsigned int TIndex> struct dim3_accessor;
template <> struct dim3_accessor<0> { static CUDA_INLINE unsigned int get(const dim3& d) { return d.x; } };
template <> struct dim3_accessor<1> { static CUDA_INLINE unsigned int get(const dim3& d) { return d.y; } };
template <> struct dim3_accessor<2> { static CUDA_INLINE unsigned int get(const dim3& d) { return d.z; } };
// clang-format on

template <unsigned int TIndex>
struct grid_stride_index
{
    typedef dim3_accessor<TIndex> dim_accessor;
    unsigned int                  value;
    CUDA_INLINE
    grid_stride_index() :
        value(dim_accessor::get(blockDim) * dim_accessor::get(blockIdx) + dim_accessor::get(threadIdx))
    {
    }
    CUDA_INLINE
    grid_stride_index(unsigned int val) :
        value(val) {}
    CUDA_INLINE void next()
    {
        value += (dim_accessor::get(blockDim) * dim_accessor::get(gridDim));
    }
    CUDA_INLINE
    bool operator<(unsigned int end) const { return (value < end); }
};

////////////////////////////////////////////////////////////////////////
// warp_inclusive_scan()
// Performs an inclusive scan using all threads in a warp. Assumes all
// threads are active.
template <typename T, unsigned int TLog2=5>
__device__
T warp_inclusive_scan(T value)
{
    const unsigned int LANEID = (threadIdx.x & 0x1F);
    #pragma unroll
    for(int i = 0; i < TLog2; ++i)
    {
        int shift = (1 << i);
        value = ((LANEID >= shift)  ? value : 0) + __shfl_up_sync(0xFFFFFFFF, value, shift);
    }
    return value;
}

////////////////////////////////////////////////////////////////////////
// warp_exclusive_scan()
// Performs an exclusive scan using all threads in a warp. Assumes all
// threads are active.
template <typename T, unsigned int TLog2=5>
__device__
T warp_exclusive_scan(T value)
{
    const unsigned int LANEID = (threadIdx.x & 0x1F);
    
    T v0 = __shfl_up_sync(0xFFFFFFFF, value, 1);
    if(0 == LANEID)
    {
        v0 = 0;
    }
    return warp_inclusive_scan<T, TLog2>(v0);
}

#endif // !defined(CUPHY_KERNEL_UTIL_CUH_INCLUDED_)
