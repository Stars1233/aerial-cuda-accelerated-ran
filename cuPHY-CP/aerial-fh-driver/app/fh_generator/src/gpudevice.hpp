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

#ifndef FH_GPUDEVICE_H
#define FH_GPUDEVICE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <vector>
#include <array>
#include <stdio.h>
#include <memory>
#include <cstring>
#include <gdrapi.h>
#include "utils.hpp"

namespace fh_gen
{

#ifdef DEVICE_TEGRA
#define GPU_PAGE_SHIFT 12
#else
#define GPU_PAGE_SHIFT 16
#endif
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)

#ifdef DEVICE_TEGRA
#define GPU_MIN_PIN_SIZE GPU_PAGE_SIZE
#else
#define GPU_MIN_PIN_SIZE 4
#endif
#define GPU_MAX_STREAMS 16

#define CUDA_CHECK(stmt)                   \
    do                                               \
    {                                                \
        cudaError_t result = (stmt);                 \
        if(cudaSuccess != result)                    \
        {                                            \
            NVLOGF_FMT(TAG, AERIAL_CUDA_API_EVENT, "[{}:{}] cuda failed with {} ", \
                   __FILE__,                         \
                   __LINE__,                         \
                   cudaGetErrorString(result));      \
        }                                            \
    } while(0)

#define CU_CHECK(stmt)                   \
    do                                             \
    {                                              \
        CUresult result = (stmt);                  \
        if(CUDA_SUCCESS != result)                 \
        {                                          \
            NVLOGF_FMT(TAG, AERIAL_CUDA_API_EVENT, "[{}:{}] cu failed with {} ", \
                   __FILE__,                       \
                   __LINE__,                       \
                   +result);                        \
        }                                          \
    } while(0)

// NB. Buffer management inspired by cuPHY
struct hpinned_alloc
{
    static void* allocate(size_t nbytes)
    {
        void* addr;
        // CUDA_CHECK(cudaMallocHost(&addr, nbytes));
        CUDA_CHECK(cudaHostAlloc(&addr, nbytes, cudaHostAllocDefault | cudaHostAllocPortable));
        return addr;
    }

    static void deallocate(void* addr)
    {
        // NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "hpinned_alloc CUDA_CHECK(cudaFreeHost"));
        CUDA_CHECK(cudaFreeHost(addr));
    }

    static void clear(void* addr, size_t nbytes)
    {
        memset(addr, 0, nbytes);
    }
};

typedef struct gpinned_buffer
{
public:
    gpinned_buffer(gdr_t* _g, size_t _size_input, bool _is_rdma_supported) :
        g(_g),
        size_input(_size_input),
        is_rdma_supported(_is_rdma_supported)
    {
        CUdeviceptr        dev_addr = 0;
        void*              host_ptr = NULL;
        const unsigned int FLAG     = 1;
        size_t             pin_size, alloc_size, rounded_size;

        if(g == nullptr || size_input == 0)
            THROW("gpinned_buffer bad input arguments");
        
        // If RDMA is supported - proceed to using GDRCopy library for further allocation
        // If not, use CUDA pinned host memory allocation. 
        if (!is_rdma_supported)
        {

            host_ptr = hpinned_alloc::allocate(size_input); 
            // In a system with full unified memory, the host and the device pointer _may_ match.
            CU_CHECK(cuMemHostGetDevicePointer(&dev_addr, host_ptr, 0));

            addr_d    = (uintptr_t)dev_addr;
            addr_h    = (uintptr_t)host_ptr;
            return; 
        }

        if(size_input < GPU_MIN_PIN_SIZE)
            size_input = GPU_MIN_PIN_SIZE;

        // GDRDRV and the GPU driver require GPU page size-aligned address and size
        // arguments to gdr_pin_buffer, so we need to be paranoid here and allocate
        // an extra page so we can safely pass the rounded size
        rounded_size = (size_input + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
        pin_size     = rounded_size;
        alloc_size   = rounded_size + GPU_PAGE_SIZE;

/*----------------------------------------------------------------*
            * Allocate device memory.                                        */
#ifdef DEVICE_TEGRA
        void* cudaHost_A;

        CUresult e = cuMemHostAlloc(&cudaHost_A, alloc_size, 0);
        if(CUDA_SUCCESS != e)
            THROW("cuMemHostAlloc");

        e = cuMemHostGetDevicePointer(&dev_addr, cudaHost_A, 0);
        if(CUDA_SUCCESS != e)
            THROW("cuMemHostGetDevicePointer");
#else
        // CU_CHECK(cuMemAlloc(&dev_addr, alloc_size));
        CU_CHECK(cuMemAlloc(&dev_addr, alloc_size));
#endif

        addr_free = (uintptr_t)dev_addr;
        // Offset into a page-aligned address if necessary
        if(dev_addr % GPU_PAGE_SIZE)
        {
            dev_addr += (GPU_PAGE_SIZE - (dev_addr % GPU_PAGE_SIZE));
        }
        /*----------------------------------------------------------------*
            * Set attributes for the allocated device memory.                */
        CU_CHECK(cuPointerSetAttribute(&FLAG, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_addr));
        // if(CUDA_SUCCESS != cuPointerSetAttribute(&FLAG, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dev_addr))
        // {
        //     cuMemFree(dev_addr);
        //     // gdr_close(g);
            // THROW("cuPointerSetAttribute");
        // }
        /*----------------------------------------------------------------*
            * Pin the device buffer                                          */
        if(0 != gdr_pin_buffer(*g, dev_addr, pin_size, 0, 0, &mh))
        {
            CU_CHECK(cuMemFree(dev_addr));
            THROW("gdr_pin_buffer");
        }
        /*----------------------------------------------------------------*
            * Map the buffer to user space                                   */
        if(0 != gdr_map(*g, mh, &host_ptr, pin_size))
        {
            gdr_unpin_buffer(*g, mh);
            CU_CHECK(cuMemFree(dev_addr));
            THROW("gdr_map");
        }
        /*----------------------------------------------------------------*
            * Retrieve info about the mapping                                */
        if(0 != gdr_get_info(*g, mh, &info))
        {
            gdr_unmap(*g, mh, host_ptr, pin_size);
            gdr_unpin_buffer(*g, mh);
            CU_CHECK(cuMemFree(dev_addr));
            THROW("gdr_get_info");
        }

        addr_d    = (uintptr_t)dev_addr;
        addr_h    = (uintptr_t)host_ptr;
        size_free = pin_size;
        size_alloc = alloc_size;
    };

    ~gpinned_buffer()
    {
        if (is_rdma_supported) {
        gdr_unmap(*g, mh, (void*)addr_h, size_free);
        gdr_unpin_buffer(*g, mh);
        CU_CHECK(cuMemFree((CUdeviceptr)addr_free));
        } else {
            hpinned_alloc::deallocate((void*)addr_h); 
        }
    };

    void* addrh()
    {
        return (void*)addr_h;
    }

    void* addrd()
    {
        return (void*)addr_d;
    }

    size_t size()
    {
        return size_input;
    }

    // Used to measure memory footprint
    size_t     size_free;
    size_t     size_alloc;

protected:
    gdr_t*     g;
    gdr_mh_t   mh;
    gdr_info_t info;
    uintptr_t  addr_d;    //device memory
    uintptr_t  addr_h;    //host memory
    uintptr_t  addr_free; //must be used to free memory
    size_t     size_input;
    bool       is_rdma_supported; 
} gpinned_buffer;

class GpuDevice {
public:
    GpuDevice(uint32_t _id, bool init_gdr);
    ~GpuDevice();
    struct gpinned_buffer* newGDRbuf(size_t size);
    gdr_t*                 getGDRhandler();
    void                   setDevice();
    void                   print_info();
private:
    uint32_t              id;
    int                   tot_devs;
    struct cudaDeviceProp deviceProp;
    int                   device_attr_clock_rate;
    int                   device_is_direct_rdma_supported; 
    gdr_t                 gdrc_h;
    bool                  init_gdr;
};

// NB. Buffer management inspired by cuPHY
struct device_alloc
{
    static void* allocate(size_t nbytes)
    {
        void* addr;
        CUDA_CHECK(cudaMalloc(&addr, nbytes));
        return addr;
    }
    static void deallocate(void* addr)
    {
        // NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "device_alloc CUDA_CHECK(cudaFree"));
        CUDA_CHECK(cudaFree(addr));
    }

    static void clear(void* addr, size_t nbytes)
    {
        CUDA_CHECK(cudaMemset(addr, 0, nbytes));
    }
};

template <typename T, class TAllocator>
class IOBuf {
public:
    IOBuf() :
        _addr(nullptr),
        _size(0),
        gDev(nullptr) {}
    IOBuf(size_t numElements, GpuDevice* _gDev) :
        // addr(static_cast<T*>(TAllocator::allocate(numElements * sizeof(T)))),
        _size(numElements),
        gDev(_gDev)
    {
        _addr = static_cast<T*>(TAllocator::allocate(_size * sizeof(T)));
        // std::cout << "Allocated a new buffer of " << _size << "bytes" << std::endl;
    };

    ~IOBuf()
    {
        if(_addr)
        {
            // NVLOGI_FMT(TAG, "Free the IOBuf of {} bytes", _size);
            TAllocator::deallocate(_addr);
        }
    }

    T*     addr() { return _addr; }
    size_t size() const { return _size; }
    void   clear() { TAllocator::clear(_addr, _size); }

private:
    T*         _addr;
    size_t     _size;
    GpuDevice* gDev;
};

typedef IOBuf<uint8_t, device_alloc>  dev_buf;
typedef IOBuf<uint8_t, hpinned_alloc> host_buf;

}

#endif
