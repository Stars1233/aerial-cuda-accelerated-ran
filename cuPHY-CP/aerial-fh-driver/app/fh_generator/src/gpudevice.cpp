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

#include "gpudevice.hpp"

namespace fh_gen
{
GpuDevice::GpuDevice(
    uint32_t         _id,
    bool             _init_gdr) :
    id(_id),
    init_gdr(_init_gdr)
{
    CUDA_CHECK(cudaGetDeviceCount(&tot_devs));
    if(id > tot_devs)
        THROW("Device not found in the system");

    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, id)); // can also consider getting attributes individually via cudaDeviceGetAttribute
    CUDA_CHECK(cudaDeviceGetAttribute(&device_attr_clock_rate, cudaDevAttrClockRate, id)); // get attribute directly; marked as deprecated in cudaDeviceProp
    CUDA_CHECK(cudaDeviceGetAttribute(&device_is_direct_rdma_supported, cudaDevAttrGPUDirectRDMASupported, id)); // get attribute directly; unavailable in cudaDeviceProp
    
    setDevice();

    /*
    * Create a handle to the gdrcopy library
    * GDRCopy Required to flush GPUDirect RDMA writes NIC -> GPU
    */
    //This should not stay here because we may have multiple GPU devices
    //Maybe constructor can take as input a GDRCopy descriptor

    gdrc_h = nullptr;
    if(init_gdr == true && device_is_direct_rdma_supported == 1)
    {
        gdrc_h = gdr_open();
        if(gdrc_h == nullptr)
            THROW("GDRcopy open failed");
    }
    print_info();
}

GpuDevice::~GpuDevice()
{
    if(init_gdr == true)
    {
        if(gdrc_h != nullptr)
            gdr_close(gdrc_h);
    }
};

void GpuDevice::setDevice()
{
    CUDA_CHECK(cudaSetDevice(id));
}

gdr_t* GpuDevice::getGDRhandler()
{
    return &gdrc_h;
}

struct gpinned_buffer* GpuDevice::newGDRbuf(size_t size)
{
    return new gpinned_buffer{&gdrc_h, size, device_is_direct_rdma_supported == 1};
}

void GpuDevice::print_info()
{
    NVLOGI_FMT(TAG, "Using GPU {} {}:{}:{} {} kHz isRDMASupported:{}",deviceProp.name,deviceProp.pciBusID, deviceProp.pciDeviceID,deviceProp.pciDomainID, device_attr_clock_rate, device_is_direct_rdma_supported);
    // Hz=int64_t(device_attr_clock_rate) * 1000;
}


}
