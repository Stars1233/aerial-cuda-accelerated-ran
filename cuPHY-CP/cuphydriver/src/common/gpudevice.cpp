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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 7) // "DRV.GPUDEV"

#include "gpudevice.hpp"
#include "context.hpp"
#include "exceptions.hpp"
#include "nvlog.hpp"

GpuDevice::GpuDevice(
    phydriver_handle _pdh,
    uint32_t         _id,
    bool             _init_gdr) :
    pdh(_pdh),
    id(_id),
    init_gdr(_init_gdr)
{
    CUDA_CHECK_PHYDRIVER(cudaGetDeviceCount(&tot_devs));
    if(id > tot_devs)
        PHYDRIVER_THROW_EXCEPTIONS(-1, "Device not found in the system");

    CUDA_CHECK_PHYDRIVER(cudaGetDeviceProperties(&deviceProp, id)); // can also consider getting attributes individually via cudaDeviceGetAttribute
    CUDA_CHECK_PHYDRIVER(cudaDeviceGetAttribute(&device_attr_clock_rate, cudaDevAttrClockRate, id)); // get attribute directly; marked as deprecated in cudaDeviceProp
    CUDA_CHECK_PHYDRIVER(cudaDeviceGetAttribute(&device_is_direct_rdma_supported, cudaDevAttrGPUDirectRDMASupported, id)); // get attribute directly; unavailable in cudaDeviceProp

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
            PHYDRIVER_THROW_EXCEPTIONS(-1, "GDRcopy open failed");
    }

    mf.init(_pdh, std::string("GpuDevice"), sizeof(GpuDevice));
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

gdr_t* GpuDevice::getGDRhandler()
{
    return &gdrc_h;
}

struct gpinned_buffer* GpuDevice::newGDRbuf(size_t size)
{
    return new gpinned_buffer{&gdrc_h, size, device_is_direct_rdma_supported == 1};
}

int GpuDevice::runWarmup(int n, cudaStream_t s)
{
    for(int i = 0; i < n; i++)
    {
        launch_kernel_warmup(s);
    }

    return 0;
}

phydriver_handle GpuDevice::getPhyDriverHandler(void) const
{
    return pdh;
}

void GpuDevice::setDevice()
{
    CUDA_CHECK_PHYDRIVER(cudaSetDevice(id));
}

uint32_t GpuDevice::getId()
{
    return id;
}

void GpuDevice::print_info()
{
    NVLOGI_FMT(TAG, "Using GPU {} {}:{}:{} {} kHz isRDMASupported:{}",deviceProp.name,deviceProp.pciBusID, deviceProp.pciDeviceID,deviceProp.pciDomainID, device_attr_clock_rate, device_is_direct_rdma_supported);
    // Hz=int64_t(device_attr_clock_rate) * 1000;
}

void GpuDevice::synchronizeStream(cudaStream_t stream)
{
    CUDA_CHECK_PHYDRIVER(cudaStreamSynchronize(stream));
}
