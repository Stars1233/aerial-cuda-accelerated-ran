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
#include <cstddef>
#include "context.hpp"
#include "exceptions.hpp"
#include "nvlog.hpp"

GpuDevice::GpuDevice(
    phydriver_handle _pdh,
    uint32_t         _id,
    bool             _init_gdr) :
    pdh(_pdh),
    id(_id),
    primary_ctx(nullptr),
    init_gdr(_init_gdr)
{
    CUDA_DRIVER_CHECK(cuDeviceGetCount(&tot_devs));
    if(id >= tot_devs)
        PHYDRIVER_THROW_EXCEPTIONS(-1, "Device not found in the system");

    CUdevice device;
    CUDA_DRIVER_CHECK(cuDeviceGet(&device, static_cast<int>(id)));
    CUDA_DRIVER_CHECK(cuDeviceGetName(deviceProp.name, sizeof(deviceProp.name), device));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&deviceProp.pciBusID, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&deviceProp.pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&deviceProp.pciDomainID, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&device_attr_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&device_is_direct_rdma_supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, device));

    /* Retain primary context once; setDevice() will only call cuCtxSetCurrent. */
    CUDA_DRIVER_CHECK(cuDevicePrimaryCtxRetain(&primary_ctx, device));
    CUDA_DRIVER_CHECK(cuCtxSetCurrent(primary_ctx));

    /*
    * Create a handle to the gdrcopy library
    * GDRCopy Required to flush GPUDirect RDMA writes NIC -> GPU
    */
    //This should not stay here because we may have multiple GPU devices
    //Maybe constructor can take as input a GDRCopy descriptor

    gdrc_h = nullptr;
    if(init_gdr == true && device_is_direct_rdma_supported != 0)
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
    CUdevice device;
    CUresult r = cuDeviceGet(&device, static_cast<int>(id));
    if (r == CUDA_SUCCESS) {
        CUDA_DRIVER_CHECK_NON_FATAL(cuDevicePrimaryCtxRelease(device));
    }
    else {
        NVLOGW_FMT(TAG,"[{}:{}] cuDeviceGet failed in ~GpuDevice (result {}), skipping PrimaryCtxRelease",
                   __FILE__, __LINE__, static_cast<unsigned>(r));
    }
    
}

gdr_t* GpuDevice::getGDRhandler()
{
    return &gdrc_h;
}

[[nodiscard]] struct gpinned_buffer* GpuDevice::newGDRbuf(const std::size_t size)
{
    // Use GDR device-memory path only when GDRCopy is open; CUDA may report RDMA support
    // while init_gdr is false, in which case we must use the pinned-host fallback.
    const bool use_gdr_path =
        (device_is_direct_rdma_supported != 0) && (gdrc_h != nullptr);
    return new gpinned_buffer{&gdrc_h, size, use_gdr_path};
}

int GpuDevice::runWarmup(CUfunction warmup_func, int n, CUstream s)
{
    for(int i = 0; i < n; i++)
    {
        launch_kernel_warmup(warmup_func, s);
    }

    return 0;
}

phydriver_handle GpuDevice::getPhyDriverHandler(void) const
{
    return pdh;
}

void GpuDevice::setDevice()
{
    CUDA_DRIVER_CHECK(cuCtxSetCurrent(primary_ctx));
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

void GpuDevice::synchronizeStream(CUstream stream)
{
    CUDA_DRIVER_CHECK(cuStreamSynchronize(stream));
}