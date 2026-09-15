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
#include <cstddef>

namespace fh_gen
{
GpuDevice::GpuDevice(
    uint32_t         _id,
    bool             _init_gdr) :
    id(_id),
    init_gdr(_init_gdr)
{
    CUDA_DRIVER_CHECK(cuDeviceGetCount(&tot_devs));
    if(id >= static_cast<uint32_t>(tot_devs))
        THROW("Device not found in the system");

    CUDA_DRIVER_CHECK(cuDeviceGet(&cuDevice_, id));
    CUDA_DRIVER_CHECK(cuDeviceGetName(deviceName_, sizeof(deviceName_), cuDevice_));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&devicePciBusId_, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cuDevice_));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&devicePciDeviceId_, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cuDevice_));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&devicePciDomainId_, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, cuDevice_));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&device_attr_clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cuDevice_));
    CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&device_is_direct_rdma_supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, cuDevice_));

    CUDA_DRIVER_CHECK(cuDevicePrimaryCtxRetain(&cuCtx_, cuDevice_));
    setDevice();

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
    CUDA_DRIVER_CHECK_NON_FATAL(cuDevicePrimaryCtxRelease(cuDevice_));
};

void GpuDevice::setDevice()
{
    CUDA_DRIVER_CHECK(cuCtxSetCurrent(cuCtx_));
}

gdr_t* GpuDevice::getGDRhandler()
{
    return &gdrc_h;
}

[[nodiscard]] struct gpinned_buffer* GpuDevice::newGDRbuf(const std::size_t size)
{
    const bool use_gdr_path =
        (device_is_direct_rdma_supported != 0) && (gdrc_h != nullptr);
    return new gpinned_buffer{&gdrc_h, size, use_gdr_path};
}

void GpuDevice::print_info()
{
    NVLOGI_FMT(TAG, "Using GPU {} {}:{}:{} {} kHz isRDMASupported:{}", deviceName_, devicePciBusId_, devicePciDeviceId_, devicePciDomainId_, device_attr_clock_rate, device_is_direct_rdma_supported);
}


}
