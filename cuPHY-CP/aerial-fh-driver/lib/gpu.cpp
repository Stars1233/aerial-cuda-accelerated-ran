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

#include "gpu.hpp"

#include "utils.hpp"

#define TAG "FH.GPU"

namespace aerial_fh
{
Gpu::Gpu(Fronthaul* fhi, GpuId cuda_device_id) :
    fhi_{fhi},
    cuda_device_id_{cuda_device_id},
    pci_bus_id_{cuda_device_id_to_pci_bus_id(cuda_device_id)}
{
}

Gpu::~Gpu()
{

}

GpuId Gpu::get_cuda_device_id() const
{
    return cuda_device_id_;
}

std::string Gpu::get_pci_bus_id() const
{
    return pci_bus_id_;
}

std::string Gpu::cuda_device_id_to_pci_bus_id(GpuId cuda_device_id)
{
    CUdevice dev;
    CHECK_CU_THROW(cuDeviceGet(&dev, cuda_device_id));

    constexpr size_t                         kGpuPciBusIdBufferSize = 32;
    std::array<char, kGpuPciBusIdBufferSize> buffer;
    CHECK_CU_THROW(cuDeviceGetPCIBusId(buffer.data(), static_cast<int>(kGpuPciBusIdBufferSize), dev));

    auto pci_bus_id = std::string(buffer.data());
    std::transform(pci_bus_id.begin(), pci_bus_id.end(), pci_bus_id.begin(), [](unsigned char c) { return std::tolower(c); });
    return pci_bus_id;
}

int16_t Gpu::cuda_device_id_to_dpdk_gpu_id(GpuId cuda_device_id)
{
    auto         pci_bus_id  = cuda_device_id_to_pci_bus_id(cuda_device_id);
    int16_t      dpdk_gpu_id = 0;
    rte_gpu_info gpu_info{};

    RTE_GPU_FOREACH(dpdk_gpu_id)
    {
        auto ret = rte_gpu_info_get(dpdk_gpu_id, &gpu_info);
        if(ret)
        {
            THROW_FH(ret, StringBuilder() << "Failed to get info for DPDK GPU device " << dpdk_gpu_id << ": " << rte_strerror(-ret));
        }

        auto current_pci_bus_id = std::string(gpu_info.name);
        std::transform(current_pci_bus_id.begin(), current_pci_bus_id.end(), current_pci_bus_id.begin(), [](unsigned char c) { return std::tolower(c); });

        if(current_pci_bus_id == pci_bus_id)
        {
            return dpdk_gpu_id;
        }
    }

    return -1;
}

} // namespace aerial_fh
