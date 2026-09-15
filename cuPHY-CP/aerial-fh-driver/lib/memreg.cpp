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

#include "memreg.hpp"

#include "dpdk.hpp"
#include "nic.hpp"
#include "utils.hpp"
#include <fstream>

#define TAG "FH.MEMREG"

namespace aerial_fh
{
inline bool is_gpu_memory(const void* ptr) {
    unsigned int memType = 0;
    CUresult res = cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(ptr));
    if (res != CUDA_SUCCESS) {
        return false;
    }
    return (memType == CU_MEMORYTYPE_DEVICE);
}

MemReg::MemReg(Fronthaul* fhi, MemRegInfo const* info) :
    fhi_{fhi},
    info_{*info}
{
    info_.len = RTE_ALIGN(info_.len, info_.page_sz);

    NVLOGI_FMT(TAG, "Registering memory region of size {} bytes @{} with {}-byte backing page size", info_.len, info_.addr, info_.page_sz);

    auto ret = rte_extmem_register(info_.addr, info_.len, nullptr, 0, info_.page_sz);
    if(ret)
    {
        if(is_gpu_memory(info_.addr))
        {
            if(is_nvidia_peermem_loaded())
            {
                NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "nvidia-peermem is not loaded, please run 'sudo modprobe nvidia-peermem' on the host to load it, or use other BFW chaining mode");
            }
            THROW_FH(ret, StringBuilder() << "Failed to register device memory addr " << info_.addr << ": " << rte_strerror(-ret));
        }
        else
        {
            THROW_FH(ret, StringBuilder() << "Failed to register memory addr " << info_.addr << ": " << rte_strerror(-ret));
        }
    }

    for(auto& nic : fhi_->nics())
    {
        auto             port_id  = nic->get_port_id();
        auto             dev_name = nic->get_name();
        rte_eth_dev_info dev_info;

        ret = rte_eth_dev_info_get(port_id, &dev_info);
        if(ret)
        {
            THROW_FH(ret, StringBuilder() << "Failed to get device info for NIC " << dev_name);
        }

        ret = rte_dev_dma_map(dev_info.device, info_.addr, RTE_BAD_IOVA, info_.len);
        if(ret)
        {
            THROW_FH(ret, StringBuilder() << "Failed to DMA map addr " << info_.addr << " for NIC " << dev_name << ": " << rte_strerror(-ret));
        }
    }
}

MemReg::~MemReg()
{
    NVLOGI_FMT(TAG, "De-registering memory region of size {} bytes @{}", info_.len, info_.addr);

    for(auto& nic : fhi_->nics())
    {
        auto             port_id  = nic->get_port_id();
        auto             dev_name = nic->get_name();
        rte_eth_dev_info dev_info;

        auto ret = rte_eth_dev_info_get(nic->get_port_id(), &dev_info);
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to get device info for NIC ", dev_name.c_str());
            return;
        }

        ret = rte_dev_dma_unmap(dev_info.device, info_.addr, RTE_BAD_IOVA, info_.len);
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to DMA unmap addr {} from NIC {}: {}", info_.addr, dev_name.c_str(), rte_strerror(-ret));
        }
    }

    auto ret = rte_extmem_unregister(info_.addr, info_.len);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to deregister memory addr {}: {}",info_.addr,rte_strerror(-ret));
    }
}

bool MemReg::is_nvidia_peermem_loaded() const noexcept
{
    std::ifstream modules_file("/proc/modules");
    if (!modules_file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(modules_file, line)) {
        if (line.find("nvidia_peermem") != std::string::npos) {
            return true;
        }
    }

    return false;
}

Fronthaul* MemReg::get_fronthaul() const
{
    return fhi_;
}

uint32_t MemReg::get_lkey(Nic* nic, void* addr)
{
    auto             port_id  = nic->get_port_id();
    auto             dev_name = nic->get_name();
    rte_eth_dev_info dev_info;
    uint32_t lkey = 0;
    int ret = 0;

    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if(ret)
    {
        THROW_FH(ret, StringBuilder() << "Failed to get device info for NIC " << dev_name);
    }

    // Should we deprecate this function now that we are removing this?
    // ret = rte_pmd_mlx5_get_mr_lkey(dev_info.device, addr, &lkey);
    if(ret)
    {
        THROW_FH(ret, StringBuilder() << "Failed to get lkey for address " << addr);
    }

    return lkey;
}

} // namespace aerial_fh
