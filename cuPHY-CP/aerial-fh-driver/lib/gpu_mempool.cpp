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

#include "gpu_mempool.hpp"

#include "gpu.hpp"
#include "nic.hpp"
#include "utils.hpp"

#define TAG "FH.GPU_MP"

namespace aerial_fh
{
GpuMempool::GpuMempool(Gpu* gpu, Nic* nic,bool host_pinned) :
    gpu_{gpu},
    nic_{nic},
    host_pinned_{host_pinned}
{
    auto cuda_device_id = gpu_->get_cuda_device_id();
    auto mbuf_num       = rte_align32pow2(kGpuMbufPoolSz) - 1;

    uint16_t         droom_sz    = RTE_ALIGN_MUL_CEIL(nic_->get_mtu() + RTE_PKTMBUF_HEADROOM + RTE_ETHER_HDR_LEN + RTE_ETHER_CRC_LEN, kMbufPoolDroomSzAlign);
    auto             buffer_size = RTE_ALIGN(mbuf_num * droom_sz, kNvGpuPageSize);
    auto             nic_name    = nic_->get_name();
    auto             port_id     = nic_->get_port_id();
    std::string      mp_name     = StringBuilder() << "gpu_" << cuda_device_id << "_mbuf_" << nic_name;
    rte_eth_dev_info eth_dev_info;
    void*            host_buf_ptr;

    NVLOGI_FMT(TAG, "Initializing mempool for CUDA device {} and NIC {}. It will have {} mbuf entries. Mbuf droom size: {}",
             cuda_device_id, nic_name.c_str(), mbuf_num, droom_sz);

    if(host_pinned)
    {
        ASSERT_CUDA_FH(cudaMallocHost((void**)&host_buf_ptr, buffer_size));
        gpu_mem_ = {host_buf_ptr, RTE_BAD_IOVA, buffer_size, droom_sz};
    }
    else
    {
        THROW_FH(ENOTSUP, StringBuilder() << "Device memory allocation is not supported due to DPDK GPU ID non-availability");
    }
    if(!gpu_mem_.buf_ptr)
    {
        THROW_FH(ENOMEM, StringBuilder() << "Failed to allocate device memory for CUDA device " << cuda_device_id);
    }

    auto ret = rte_extmem_register(gpu_mem_.buf_ptr, gpu_mem_.buf_len, nullptr, gpu_mem_.buf_iova, kNvGpuPageSize);
    if(ret)
    {
        THROW_FH(ret, StringBuilder() << "Failed to register CUDA device " << cuda_device_id << " memory @" << gpu_mem_.buf_ptr << ": " << rte_strerror(-ret));
    }

    ret = rte_eth_dev_info_get(port_id, &eth_dev_info);
    if(ret)
    {
        THROW_FH(ret, StringBuilder() << "Failed to get device info for NIC " << nic_name);
    }

    ret = rte_dev_dma_map(eth_dev_info.device, gpu_mem_.buf_ptr, gpu_mem_.buf_iova, gpu_mem_.buf_len);
    if(ret)
    {
        THROW_FH(ret, StringBuilder() << "Failed to DMA map CUDA device " << cuda_device_id << " memory @" << gpu_mem_.buf_ptr << " to NIC " << nic_name << ": " << rte_strerror(-ret));
    }

    auto gpu_mempool = rte_pktmbuf_pool_create_extbuf(mp_name.c_str(), mbuf_num, 0, 0, droom_sz, rte_eth_dev_socket_id(port_id), &gpu_mem_, 1);
    if(gpu_mempool == nullptr)
    {
        THROW_FH(rte_errno, StringBuilder() << "Could not create " << mp_name << " mempool: " << rte_strerror(rte_errno));
    }

    mempool_.reset(gpu_mempool);
}

GpuMempool::~GpuMempool()
{
    auto             cuda_device_id = gpu_->get_cuda_device_id();
    auto             nic_name       = nic_->get_name();
    rte_eth_dev_info eth_dev_info;

    auto ret = rte_eth_dev_info_get(nic_->get_port_id(), &eth_dev_info);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to get device info for NIC {}", nic_name.c_str());
        return;
    }

    ret = rte_dev_dma_unmap(eth_dev_info.device, gpu_mem_.buf_ptr, gpu_mem_.buf_iova, gpu_mem_.buf_len);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to DMA unmap CUDA device {} memory @{} from NIC {}: {}", cuda_device_id , gpu_mem_.buf_ptr, nic_name.c_str(), rte_strerror(-ret));
    }

    ret = rte_extmem_unregister(gpu_mem_.buf_ptr, gpu_mem_.buf_len);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to unregister CUDA device {} memory @{}: {}", cuda_device_id, gpu_mem_.buf_ptr, rte_strerror(-ret));
    }

    if(host_pinned_)
    {
        ASSERT_CUDA_FH(cudaFreeHost(gpu_mem_.buf_ptr));
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Device memory deallocation is not supported due to DPDK GPU ID non-availability");
    }
}

rte_mempool* GpuMempool::get_pool() const
{
    return mempool_.get();
}

} // namespace aerial_fh
