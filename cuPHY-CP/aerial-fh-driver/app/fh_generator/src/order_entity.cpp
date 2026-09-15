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

#include "order_entity.hpp"
#include <mutex>

namespace fh_gen
{

OrderEntity::OrderEntity(GpuDevice* _gDev, uint64_t _id) :
    gDev(_gDev), id(_id)
{
    static std::once_flag kernels_resolved;
    std::call_once(kernels_resolved, []() {
        if (!resolve_fhgen_cuda_kernels())
            THROW("Failed to resolve fh_generator CUDA kernel handles");
    });

    CUDA_DRIVER_CHECK(cuMemAllocHost(reinterpret_cast<void**>(&order_kernel_config_params), sizeof(orderKernelConfigParams_t)));

    int index = 0;
    for(auto& order_kernel_exit_cond:order_kernel_exit_cond_gdr){
        order_kernel_exit_cond.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        ((uint32_t*)(order_kernel_exit_cond->addrh()))[0] = ORDER_KERNEL_RUNNING;

        order_kernel_config_params->order_kernel_exit_cond_d[index] = (uint32_t*)order_kernel_exit_cond->addrd();
        index++;
    }
    //Ordered PRBs
    for(auto& oprbs:ordered_prbs) {
        oprbs.reset(gDev->newGDRbuf(sizeof(uint32_t)));
    }

    for(auto& early_rx_pkts:early_rx_packets){
        early_rx_pkts.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        ACCESS_ONCE(*((uint32_t*)early_rx_pkts->addrh())) = 0;
    }
    for(auto& on_time_rx_pkts:on_time_rx_packets){
        on_time_rx_pkts.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        ACCESS_ONCE(*((uint32_t*)on_time_rx_pkts->addrh())) = 0;
    }
    for(auto& late_rx_pkts:late_rx_packets){
        late_rx_pkts.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        ACCESS_ONCE(*((uint32_t*)late_rx_pkts->addrh())) = 0;
    }

    //End of order kernel for CPU thread
    order_end.reset(new host_buf(1 * sizeof(uint32_t), gDev));
    ACCESS_ONCE(*((uint32_t*)order_end->addr())) = 0;

    CUDA_DRIVER_CHECK(cuEventCreate(&start_order, CU_EVENT_DEFAULT));
    CUDA_DRIVER_CHECK(cuEventCreate(&end_order, CU_EVENT_DEFAULT));
    active.store(false);

    for(auto& rx_pkts_ts:rx_packets_ts){
        rx_pkts_ts.reset(gDev->newGDRbuf(ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_ALL_SYMBOLS*sizeof(uint64_t)));
    }
    for(auto& rx_pkts_count:rx_packets_count){
        rx_pkts_count.reset(gDev->newGDRbuf(ORAN_ALL_SYMBOLS*sizeof(uint32_t)));
    }
    for(auto& rx_pkts_ts_earliest:rx_packets_ts_earliest){
        rx_pkts_ts_earliest.reset(gDev->newGDRbuf(ORAN_ALL_SYMBOLS*sizeof(uint64_t)));
    }
    for(auto& rx_pkts_ts_latest:rx_packets_ts_latest){
        rx_pkts_ts_latest.reset(gDev->newGDRbuf(ORAN_ALL_SYMBOLS*sizeof(uint64_t)));
    }
}

OrderEntity::~OrderEntity()
{
    CUDA_DRIVER_CHECK_NON_FATAL(cuMemFreeHost(order_kernel_config_params));
}

uint64_t OrderEntity::getId() const
{
    return id;
}

int OrderEntity::runOrder(cudaStream_t stream)
{
    for(int cell_idx = 0; cell_idx < order_kernel_config_params->num_cells; ++cell_idx)
    {
        order_kernel_config_params->early_rx_packets[cell_idx] = (uint32_t*)early_rx_packets[cell_idx]->addrd();
        order_kernel_config_params->on_time_rx_packets[cell_idx] = (uint32_t*)on_time_rx_packets[cell_idx]->addrd();
        order_kernel_config_params->late_rx_packets[cell_idx] = (uint32_t*)late_rx_packets[cell_idx]->addrd();
        order_kernel_config_params->rx_packets_count[cell_idx] = (uint32_t*)rx_packets_count[cell_idx]->addrd();
        order_kernel_config_params->rx_packets_ts[cell_idx] = (uint64_t*)rx_packets_ts[cell_idx]->addrd();
        order_kernel_config_params->rx_packets_ts_earliest[cell_idx] = (uint64_t*)rx_packets_ts_earliest[cell_idx]->addrd();
        order_kernel_config_params->rx_packets_ts_latest[cell_idx] = (uint64_t*)rx_packets_ts_latest[cell_idx]->addrd();
    }
    kernel_receive_slot(stream, order_kernel_config_params);
    launch_kernel_write(stream, reinterpret_cast<uint32_t*>(order_end->addr()), 1);
    return 1;
}

uint32_t OrderEntity::getEarlyRxPackets(int cell_idx) {
    return *((uint32_t*)early_rx_packets[cell_idx]->addrh());
}
uint32_t OrderEntity::getOnTimeRxPackets(int cell_idx) {
    return *((uint32_t*)on_time_rx_packets[cell_idx]->addrh());
}
uint32_t OrderEntity::getLateRxPackets(int cell_idx) {
    return *((uint32_t*)late_rx_packets[cell_idx]->addrh());
}

int OrderEntity::checkOrderCPU()
{
    if(ACCESS_ONCE(*((uint32_t*)order_end->addr())) == 1)
        return 1;
    return 0;
}

uint32_t OrderEntity::getOrderExitCondition(int cell_idx){
    return *((uint32_t*)order_kernel_exit_cond_gdr[cell_idx]->addrh());
}

void OrderEntity::setOrderExitCondition(int cell_idx, uint32_t order_state){
    ((uint32_t*)(order_kernel_exit_cond_gdr[cell_idx]->addrh()))[0] = order_state;
}

uint64_t  OrderEntity::getRxPacketTsEarliest(int cell_idx,int sym_idx){
    return *((uint64_t*)rx_packets_ts_earliest[cell_idx]->addrh()+sym_idx);
}
uint64_t  OrderEntity::getRxPacketTsLatest(int cell_idx,int sym_idx){
    return *((uint64_t*)rx_packets_ts_latest[cell_idx]->addrh()+sym_idx);
}

bool OrderEntity::isActive()
{
    return active.load();
}

int OrderEntity::reserve()
{
    int ret = 0;
    if(active == true)
    {
        ret = -1;
    }
    else
    {
        active.store(true);
    }
    return ret;
}

void OrderEntity::cleanup()
{
    for(int cell_idx = 0; cell_idx < order_kernel_config_params->num_cells; cell_idx++)
    {
        ACCESS_ONCE(*((uint32_t*)early_rx_packets[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)on_time_rx_packets[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)late_rx_packets[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)order_end->addr())) = 0;
        ACCESS_ONCE(*((uint32_t*)order_kernel_exit_cond_gdr[cell_idx]->addrh())) = ORDER_KERNEL_RUNNING;
    }
}

void OrderEntity::release()
{
    active.store(false);
}

}