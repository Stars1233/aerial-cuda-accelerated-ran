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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 20) // "DRV.ORDER_ENTITY"

#include "order_entity.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "cuda_events.hpp"
#include "cuphyoam.hpp"
#include <typeinfo>
#include <slot_command/slot_command.hpp>

OrderEntity::OrderEntity(phydriver_handle _pdh, GpuDevice* _gDev) :
    pdh(_pdh),
    gDev(_gDev)
{
    active      = false;
    std::fill(std::begin(cell_id),std::end(cell_id),0);
    id          = Time::nowNs().count();

    mf.init(_pdh, std::string("OrderEntity"), sizeof(OrderEntity));

    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->setUlCtx();

    //End of order kernel for CPU thread
    start_cuphy_cpu_h.reset(new host_buf(1 * sizeof(uint32_t), gDev));
    ACCESS_ONCE(*((uint32_t*)start_cuphy_cpu_h->addr())) = 0;
    mf.addCpuPinnedSize(sizeof(uint32_t));

    start_cuphy_srs_cpu_h.reset(new host_buf(1 * sizeof(uint32_t), gDev));
    ACCESS_ONCE(*((uint32_t*)start_cuphy_srs_cpu_h->addr())) = 0;
    mf.addCpuPinnedSize(sizeof(uint32_t));    

    //End of order kernel for GPU pusch
    for(auto& start_cuphy:start_cuphy_gdr){
        start_cuphy.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        ((uint32_t*)(start_cuphy->addrh()))[0] = 0;
        mf.addGpuPinnedSize(start_cuphy->size_alloc);
    }

    for(auto& order_kernel_exit_cond:order_kernel_exit_cond_gdr){
        order_kernel_exit_cond.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        ((uint32_t*)(order_kernel_exit_cond->addrh()))[0] = ORDER_KERNEL_RUNNING;
        mf.addGpuPinnedSize(order_kernel_exit_cond->size_alloc);
    }

    for(auto& start_cuphy:start_cuphy_srs_gdr){
        start_cuphy.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        ((uint32_t*)(start_cuphy->addrh()))[0] = 0;
        mf.addGpuPinnedSize(start_cuphy->size_alloc);
    }

    for(auto& order_kernel_exit_cond:order_kernel_srs_exit_cond_gdr){
        order_kernel_exit_cond.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        ((uint32_t*)(order_kernel_exit_cond->addrh()))[0] = ORDER_KERNEL_RUNNING;
        mf.addGpuPinnedSize(order_kernel_exit_cond->size_alloc);
    }    

    //Ordered PRBs
    for(auto& oprbs_prach:ordered_prbs_prach){
        oprbs_prach.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(oprbs_prach->size_alloc);
    }
    for(auto& oprbs_pusch:ordered_prbs_pusch){
        oprbs_pusch.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(oprbs_pusch->size_alloc);
    }

    for(auto& oprbs_srs:ordered_prbs_srs){
        oprbs_srs.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(oprbs_srs->size_alloc);
    }

    /*Sub-slot Processing specific*/
    pusch_prb_symbol_map_gdr.reset(gDev->newGDRbuf(sizeof(uint32_t)*UL_MAX_CELLS_PER_SLOT*ORAN_PUSCH_SYMBOLS_X_SLOT));
    mf.addGpuPinnedSize(pusch_prb_symbol_map_gdr->size_alloc);
    sym_ord_done_mask_arr.reset(gDev->newGDRbuf(ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(uint32_t)));
    mf.addGpuPinnedSize(sym_ord_done_mask_arr->size_alloc);
    num_order_cells_sym_mask_arr.reset(gDev->newGDRbuf(ORAN_PUSCH_SYMBOLS_X_SLOT * sizeof(uint32_t)));
    mf.addGpuPinnedSize(num_order_cells_sym_mask_arr->size_alloc);

    //Multi-Block barrier
    // order_barrier_flag.reset(new dev_buf(1 * sizeof(int), gDev));
    // order_barrier_flag->clear();
    order_barrier_flag.reset(gDev->newGDRbuf(sizeof(int)));
    mf.addGpuPinnedSize(order_barrier_flag->size_alloc);

    for(auto& done_sh:done_shared){
        done_sh.reset(new dev_buf(1 * sizeof(uint8_t), gDev));
        done_sh->clear();
        mf.addGpuRegularSize(done_sh->size_alloc);
    }
    for(auto& done_sh:done_shared_srs){
        done_sh.reset(new dev_buf(1 * sizeof(uint8_t), gDev));
        done_sh->clear();
        mf.addGpuRegularSize(done_sh->size_alloc);
    }    

    ready_shared.reset(new dev_buf(1 * sizeof(int), gDev));
    ready_shared->clear();
    mf.addGpuRegularSize(ready_shared->size_alloc);

    rx_queue_index.reset(new dev_buf(1 * sizeof(int), gDev));
    rx_queue_index->clear();
    mf.addGpuRegularSize(rx_queue_index->size_alloc);

    //Early/On-time/Late RX packets
    for(auto& early_rx_pkts:early_rx_packets){
        early_rx_pkts.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(early_rx_pkts->size_alloc);
        ACCESS_ONCE(*((uint32_t*)early_rx_pkts->addrh())) = 0;
    }
    for(auto& on_time_rx_pkts:on_time_rx_packets){
        on_time_rx_pkts.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(on_time_rx_pkts->size_alloc);
        ACCESS_ONCE(*((uint32_t*)on_time_rx_pkts->addrh())) = 0;
    }
    for(auto& late_rx_pkts:late_rx_packets){
        late_rx_pkts.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(late_rx_pkts->size_alloc);
        ACCESS_ONCE(*((uint32_t*)late_rx_pkts->addrh())) = 0;
    }
    for(auto& rx_pkts_ts:rx_packets_ts){
        rx_pkts_ts.reset(gDev->newGDRbuf(ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint64_t)));
        mf.addGpuPinnedSize(rx_pkts_ts->size_alloc);
    }    
    for(auto& rx_pkts_count:rx_packets_count){
        rx_pkts_count.reset(gDev->newGDRbuf(ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));
        mf.addGpuPinnedSize(rx_pkts_count->size_alloc);
    }
    for(auto& rx_bts_count:rx_bytes_count){
        rx_bts_count.reset(gDev->newGDRbuf(ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));
        mf.addGpuPinnedSize(rx_bts_count->size_alloc);
    }
    for(auto& rx_pkts_dropped_count:rx_packets_dropped_count){
        rx_pkts_dropped_count.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(rx_pkts_dropped_count->size_alloc);
        ACCESS_ONCE(*((uint32_t*)rx_pkts_dropped_count->addrh())) = 0;
    }

    for(auto& rx_pkts_ts_earliest:rx_packets_ts_earliest){
        rx_pkts_ts_earliest.reset(gDev->newGDRbuf(ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint64_t)));
        mf.addGpuPinnedSize(rx_pkts_ts_earliest->size_alloc);
    }
    for(auto& rx_pkts_ts_latest:rx_packets_ts_latest){
        rx_pkts_ts_latest.reset(gDev->newGDRbuf(ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint64_t)));
        mf.addGpuPinnedSize(rx_pkts_ts_latest->size_alloc);
    }

    //Early/On-time/Late RX packets for SRS
    for(auto& early_rx_pkts_srs:early_rx_packets_srs){
        early_rx_pkts_srs.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(early_rx_pkts_srs->size_alloc);
        ACCESS_ONCE(*((uint32_t*)early_rx_pkts_srs->addrh())) = 0;
    }
    for(auto& on_time_rx_pkts_srs:on_time_rx_packets_srs){
        on_time_rx_pkts_srs.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(on_time_rx_pkts_srs->size_alloc);
        ACCESS_ONCE(*((uint32_t*)on_time_rx_pkts_srs->addrh())) = 0;
    }
    for(auto& late_rx_pkts_srs:late_rx_packets_srs){
        late_rx_pkts_srs.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(late_rx_pkts_srs->size_alloc);
        ACCESS_ONCE(*((uint32_t*)late_rx_pkts_srs->addrh())) = 0;
    }

    for(auto& rx_pkts_count_srs:rx_packets_count_srs){
        rx_pkts_count_srs.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(rx_pkts_count_srs->size_alloc);
        ACCESS_ONCE(*((uint32_t*)rx_pkts_count_srs->addrh())) = 0;
    }
    for(auto& rx_bts_count_srs:rx_bytes_count_srs){
        rx_bts_count_srs.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(rx_bts_count_srs->size_alloc);
        ACCESS_ONCE(*((uint32_t*)rx_bts_count_srs->addrh())) = 0;
    }
    for(auto& rx_pkts_dropped_count_srs:rx_packets_dropped_count_srs){
        rx_pkts_dropped_count_srs.reset(gDev->newGDRbuf(sizeof(uint32_t)));
        mf.addGpuPinnedSize(rx_pkts_dropped_count_srs->size_alloc);
        ACCESS_ONCE(*((uint32_t*)rx_pkts_dropped_count_srs->addrh())) = 0;
    }

    for(auto& rx_pkts_ts:rx_packets_ts_srs){
        rx_pkts_ts.reset(gDev->newGDRbuf(ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*ORAN_SRS_SYMBOLS_X_SLOT*sizeof(uint64_t)));
        mf.addGpuPinnedSize(rx_pkts_ts->size_alloc);
    }    
    for(auto& rx_pkts_count_srs:rx_packets_count_per_sym_srs){
        rx_pkts_count_srs.reset(gDev->newGDRbuf(ORAN_SRS_SYMBOLS_X_SLOT*sizeof(uint32_t)));
        mf.addGpuPinnedSize(rx_pkts_count_srs->size_alloc);
    }
    for(auto& rx_pkts_ts_earliest_srs:rx_packets_ts_earliest_srs){
        rx_pkts_ts_earliest_srs.reset(gDev->newGDRbuf(ORAN_SRS_SYMBOLS_X_SLOT*sizeof(uint64_t)));
        mf.addGpuPinnedSize(rx_pkts_ts_earliest_srs->size_alloc);
    }
    for(auto& rx_pkts_ts_latest_srs:rx_packets_ts_latest_srs){
        rx_pkts_ts_latest_srs.reset(gDev->newGDRbuf(ORAN_SRS_SYMBOLS_X_SLOT*sizeof(uint64_t)));
        mf.addGpuPinnedSize(rx_pkts_ts_latest_srs->size_alloc);
    }      

    /*
     * Associate eAxC id to index for the order kernel
     */
    for(auto& eAxC_map:eAxC_map_gdr){
        eAxC_map.reset(gDev->newGDRbuf(MAX_AP_PER_SLOT * sizeof(uint16_t)));
        mf.addGpuPinnedSize(eAxC_map->size_alloc);
    }
    std::fill(std::begin(eAxC_num),std::end(eAxC_num),0);

    for(auto& eAxC_prach_map:eAxC_prach_map_gdr){
        eAxC_prach_map.reset(gDev->newGDRbuf(MAX_AP_PER_SLOT * sizeof(uint16_t)));
        mf.addGpuPinnedSize(eAxC_prach_map->size_alloc);
    }
    std::fill(std::begin(eAxC_prach_num),std::end(eAxC_prach_num),0);

    for(auto& eAxC_srs_map:eAxC_srs_map_gdr){
        eAxC_srs_map.reset(gDev->newGDRbuf(MAX_AP_PER_SLOT_SRS * sizeof(uint16_t)));
        mf.addGpuPinnedSize(eAxC_srs_map->size_alloc);
    }
    std::fill(std::begin(eAxC_srs_num),std::end(eAxC_srs_num),0);


    cell_order_list_size = 0;

    CUDA_CHECK_PHYDRIVER(cudaMallocHost((void**)&order_kernel_config_params, sizeof(orderKernelConfigParams_t)));
    CUDA_CHECK_PHYDRIVER(cudaMallocHost((void**)&order_kernel_config_params_srs, sizeof(orderKernelConfigParamsSrs_t)));
    CUDA_CHECK_PHYDRIVER(cudaMallocHost((void**)&orderKernelConfigParamsCpuInitComms, sizeof(orderKernelConfigParamsCpuInitComms_t)));

    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_idle));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_order));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_order));

    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_idle_srs));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_order_srs));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_order_srs));    

    setOrderLaunchedStatus(false);
    setOrderLaunchedStatusSrs(false);
}

OrderEntity::~OrderEntity()
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->setUlCtx();

    CUDA_CHECK_PHYDRIVER(cudaFreeHost(order_kernel_config_params));
    CUDA_CHECK_PHYDRIVER(cudaFreeHost(order_kernel_config_params_srs));
    CUDA_CHECK_PHYDRIVER(cudaFreeHost(orderKernelConfigParamsCpuInitComms));

    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_idle));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_order));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_order));

    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_idle_srs));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_order_srs));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_order_srs));
}

phydriver_handle OrderEntity::getPhyDriverHandler(void) const
{
    return pdh;
}

uint64_t OrderEntity::getId() const
{
    return id;
}

int OrderEntity::reserve(int32_t* cell_idx_list, uint8_t cell_idx_list_size, bool new_order_entity)
{
    int ret = 0;
    Cell * cell_ptr=nullptr;
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    mlock.lock();
    if(new_order_entity)
    {
        if(active == true)
            ret = -1;
        else
        {
            active   = true;
            cell_num=0;
            memset(order_kernel_config_params, 0, sizeof(orderKernelConfigParams_t));
            if(pdctx->get_enable_srs()){
                memset(order_kernel_config_params_srs, 0, sizeof(orderKernelConfigParamsSrs_t));
            }            
        }
    }
    mlock.unlock();

    for(uint32_t j = 0; j < cell_idx_list_size; j ++)
    {
        auto cell_idx = cell_idx_list[j];
        bool cell_idx_found = false;
        for(uint32_t i =0; i < cell_order_list_size; i++)
        {
            if(cell_order_list[i] == (int) cell_idx)
            {
                cell_idx_found = true;
                break;
            }
        }
        if(cell_idx_found == true)
        {
            continue;
        }

        cell_ptr = pdctx->getCellByPhyId(cell_idx);
        if(cell_ptr == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No valid cell associated to OrderEntity obj");
            return -1;
        }
        cell_id[cell_num] = cell_ptr->getId();
        rx_pkts[cell_num]             = 0;
        eAxC_num[cell_num]            = 0;
        eAxC_prach_num[cell_num]      = 0;
        eAxC_srs_num[cell_num]      = 0;

        // TODO FIXME HACK
        for(auto f : cell_ptr->geteAxCIdsPusch())
        {
            ((uint16_t*)(eAxC_map_gdr[cell_num]->addrh()))[eAxC_num[cell_num]] = f;
            eAxC_num[cell_num]++;
        }

        for(int tmp = eAxC_num[cell_num]; tmp < MAX_AP_PER_SLOT; tmp++)
            ((uint16_t*)(eAxC_map_gdr[cell_num]->addrh()))[tmp] = 0xFFFF;

        for(auto f : cell_ptr->geteAxCIdsPrach())
        {
            ((uint16_t*)(eAxC_prach_map_gdr[cell_num]->addrh()))[eAxC_prach_num[cell_num]] = f;
            eAxC_prach_num[cell_num]++;
        }

        for(auto f : cell_ptr->geteAxCIdsSrs())
        {
            ((uint16_t*)(eAxC_srs_map_gdr[cell_num]->addrh()))[eAxC_srs_num[cell_num]] = f;
            eAxC_srs_num[cell_num]++;
        }

        for(int tmp = eAxC_srs_num[cell_num]; tmp < MAX_AP_PER_SLOT_SRS; tmp++)
            ((uint16_t*)(eAxC_srs_map_gdr[cell_num]->addrh()))[tmp] = 0xFFFF;

        NVLOGD_FMT(TAG,"cell_num({}),cell_id({})",cell_num,cell_id[cell_num]);
        cell_num++;
        cell_order_list[cell_order_list_size] = cell_idx;
        cell_order_list_size++;
    }
    reserve_t = Time::nowNs();
    return ret;
}

void OrderEntity::release()
{
    mlock.lock();

    active  = false;
    std::fill(std::begin(cell_id),std::end(cell_id),0);
    std::fill(std::begin(rx_pkts),std::end(rx_pkts),0);
    cell_order_list_size = 0;
    cell_num=0;

    // NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ORDER {} released after {} us", getId(), Time::NsToUs(Time::nowNs() - reserve_t).count());

    mlock.unlock();
}

void OrderEntity::cleanup() {
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    int cell_idx;

    for(cell_idx=0;cell_idx<cell_num;cell_idx++)
    {
        Cell * cell_ptr = pdctx->getCellById(cell_id[cell_idx]);
        if(cell_ptr == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No cell for id {}",cell_id[cell_idx]);
            return;
        }

        cell_ptr->updateMetric(CellMetric::kPrachLostPrbsTotal, prach_prbs_x_slot[cell_idx] - *((uint32_t*)ordered_prbs_prach[cell_idx]->addrh()));
        cell_ptr->updateMetric(CellMetric::kPuschLostPrbsTotal, pusch_prbs_x_slot[cell_idx] - *((uint32_t*)ordered_prbs_pusch[cell_idx]->addrh()));

        if(cell_idx == 0) {
            NVLOGD_FMT(TAG,"[Order Kernel config Params After-1] pusch_eAxC_map({}),eAxC_pusch_num({})pusch_buffer({}),pusch_prb_x_slot({}),pusch_prb_stride({}),pusch_ordered_prbs({})",
                (void*)order_kernel_config_params->pusch_eAxC_map[0],order_kernel_config_params->pusch_eAxC_num[0],(void*)order_kernel_config_params->pusch_buffer[0],
                order_kernel_config_params->pusch_prb_x_slot[0],order_kernel_config_params->pusch_prb_stride[0],*((uint32_t*)ordered_prbs_pusch[0]->addrh()));
            NVLOGD_FMT(TAG,"[Order Kernel config Params After-2] srs_eAxC_map({}),eAxC_srs_num({}),srs_buffer({}),srs_prb_x_slot({}),srs_prb_stride({}),srs_ordered_prbs({})",
                (void*)order_kernel_config_params->srs_eAxC_map[0],order_kernel_config_params->srs_eAxC_num[0],(void*)order_kernel_config_params->srs_buffer[0],
                order_kernel_config_params->srs_prb_x_slot[0],order_kernel_config_params->srs_prb_stride[0],*((uint32_t*)ordered_prbs_srs[0]->addrh()));
        }

        cell_ptr->updateMetric(CellMetric::kEarlyUplanePackets, *((uint32_t*)early_rx_packets[cell_idx]->addrh()));
        cell_ptr->updateMetric(CellMetric::kOnTimeUplanePackets, *((uint32_t*)on_time_rx_packets[cell_idx]->addrh()));
        cell_ptr->updateMetric(CellMetric::kLateUplanePackets, *((uint32_t*)late_rx_packets[cell_idx]->addrh()));

        ACCESS_ONCE(*((uint32_t*)early_rx_packets[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)on_time_rx_packets[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)late_rx_packets[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)rx_packets_count[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)rx_bytes_count[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)rx_packets_dropped_count[cell_idx]->addrh())) = 0;

        cell_ptr->updateMetric(CellMetric::kEarlyUplanePacketsSrs, *((uint32_t*)early_rx_packets_srs[cell_idx]->addrh()));
        cell_ptr->updateMetric(CellMetric::kOnTimeUplanePacketsSrs, *((uint32_t*)on_time_rx_packets_srs[cell_idx]->addrh()));
        cell_ptr->updateMetric(CellMetric::kLateUplanePacketsSrs, *((uint32_t*)late_rx_packets_srs[cell_idx]->addrh()));

        ACCESS_ONCE(*((uint32_t*)early_rx_packets_srs[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)on_time_rx_packets_srs[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)late_rx_packets_srs[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)rx_packets_count_srs[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)rx_bytes_count_srs[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)rx_packets_dropped_count_srs[cell_idx]->addrh())) = 0;
        ACCESS_ONCE(*((uint32_t*)start_cuphy_gdr[cell_idx]->addrh()))  = 0;
        if(pdctx->cpuCommEnabled())
        {
            ACCESS_ONCE(*((uint32_t*)order_kernel_exit_cond_gdr[cell_idx]->addrh())) = ORDER_KERNEL_IDLE;        
        }
        else
        {
            ACCESS_ONCE(*((uint32_t*)order_kernel_exit_cond_gdr[cell_idx]->addrh())) = ORDER_KERNEL_RUNNING;    
        }
        ACCESS_ONCE(*((uint32_t*)start_cuphy_srs_gdr[cell_idx]->addrh()))  = 0;
        ACCESS_ONCE(*((uint32_t*)order_kernel_srs_exit_cond_gdr[cell_idx]->addrh())) = ORDER_KERNEL_RUNNING;        
        ACCESS_ONCE(*((uint32_t*)start_cuphy_cpu_h->addr())) = 0;
        ACCESS_ONCE(*((uint32_t*)start_cuphy_srs_cpu_h->addr())) = 0;
        for(int sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
            ACCESS_ONCE(*((uint32_t*)rx_packets_count[cell_idx]->addrh()+sym_idx))=0;
            ACCESS_ONCE(*((uint32_t*)rx_bytes_count[cell_idx]->addrh()+sym_idx))=0;
        }        
    }
    for(int sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
        ACCESS_ONCE(*((uint32_t*)sym_ord_done_mask_arr->addrh()+sym_idx))=0;
        ACCESS_ONCE(*((uint32_t*)num_order_cells_sym_mask_arr->addrh()+sym_idx))=0;
    }
    
    setOrderLaunchedStatus(false);
    setOrderLaunchedStatusSrs(false);
}

int OrderEntity::runOrder(
    slot_command_api::oran_slot_ind oran_ind,
    uint16_t* puschNumPrb, uint8_t ** buf_st_1, uint8_t** buf_pcap_capture, uint8_t** buf_pcap_capture_ts,
    uint16_t* prachNumPrb,
    uint8_t ** buf_st_3_o0, uint8_t ** buf_st_3_o1, uint8_t ** buf_st_3_o2, uint8_t ** buf_st_3_o3,
    uint16_t prachSectionId_o0, uint16_t prachSectionId_o1, uint16_t prachSectionId_o2, uint16_t prachSectionId_o3,
    uint32_t* srsNumPrb, uint8_t **buf_srs,
    std::array<t_ns,UL_MAX_CELLS_PER_SLOT>& slot_start, 
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& ta4_min_ns, std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& ta4_max_ns,
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& ta4_min_ns_srs, std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& ta4_max_ns_srs,
    uint8_t num_order_cells,uint32_t srsCellMask,std::array<uint8_t,UL_MAX_CELLS_PER_SLOT>& srs_start_symbol,
    uint32_t nonSrsUlCellMask,uint32_t* sym_ord_arr_addr,
    std::array<uint32_t,UL_MAX_CELLS_PER_SLOT*ORAN_PUSCH_SYMBOLS_X_SLOT>& pusch_prb_symbol_map,
    std::array<uint32_t,ORAN_PUSCH_SYMBOLS_X_SLOT>& num_order_cells_sym_mask,uint8_t pusch_prb_non_zero,uint32_t slot_map_id
)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell * cell_ptr = nullptr;
    int cell_idx,sym_idx;
    FhProxy * fhproxy = pdctx->getFhProxy();
    struct rx_order_t * rx_order_item;
    cudaStream_t first_strm;
    pdctx->setUlCtx();
    uint8_t num_order_cells_srs =0;
    uint8_t num_order_cells_nonSrsUl =0;
    uint8_t cell_count =0;
    CuphyOAM *oam = CuphyOAM::getInstance();
    t_ns t1 = Time::nowNs();
    auto pcap_capture_cell_bitmask = oam->ul_pcap_arm_cell_bitmask.load();
    /////////////////////////////////////////////////////////////////
    ///////// Validate params before starting
    /////////////////////////////////////////////////////////////////

    if(num_order_cells!=cell_num)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Mis-match between cell_num : {}" "and num_order_cells : {}" "for this ORDER {}\n" ,cell_num , num_order_cells, getId());
        return EINVAL;
    }

    if(srsCellMask)
    {
        memset(order_kernel_config_params_srs, 0, sizeof(orderKernelConfigParamsSrs_t));
        
        ACCESS_ONCE(*((int*)order_barrier_flag->addrh())) = 0;
        order_kernel_config_params_srs->barrier_flag=(int*)order_barrier_flag->addrd();
    
        first_strm=pdctx->getUlOrderStreamSrsPd();
        
        NVLOGD_FMT(TAG, "Cell mask for SRS channel: {}", srsCellMask);
        uint32_t fh_buf_slot_idx=pdctx->getConfigOkTbSrsNumSlots();
        NVLOGI_FMT(TAG,"[SRS]fh_buf_slot_idx:{} frameId:{} subframeId:{} slotId:{}",fh_buf_slot_idx,oran_ind.oframe_id_,oran_ind.osfid_,oran_ind.oslotid_);        

        //Loop to populate Order kernel config params for all the needed cells
        for(cell_idx=0;cell_idx<num_order_cells;cell_idx++)
        {
            if(srsCellMask & (1 << cell_idx))
            {
                cell_ptr = pdctx->getCellById(cell_id[cell_idx]);
                if(cell_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No cell for this ORDER {}", getId());
                    return EINVAL;
                }
                if(buf_srs[cell_idx] == nullptr || srsNumPrb[cell_idx] == 0)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Nothing to reorder ORDER {}", getId());
                    return EINVAL;
                }

                srs_prbs_x_slot[cell_count] = srsNumPrb[cell_idx];

                order_kernel_config_params_srs->rxq_info_gpu[cell_count]=cell_ptr->docaGetRxqInfoSrs()->eth_rxq_gpu; //FIXME : Add seperate DOCA RxQ API for SRS
                order_kernel_config_params_srs->sem_gpu[cell_count]=cell_ptr->docaGetRxqInfoSrs()->sem_gpu;//FIXME : Add seperate DOCA  API for SRS
                order_kernel_config_params_srs->sem_gpu_aerial_fh[cell_count]=cell_ptr->docaGetRxqInfoSrs()->sem_gpu_aerial_fh;
                order_kernel_config_params_srs->sem_order_num[cell_count]=cell_ptr->docaGetSemNumSrs();//FIXME : Add seperate DOCA  API for SRS

                order_kernel_config_params_srs->cell_id[cell_count]=cell_ptr->getPhyId();
                order_kernel_config_params_srs->ru_type[cell_count]=(int)cell_ptr->getRUType();
                order_kernel_config_params_srs->cell_health[cell_count]=cell_ptr->isHealthySrs();

                order_kernel_config_params_srs->start_cuphy_d[cell_count]= (uint32_t*)start_cuphy_srs_gdr[cell_idx]->addrd();
                order_kernel_config_params_srs->order_kernel_exit_cond_d[cell_count]=(uint32_t*)order_kernel_srs_exit_cond_gdr[cell_idx]->addrd();

                order_kernel_config_params_srs->last_sem_idx_rx_h[cell_count]=cell_ptr->getLastRxSrsItem();
                order_kernel_config_params_srs->last_sem_idx_order_h[cell_count]=cell_ptr->getLastOrderedSrsItem();

                order_kernel_config_params_srs->comp_meth[cell_count]=(int)cell_ptr->getULCompMeth();
                order_kernel_config_params_srs->bit_width[cell_count]=(int)cell_ptr->getULBitWidth();
                if((order_kernel_config_params_srs->bit_width[cell_count] != BFP_NO_COMPRESSION &&
                    order_kernel_config_params_srs->bit_width[cell_count] != BFP_COMPRESSION_14_BITS &&
                    order_kernel_config_params_srs->bit_width[cell_count] != BFP_COMPRESSION_9_BITS &&
                    order_kernel_config_params_srs->comp_meth[cell_count] == (int)aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT) ||
                (order_kernel_config_params_srs->bit_width[cell_count] != 16 &&
                    order_kernel_config_params_srs->comp_meth[cell_count] == (int)aerial_fh::UserDataCompressionMethod::NO_COMPRESSION))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Run ORDER {} error, cell id {}, unsupported comp_meth {}, bitwidth {}\n", getId(), cell_idx, order_kernel_config_params_srs->comp_meth[cell_count], order_kernel_config_params_srs->bit_width[cell_count]);
                    return EINVAL;
                }
                order_kernel_config_params_srs->beta[cell_count]=(float)cell_ptr->getBetaUlPowerScaling();

                order_kernel_config_params_srs->done_shared[cell_count]=(uint8_t*)done_shared_srs[cell_idx]->addr();

                order_kernel_config_params_srs->early_rx_packets[cell_count]=(uint32_t*)early_rx_packets_srs[cell_idx]->addrd();
                order_kernel_config_params_srs->on_time_rx_packets[cell_count]=(uint32_t*)on_time_rx_packets_srs[cell_idx]->addrd();
                order_kernel_config_params_srs->late_rx_packets[cell_count]=(uint32_t*)late_rx_packets_srs[cell_idx]->addrd();
                order_kernel_config_params_srs->next_slot_early_rx_packets[cell_count]=cell_ptr->getNextSlotEarlyRxPacketsSRS();
                order_kernel_config_params_srs->next_slot_on_time_rx_packets[cell_count]=cell_ptr->getNextSlotOnTimeRxPacketsSRS();
                order_kernel_config_params_srs->next_slot_late_rx_packets[cell_count]=cell_ptr->getNextSlotLateRxPacketsSRS();

                order_kernel_config_params_srs->next_slot_rx_packets_count[cell_count] = cell_ptr->getNextSlotRxPacketCountSRS();
                order_kernel_config_params_srs->next_slot_rx_bytes_count[cell_count]   = cell_ptr->getNextSlotRxByteCountSRS();

                if(pdctx->enableOKTb())
                {
                    order_kernel_config_params_srs->fh_buf_ok_tb_slot[cell_count]=pdctx->getFhBufOkTbSrs(cell_idx)+ (fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB)*(MAX_PKTS_PER_SLOT_OK_TB*pdctx->getConfigOkTbMaxPacketSize());
                    order_kernel_config_params_srs->fh_buf_ok_tb_next_slot[cell_count]=pdctx->getFhBufOkTbSrs(cell_idx)+ ((fh_buf_slot_idx+1)%MAX_UL_SLOTS_OK_TB)*(MAX_PKTS_PER_SLOT_OK_TB*pdctx->getConfigOkTbMaxPacketSize());
                    pdctx->getOkTbConfigSrs(cell_idx)->num_rx_packets[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=0;
                    pdctx->getOkTbConfigSrs(cell_idx)->num_srs_prbs[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=srsNumPrb[cell_idx];
                    pdctx->getOkTbConfigSrs(cell_idx)->frameId[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=oran_ind.oframe_id_;
                    pdctx->getOkTbConfigSrs(cell_idx)->subframeId[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=oran_ind.osfid_;
                    pdctx->getOkTbConfigSrs(cell_idx)->slotId[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=oran_ind.oslotid_;
                    pdctx->getOkTbConfigSrs(cell_idx)->srs_eAxC_num=eAxC_srs_num[cell_idx];
                    order_kernel_config_params_srs->next_slot_num_prb_ch1[cell_count]=cell_ptr->getNextSlotNumPrbCh3();
                    for(int tmp = 0; tmp < eAxC_srs_num[cell_idx]; tmp++)
                    {
                        pdctx->getOkTbConfigSrs(cell_idx)->srs_eAxC_map[tmp]=((uint16_t*)(eAxC_srs_map_gdr[cell_idx]->addrh()))[tmp];
                    }
                    pdctx->getOkTbConfigSrs(cell_idx)->cell_id=cell_ptr->getPhyId();
                    order_kernel_config_params_srs->rx_packets_count[cell_count]=(uint32_t*)&pdctx->getOkTbConfigSrs(cell_idx)->num_rx_packets[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB];
                }
                else
                {
                    order_kernel_config_params_srs->rx_packets_count[cell_count]           = (uint32_t*)rx_packets_count_srs[cell_idx]->addrd();
                    order_kernel_config_params_srs->rx_bytes_count[cell_count]             = (uint32_t*)rx_bytes_count_srs[cell_idx]->addrd();
                    order_kernel_config_params_srs->rx_packets_dropped_count[cell_count]   = (uint32_t*)rx_packets_dropped_count_srs[cell_idx]->addrd();
                }                

                order_kernel_config_params_srs->rx_packets_ts[cell_count]=(uint64_t*)rx_packets_ts_srs[cell_idx]->addrd();
                order_kernel_config_params_srs->rx_packets_ts_earliest[cell_count]=(uint64_t*)rx_packets_ts_earliest_srs[cell_idx]->addrd();
                order_kernel_config_params_srs->rx_packets_ts_latest[cell_count]=(uint64_t*)rx_packets_ts_latest_srs[cell_idx]->addrd();
                order_kernel_config_params_srs->rx_packets_per_sym_count[cell_count]=(uint32_t*)rx_packets_count_per_sym_srs[cell_idx]->addrd();
                order_kernel_config_params_srs->next_slot_rx_packets_ts[cell_count]=(uint64_t*)cell_ptr->getNextSlotRxPacketTsSRS();
                order_kernel_config_params_srs->next_slot_rx_packets_per_sym_count[cell_count]=(uint32_t*)cell_ptr->getNextSlotRxPacketCountPerSymSRS();                 

                order_kernel_config_params_srs->slot_start[cell_count]=slot_start[cell_idx].count();
                order_kernel_config_params_srs->ta4_min_ns[cell_count]=ta4_min_ns_srs[cell_idx];
                order_kernel_config_params_srs->ta4_max_ns[cell_count]=ta4_max_ns_srs[cell_idx];

                order_kernel_config_params_srs->slot_duration[cell_count]=cell_ptr->getTtiNsFromMu(cell_ptr->getMu());

                order_kernel_config_params_srs->srs_eAxC_map[cell_count]=(uint16_t*)eAxC_srs_map_gdr[cell_idx]->addrd();
                order_kernel_config_params_srs->srs_eAxC_num[cell_count]=eAxC_srs_num[cell_idx];
                order_kernel_config_params_srs->srs_buffer[cell_count]=buf_srs[cell_idx];
                order_kernel_config_params_srs->srs_prb_x_slot[cell_count]=srs_prbs_x_slot[cell_count];
                order_kernel_config_params_srs->srs_prb_stride[cell_count]=cell_ptr->getSrsPrbStride();
                order_kernel_config_params_srs->srs_ordered_prbs[cell_count]=(uint32_t*)ordered_prbs_srs[cell_idx]->addrd();
                order_kernel_config_params_srs->srs_start_sym[cell_count]=srs_start_symbol[cell_idx];
                order_kernel_config_params_srs->order_kernel_last_timeout_error_time[cell_count]=(uint64_t*)cell_ptr->getOrderkernelSrsLastTimeoutErrorTimeItem();
                DOCA_GPUNETIO_VOLATILE(((uint32_t*)(order_kernel_srs_exit_cond_gdr[cell_count]->addrh()))[0]) = ORDER_KERNEL_RUNNING;
                
                num_order_cells_srs++;
                cell_count++;
    /*
                NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-0] DOCA GPU RxQ Info({}),DOCA Sem Num({}),Last Sem Item({}),Last Ordered Item({})",
                                (void*)order_kernel_config_params_srs->rxq_info_gpu[cell_count],
                                order_kernel_config_params_srs->sem_order_num[cell_count],
                                (void*)order_kernel_config_params_srs->last_sem_idx_rx_h[cell_count],
                                (void*)order_kernel_config_params_srs->last_sem_idx_order_h[cell_count]);
                NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-1] Cell ID({}),SFN({}),Sub-frame({}),Slot({}),BFP({}),Beta({})",
                                order_kernel_config_params_srs->cell_id[cell_count],
                                oran_ind.oframe_id_,oran_ind.osfid_,oran_ind.oslotid_,
                                order_kernel_config_params_srs->comp_meth[cell_count],
                                order_kernel_config_params_srs->beta[cell_count]);
                NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-2] Ordered Barrier flag addr({}),Done shared addr({}),SlotStart({}),ta4_min_ns({}),ta4_max_ns({}),slot_duration({})",
                                (void*)order_kernel_config_params_srs->barrier_flag,
                                (void*)order_kernel_config_params_srs->done_shared[cell_count],
                                order_kernel_config_params_srs->slot_start[cell_count],
                                order_kernel_config_params_srs->ta4_min_ns[cell_count],
                                order_kernel_config_params_srs->ta4_max_ns[cell_count],
                                order_kernel_config_params_srs->slot_duration[cell_count]);
                NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-3] srs_eAxC_map({}),eAxC_srs_num({}),srs_eAxC_val0({:x}),srs_eAxC_val1({:x}),srs_buffer({}),srsNumPrb({}),srs_prb_stride({}),srs_ordered_prb_addr({})",
                                (void*)order_kernel_config_params_srs->srs_eAxC_map[cell_count],
                                order_kernel_config_params_srs->srs_eAxC_num[cell_count],
                                ((uint16_t*)(eAxC_srs_map_gdr[cell_idx]->addrh()))[0],
                                ((uint16_t*)(eAxC_srs_map_gdr[cell_idx]->addrh()))[1],
                                (void*)order_kernel_config_params_srs->srs_buffer[cell_count],
                                order_kernel_config_params_srs->srs_prb_x_slot[cell_count],
                                order_kernel_config_params_srs->srs_prb_stride[cell_count],
                                ((void*)order_kernel_config_params_srs->srs_ordered_prbs[cell_count]));
                                
    */
            }
        }

        if(pdctx->enableOKTb())
        {
            pdctx->setConfigOkTbSrsNumSlots((fh_buf_slot_idx+1));
        }        

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////// Start Order Kernel
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_idle_srs, first_strm));
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_order_srs, first_strm));

        if(pdctx->enableOKTb())
        {
            if(launch_receive_kernel_for_test_bench(first_strm,
                                    order_kernel_config_params_srs->rxq_info_gpu,
                                    order_kernel_config_params_srs->sem_gpu,
                                    order_kernel_config_params_srs->sem_order_num,

                                    order_kernel_config_params_srs->cell_id,
                                    order_kernel_config_params_srs->order_kernel_exit_cond_d, // Notify CPU order kernel completion
                                    order_kernel_config_params_srs->last_sem_idx_rx_h,
                                    order_kernel_config_params_srs->bit_width,

                                    pdctx->getUlOrderTimeoutGPUSrs(),
                                    pdctx->getUlOrderTimeoutFirstPktGPUSrs(),
                                    pdctx->getUlOrderMaxRxPkts(),
                                    pdctx->getConfigOkTbMaxPacketSize(),

                                    oran_ind.oframe_id_,
                                    oran_ind.osfid_,
                                    oran_ind.oslotid_,

                                    order_kernel_config_params_srs->rx_packets_count,
                                    order_kernel_config_params_srs->next_slot_rx_packets_count,
                                    order_kernel_config_params_srs->next_slot_num_prb_ch1,
                                    nullptr,

                                    /*FH buffer*/
                                    order_kernel_config_params_srs->fh_buf_ok_tb_slot,
                                    order_kernel_config_params_srs->fh_buf_ok_tb_next_slot,

                                    /* PUSCH/PRACH info */
                                    nullptr,
                                    nullptr,
                                    0, 0, 0, 0,

                                    /* SRS info */
                                    order_kernel_config_params_srs->srs_prb_x_slot,
                                    num_order_cells_srs
                                ))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Run ORDER {} error", getId());
                return EINVAL;
            }
        }
        else
        {
            if(pdctx->getUlOrderKernelMode()==0) //Ping-Pong mode
            {
                if(pdctx->get_ru_type_for_srs_proc() != SINGLE_SECT_MODE) { //Launch SRS order kernel only for non FX RU here -- srsCellMask is 0 for ru_type::SINGLE_SECT_MODE
                    if(launch_order_kernel_doca_single_subSlot(first_strm,
                                            order_kernel_config_params_srs->rxq_info_gpu,
                                            order_kernel_config_params_srs->sem_gpu,
                                            order_kernel_config_params_srs->sem_gpu_aerial_fh,
                                            order_kernel_config_params_srs->sem_order_num,

                                            order_kernel_config_params_srs->cell_id,
                                            order_kernel_config_params_srs->ru_type,
                                            order_kernel_config_params_srs->cell_health,

                                            order_kernel_config_params_srs->start_cuphy_d, // Unblock cuPHY
                                            order_kernel_config_params_srs->order_kernel_exit_cond_d, // Notify CPU order kernel completion

                                            order_kernel_config_params_srs->last_sem_idx_rx_h,
                                            order_kernel_config_params_srs->last_sem_idx_order_h,

                                            0, //Ping-Pong mode
                                            pdctx->getUlOrderTimeoutGPUSrs(),
                                            pdctx->getUlOrderTimeoutFirstPktGPUSrs(),
                                            pdctx->getUlOrderTimeoutLogInterval(),
                                            pdctx->getUlOrderTimeoutGPULogEnable(),
                                            pdctx->getUlOrderMaxRxPkts(),
                                            pdctx->getUlOrderRxPktsTimeout(),
                                            0,

                                            oran_ind.oframe_id_,
                                            oran_ind.osfid_,
                                            oran_ind.oslotid_,
                                            order_kernel_config_params_srs->comp_meth,
                                            order_kernel_config_params_srs->bit_width,
                                            DEFAULT_PRB_STRIDE,
                                            order_kernel_config_params_srs->beta,
                                            order_kernel_config_params_srs->barrier_flag, //((int*)order_barrier_flag->addr()),
                                            order_kernel_config_params_srs->done_shared,

                                            /* RX time stats */
                                            order_kernel_config_params_srs->early_rx_packets,
                                            order_kernel_config_params_srs->on_time_rx_packets,
                                            order_kernel_config_params_srs->late_rx_packets,
                                            order_kernel_config_params_srs->next_slot_early_rx_packets,
                                            order_kernel_config_params_srs->next_slot_on_time_rx_packets,
                                            order_kernel_config_params_srs->next_slot_late_rx_packets,                                         
                                            order_kernel_config_params_srs->slot_start,
                                            order_kernel_config_params_srs->ta4_min_ns,
                                            order_kernel_config_params_srs->ta4_max_ns,
                                            order_kernel_config_params_srs->slot_duration,
                                            order_kernel_config_params_srs->order_kernel_last_timeout_error_time,
                                            (uint8_t)pdctx->getUlRxPktTracingLevelSrs(),
                                            order_kernel_config_params_srs->rx_packets_ts,
                                            order_kernel_config_params_srs->rx_packets_count,     
                                            order_kernel_config_params_srs->rx_bytes_count,     
                                            order_kernel_config_params_srs->rx_packets_ts_earliest,
                                            order_kernel_config_params_srs->rx_packets_ts_latest,
                                            order_kernel_config_params_srs->next_slot_rx_packets_ts,
                                            order_kernel_config_params_srs->next_slot_rx_packets_count,
                                            order_kernel_config_params_srs->next_slot_rx_bytes_count,
                                            order_kernel_config_params_srs->rx_packets_dropped_count,
                                            nullptr,
                                            nullptr,

                                            /*Sub-slot Processing*/
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            pusch_prb_non_zero,

                                            /* PUSCH info */
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,

                                            /* PRACH info */
                                            nullptr,
                                            nullptr,
                                            nullptr, nullptr, nullptr, nullptr,
                                            0, 0, 0, 0,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,

                                            /* SRS info (not used for non-SRS UL launch)*/
                                            order_kernel_config_params_srs->srs_eAxC_map,
                                            order_kernel_config_params_srs->srs_eAxC_num,
                                            order_kernel_config_params_srs->srs_buffer,
                                            order_kernel_config_params_srs->srs_prb_x_slot,
                                            order_kernel_config_params_srs->srs_prb_stride,
                                            order_kernel_config_params_srs->srs_ordered_prbs,
                                            order_kernel_config_params_srs->srs_start_sym,

                                            num_order_cells_srs,
                                            /* PCAP capture info */
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            0,
                                            /* SRS enable */
                                            1
                                        ))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Run ORDER {} error", getId());
                        return EINVAL;
                    }
            }
            }
            else
            {
                if(launch_order_kernel_doca_single_srs(first_strm,
                                        order_kernel_config_params_srs->rxq_info_gpu,
                                        order_kernel_config_params_srs->sem_gpu,
                                        order_kernel_config_params_srs->sem_order_num,

                                        order_kernel_config_params_srs->cell_id,
                                        order_kernel_config_params_srs->ru_type,

                                        order_kernel_config_params_srs->start_cuphy_d, // Unblock cuPHY
                                        order_kernel_config_params_srs->order_kernel_exit_cond_d, // Notify CPU order kernel completion

                                        order_kernel_config_params_srs->last_sem_idx_rx_h,
                                        order_kernel_config_params_srs->last_sem_idx_order_h,

                                        pdctx->getUlOrderTimeoutGPUSrs(),
                                        pdctx->getUlOrderTimeoutFirstPktGPUSrs(),
                                        pdctx->getUlOrderMaxRxPkts(),//TODO:Tune this value for SRS

                                        oran_ind.oframe_id_,
                                        oran_ind.osfid_,
                                        oran_ind.oslotid_,
                                        order_kernel_config_params_srs->comp_meth,
                                        order_kernel_config_params_srs->bit_width,
                                        DEFAULT_PRB_STRIDE,
                                        order_kernel_config_params_srs->beta,
                                        order_kernel_config_params_srs->barrier_flag, //((int*)order_barrier_flag->addr()),
                                        order_kernel_config_params_srs->done_shared,

                                        /* SRS RX time stats */
                                        order_kernel_config_params_srs->early_rx_packets,
                                        order_kernel_config_params_srs->on_time_rx_packets,
                                        order_kernel_config_params_srs->late_rx_packets,
                                        order_kernel_config_params_srs->next_slot_early_rx_packets,
                                        order_kernel_config_params_srs->next_slot_on_time_rx_packets,
                                        order_kernel_config_params_srs->next_slot_late_rx_packets,
                                        order_kernel_config_params_srs->rx_packets_count,
                                        order_kernel_config_params_srs->rx_bytes_count,
                                        order_kernel_config_params_srs->next_slot_rx_packets_count,
                                        order_kernel_config_params_srs->next_slot_rx_bytes_count,
                                        (uint8_t)pdctx->getUlRxPktTracingLevelSrs(),
                                        order_kernel_config_params_srs->rx_packets_ts,
                                        order_kernel_config_params_srs->rx_packets_per_sym_count,     
                                        order_kernel_config_params_srs->rx_packets_ts_earliest,
                                        order_kernel_config_params_srs->rx_packets_ts_latest,
                                        order_kernel_config_params_srs->next_slot_rx_packets_ts,
                                        order_kernel_config_params_srs->next_slot_rx_packets_per_sym_count,                                 
                                        order_kernel_config_params_srs->slot_start,
                                        order_kernel_config_params_srs->ta4_min_ns,
                                        order_kernel_config_params_srs->ta4_max_ns,
                                        order_kernel_config_params_srs->slot_duration,
                                        /* SRS info */
                                        order_kernel_config_params_srs->srs_eAxC_map,
                                        order_kernel_config_params_srs->srs_eAxC_num,
                                        order_kernel_config_params_srs->srs_buffer,
                                        order_kernel_config_params_srs->srs_prb_x_slot,
                                        order_kernel_config_params_srs->srs_prb_stride,
                                        order_kernel_config_params_srs->srs_ordered_prbs,
                                        order_kernel_config_params_srs->srs_start_sym,
                                        num_order_cells_srs
                                    ))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Run ORDER {} error", getId());
                    return EINVAL;
                }
            }            
        }


        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_order_srs, first_strm));       
        // Notify CPU RX thread (can be replaced by cudaEvent?)
        launch_kernel_write(first_strm, (uint32_t*)start_cuphy_srs_cpu_h->addr(), 1);     
        setOrderLaunchedStatusSrs(true);                       
    }
    
    if(nonSrsUlCellMask) //PUSCH/PUCCH/PRACH
    {
        if(pdctx->cpuCommEnabled())
        {
            memset(orderKernelConfigParamsCpuInitComms, 0, sizeof(orderKernelConfigParamsCpuInitComms_t));

            ACCESS_ONCE(*((int*)order_barrier_flag->addrh())) = 0;
            orderKernelConfigParamsCpuInitComms->barrier_flag=(int*)order_barrier_flag->addrd();
        
            first_strm=pdctx->getUlOrderStreamPd();

            NVLOGD_FMT(TAG, "[CPU Init Comms]Cell mask for non SRS UL channels: {}", nonSrsUlCellMask);

            cell_count = 0;

            //Loop to populate Order kernel config params for all the needed cells
            for(cell_idx=0;cell_idx<num_order_cells;cell_idx++)
            {
                if(nonSrsUlCellMask & (1 << cell_idx))
                {
                    cell_ptr = pdctx->getCellById(cell_id[cell_idx]);
                    if(cell_ptr == nullptr)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No cell for this ORDER {}", getId());
                        return EINVAL;
                    }
                    if(
                        (buf_st_1[cell_idx] == nullptr || puschNumPrb[cell_idx] == 0) &&
                        ( (buf_st_3_o0[cell_idx] == nullptr && buf_st_3_o1[cell_idx] == nullptr && buf_st_3_o2[cell_idx] == nullptr && buf_st_3_o3[cell_idx] == nullptr) || prachNumPrb[cell_idx] == 0)
                    )
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Nothing to reorder ORDER {}", getId());
                        return EINVAL;
                    }

                    if(
                        (buf_st_1[cell_idx] != nullptr && puschNumPrb[cell_idx] == 0) ||
                        ((buf_st_3_o0[cell_idx] != nullptr && buf_st_3_o1[cell_idx] != nullptr && buf_st_3_o2[cell_idx] != nullptr && buf_st_3_o3[cell_idx] != nullptr) && prachNumPrb[cell_idx] == 0)
                    )
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Nothing to reorder ORDER {}", getId());
                        return EINVAL;
                    }

                    pusch_prbs_x_slot[cell_count] = puschNumPrb[cell_idx];
                    prach_prbs_x_slot[cell_count] = prachNumPrb[cell_idx];

                    orderKernelConfigParamsCpuInitComms->start_cuphy_d[cell_count]= (uint32_t*)start_cuphy_gdr[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->order_kernel_exit_cond_d[cell_count]=(uint32_t*)order_kernel_exit_cond_gdr[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->last_sem_idx_order_h[cell_count]=cell_ptr->getLastOrderedItem();
                    rx_order_item = fhproxy->getRxOrderItemsPeer(cell_ptr->getPeerId());
                    if(rx_order_item == nullptr)
                    {
                        NVLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT, " Peer id not valid %lu\n", cell_ptr->getPeerId());
                        return EINVAL;
                    }
                    orderKernelConfigParamsCpuInitComms->ready_list[cell_count]=(uint32_t*)rx_order_item->sync_ready_list_gdr->addrd();
                    orderKernelConfigParamsCpuInitComms->rx_queue_sync_list[cell_count]=rx_order_item->sync_list;

                    orderKernelConfigParamsCpuInitComms->comp_meth[cell_count]=(int)cell_ptr->getULCompMeth();
                    orderKernelConfigParamsCpuInitComms->bit_width[cell_count]=(int)cell_ptr->getULBitWidth();
                    
                    if((orderKernelConfigParamsCpuInitComms->bit_width[cell_count] != BFP_NO_COMPRESSION &&
                        orderKernelConfigParamsCpuInitComms->bit_width[cell_count] != BFP_COMPRESSION_14_BITS &&
                        orderKernelConfigParamsCpuInitComms->bit_width[cell_count] != BFP_COMPRESSION_9_BITS &&
                        orderKernelConfigParamsCpuInitComms->comp_meth[cell_count] == (int)aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT) ||
                    (orderKernelConfigParamsCpuInitComms->bit_width[cell_count] != 16 &&
                        orderKernelConfigParamsCpuInitComms->comp_meth[cell_count] == (int)aerial_fh::UserDataCompressionMethod::NO_COMPRESSION))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Run ORDER {} error, cell id {}, unsupported comp_meth {}, bitwidth {}\n", getId(), cell_idx, orderKernelConfigParamsCpuInitComms->comp_meth[cell_idx], orderKernelConfigParamsCpuInitComms->bit_width[cell_idx]);
                        return EINVAL;
                    }

                    orderKernelConfigParamsCpuInitComms->beta[cell_count]=(float)cell_ptr->getBetaUlPowerScaling();
                    orderKernelConfigParamsCpuInitComms->sem_order_num[cell_count]=RX_QUEUE_SYNC_LIST_ITEMS;

                    orderKernelConfigParamsCpuInitComms->done_shared[cell_count]=(uint8_t*)done_shared[cell_idx]->addr();

                    orderKernelConfigParamsCpuInitComms->early_rx_packets[cell_count]=(uint32_t*)early_rx_packets[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->on_time_rx_packets[cell_count]=(uint32_t*)on_time_rx_packets[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->late_rx_packets[cell_count]=(uint32_t*)late_rx_packets[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->next_slot_early_rx_packets[cell_count]=cell_ptr->getNextSlotEarlyRxPackets();
                    orderKernelConfigParamsCpuInitComms->next_slot_on_time_rx_packets[cell_count]=cell_ptr->getNextSlotOnTimeRxPackets();
                    orderKernelConfigParamsCpuInitComms->next_slot_late_rx_packets[cell_count]=cell_ptr->getNextSlotLateRxPackets();
                    orderKernelConfigParamsCpuInitComms->rx_packets_ts[cell_count]=(uint64_t*)rx_packets_ts[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->rx_packets_count[cell_count]=(uint32_t*)rx_packets_count[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->rx_bytes_count[cell_count]=(uint32_t*)rx_bytes_count[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->rx_packets_ts_earliest[cell_count]=(uint64_t*)rx_packets_ts_earliest[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->rx_packets_ts_latest[cell_count]=(uint64_t*)rx_packets_ts_latest[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->next_slot_rx_packets_ts[cell_count]=(uint64_t*)cell_ptr->getNextSlotRxPacketTs();
                    orderKernelConfigParamsCpuInitComms->next_slot_rx_packets_count[cell_count]=(uint32_t*)cell_ptr->getNextSlotRxPacketCount();
                    orderKernelConfigParamsCpuInitComms->next_slot_rx_bytes_count[cell_count]=(uint32_t*)cell_ptr->getNextSlotRxByteCount();  

                    orderKernelConfigParamsCpuInitComms->slot_start[cell_count]=slot_start[cell_idx].count();
                    orderKernelConfigParamsCpuInitComms->ta4_min_ns[cell_count]=ta4_min_ns[cell_idx];
                    orderKernelConfigParamsCpuInitComms->ta4_max_ns[cell_count]=ta4_max_ns[cell_idx];
                    orderKernelConfigParamsCpuInitComms->slot_duration[cell_count]=cell_ptr->getTtiNsFromMu(cell_ptr->getMu());                    
                    
                    orderKernelConfigParamsCpuInitComms->pusch_eAxC_map[cell_count]=(uint16_t*)eAxC_map_gdr[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->pusch_eAxC_num[cell_count]=eAxC_num[cell_idx];
                    orderKernelConfigParamsCpuInitComms->pusch_buffer[cell_count]=buf_st_1[cell_idx];
                    orderKernelConfigParamsCpuInitComms->pusch_prb_x_slot[cell_count]=pusch_prbs_x_slot[cell_count];
                    orderKernelConfigParamsCpuInitComms->pusch_prb_x_symbol[cell_count]=pusch_prbs_x_slot[cell_count];
                    orderKernelConfigParamsCpuInitComms->pusch_prb_x_symbol_x_antenna[cell_count]=pusch_prbs_x_slot[cell_count];
                    orderKernelConfigParamsCpuInitComms->pusch_prb_stride[cell_count]=cell_ptr->getPuschPrbStride();
                    orderKernelConfigParamsCpuInitComms->pusch_ordered_prbs[cell_count]=(uint32_t*)ordered_prbs_pusch[cell_idx]->addrd();

                    orderKernelConfigParamsCpuInitComms->prach_eAxC_map[cell_count]=(uint16_t*)eAxC_prach_map_gdr[cell_idx]->addrd();
                    orderKernelConfigParamsCpuInitComms->prach_eAxC_num[cell_count]=eAxC_prach_num[cell_idx];
                    orderKernelConfigParamsCpuInitComms->prach_buffer_0[cell_count]=buf_st_3_o0[cell_idx];
                    orderKernelConfigParamsCpuInitComms->prach_buffer_1[cell_count]=buf_st_3_o1[cell_idx];
                    orderKernelConfigParamsCpuInitComms->prach_buffer_2[cell_count]=buf_st_3_o2[cell_idx];
                    orderKernelConfigParamsCpuInitComms->prach_buffer_3[cell_count]=buf_st_3_o3[cell_idx];
                    orderKernelConfigParamsCpuInitComms->prach_prb_x_slot[cell_count]=prach_prbs_x_slot[cell_count];
                    orderKernelConfigParamsCpuInitComms->prach_prb_x_symbol[cell_count]=prach_prbs_x_slot[cell_count];
                    orderKernelConfigParamsCpuInitComms->prach_prb_x_symbol_x_antenna[cell_count]=prach_prbs_x_slot[cell_count];
                    orderKernelConfigParamsCpuInitComms->prach_prb_stride[cell_count]=cell_ptr->getPrachPrbStride();
                    orderKernelConfigParamsCpuInitComms->prach_ordered_prbs[cell_count]=(uint32_t*)ordered_prbs_prach[cell_idx]->addrd();

                    for(sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
                        ACCESS_ONCE(*((uint32_t*)pusch_prb_symbol_map_gdr->addrh()+(cell_count*ORAN_PUSCH_SYMBOLS_X_SLOT)+sym_idx))=(uint32_t)pusch_prb_symbol_map[(cell_idx*ORAN_PUSCH_SYMBOLS_X_SLOT)+sym_idx];
                    }    

                    DOCA_GPUNETIO_VOLATILE(((uint32_t*)(order_kernel_exit_cond_gdr[cell_count]->addrh()))[0]) = ORDER_KERNEL_IDLE;

                    num_order_cells_nonSrsUl++;
                    cell_count++;        
            /*
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-0] DOCA GPU RxQ Info({}),DOCA Sem Num({}),Last Sem Item({}),Last Ordered Item({})",(void*)order_kernel_config_params->rxq_info_gpu[cell_count],order_kernel_config_params->sem_order_num[cell_count],(void*)order_kernel_config_params->last_sem_idx_rx_h[cell_count],(void*)order_kernel_config_params->last_sem_idx_order_h[cell_count]);
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-1] Cell ID({}),SFN({}),Sub-frame({}),Slot({}),BFP({}),Beta({})",order_kernel_config_params->cell_id[cell_count],oran_ind.oframe_id_,oran_ind.osfid_,oran_ind.oslotid_,order_kernel_config_params->comp_meth[cell_count],order_kernel_config_params->beta[cell_count]);
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-2] Ordered Barrier flag addr({}),Done shared addr({}),SlotStart({}),ta4_min_ns({}),ta4_max_ns({}),slot_duration({})",(void*)order_kernel_config_params->barrier_flag,(void*)order_kernel_config_params->done_shared[cell_count],order_kernel_config_params->slot_start[cell_count],order_kernel_config_params->ta4_min_ns[cell_count],order_kernel_config_params->ta4_max_ns[cell_count],order_kernel_config_params->slot_duration[cell_count]);
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-3] pusch_eAxC_map({}),eAxC_pusch_num({}),pusch_buffer({}),puschNumPrb({}),pusch_prb_stride({}),pusch_ordered_prb_addr({})",(void*)order_kernel_config_params->pusch_eAxC_map[cell_count],order_kernel_config_params->pusch_eAxC_num[cell_count],(void*)order_kernel_config_params->pusch_buffer[cell_count],order_kernel_config_params->pusch_prb_x_slot[cell_count],order_kernel_config_params->pusch_prb_stride[cell_count],((void*)order_kernel_config_params->pusch_ordered_prbs[cell_count]));
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-4] prach_eAxC_map({}),eAxC_prach_num({}),prach_buffer({}),prachSectionId_o0({}),prachNumPrb({}),prach_prb_stride({}),prach_ordered_prb_addr({})",(void*)order_kernel_config_params->prach_eAxC_map[cell_count],order_kernel_config_params->prach_eAxC_num[cell_count],(void*)order_kernel_config_params->prach_buffer_0[cell_count],prachSectionId_o0,order_kernel_config_params->prach_prb_x_slot[cell_count],order_kernel_config_params->prach_prb_stride[cell_count],((void*)order_kernel_config_params->prach_ordered_prbs[cell_count]));
            */
                }
            }
            for(sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
                ACCESS_ONCE(*((uint32_t*)num_order_cells_sym_mask_arr->addrh()+sym_idx))=(uint32_t)num_order_cells_sym_mask[sym_idx];
            }            
            orderKernelConfigParamsCpuInitComms->pusch_prb_symbol_map_d=(uint32_t*)pusch_prb_symbol_map_gdr->addrd();
            orderKernelConfigParamsCpuInitComms->sym_ord_done_sig_arr=sym_ord_arr_addr;
            orderKernelConfigParamsCpuInitComms->sym_ord_done_mask_arr=(uint32_t*)sym_ord_done_mask_arr->addrd();
            orderKernelConfigParamsCpuInitComms->num_order_cells_sym_mask_arr=(uint32_t*)num_order_cells_sym_mask_arr->addrd();

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////// Start Order Kernel
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_idle, first_strm));
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_order, first_strm));
            if(launch_order_kernel_cpu_init_comms_single_subSlot(first_strm,
                                    orderKernelConfigParamsCpuInitComms->start_cuphy_d, // Unblock cuPHY
                                    orderKernelConfigParamsCpuInitComms->order_kernel_exit_cond_d, // Notify CPU order kernel completion
                                    orderKernelConfigParamsCpuInitComms->ready_list,
                                    orderKernelConfigParamsCpuInitComms->rx_queue_sync_list,
                                    orderKernelConfigParamsCpuInitComms->last_sem_idx_order_h,
                                    orderKernelConfigParamsCpuInitComms->sem_order_num,
                                    
                                    oran_ind.oframe_id_,
                                    oran_ind.osfid_,
                                    oran_ind.oslotid_,
                                    orderKernelConfigParamsCpuInitComms->comp_meth,
                                    orderKernelConfigParamsCpuInitComms->bit_width,
                                    DEFAULT_PRB_STRIDE,
                                    orderKernelConfigParamsCpuInitComms->beta,
                                    orderKernelConfigParamsCpuInitComms->barrier_flag, //((int*)order_barrier_flag->addr()),
                                    orderKernelConfigParamsCpuInitComms->done_shared,

                                    pdctx->getUlOrderTimeoutGPU(),
                                    pdctx->getUlOrderTimeoutFirstPktGPU(),

                                    /*Sub-slot Processing*/
                                    orderKernelConfigParamsCpuInitComms->sym_ord_done_sig_arr,
                                    orderKernelConfigParamsCpuInitComms->sym_ord_done_mask_arr,
                                    orderKernelConfigParamsCpuInitComms->pusch_prb_symbol_map_d,
                                    orderKernelConfigParamsCpuInitComms->num_order_cells_sym_mask_arr,

                                    /* RX time stats */
                                    orderKernelConfigParamsCpuInitComms->early_rx_packets,
                                    orderKernelConfigParamsCpuInitComms->on_time_rx_packets,
                                    orderKernelConfigParamsCpuInitComms->late_rx_packets,
                                    orderKernelConfigParamsCpuInitComms->next_slot_early_rx_packets,
                                    orderKernelConfigParamsCpuInitComms->next_slot_on_time_rx_packets,
                                    orderKernelConfigParamsCpuInitComms->next_slot_late_rx_packets,                                         
                                    orderKernelConfigParamsCpuInitComms->slot_start,
                                    orderKernelConfigParamsCpuInitComms->ta4_min_ns,
                                    orderKernelConfigParamsCpuInitComms->ta4_max_ns,
                                    orderKernelConfigParamsCpuInitComms->slot_duration,
                                    (uint8_t)pdctx->getUlRxPktTracingLevel(),
                                    orderKernelConfigParamsCpuInitComms->rx_packets_ts,
                                    orderKernelConfigParamsCpuInitComms->rx_packets_count,
                                    orderKernelConfigParamsCpuInitComms->rx_bytes_count,     
                                    orderKernelConfigParamsCpuInitComms->rx_packets_ts_earliest,
                                    orderKernelConfigParamsCpuInitComms->rx_packets_ts_latest,
                                    orderKernelConfigParamsCpuInitComms->next_slot_rx_packets_ts,
                                    orderKernelConfigParamsCpuInitComms->next_slot_rx_packets_count,
                                    orderKernelConfigParamsCpuInitComms->next_slot_rx_bytes_count,                                                
            
                                    /* PUSCH info */
                                    orderKernelConfigParamsCpuInitComms->pusch_eAxC_map,
                                    orderKernelConfigParamsCpuInitComms->pusch_eAxC_num,
                                    orderKernelConfigParamsCpuInitComms->pusch_buffer,
                                    orderKernelConfigParamsCpuInitComms->pusch_prb_x_slot,
                                    orderKernelConfigParamsCpuInitComms->pusch_prb_x_symbol,
                                    orderKernelConfigParamsCpuInitComms->pusch_prb_x_symbol_x_antenna, orderKernelConfigParamsCpuInitComms->pusch_prb_stride,
                                    orderKernelConfigParamsCpuInitComms->pusch_ordered_prbs,
            
                                    /* PRACH info */
                                    orderKernelConfigParamsCpuInitComms->prach_eAxC_map,
                                    orderKernelConfigParamsCpuInitComms->prach_eAxC_num,
                                    orderKernelConfigParamsCpuInitComms->prach_buffer_0, orderKernelConfigParamsCpuInitComms->prach_buffer_1, orderKernelConfigParamsCpuInitComms->prach_buffer_2, orderKernelConfigParamsCpuInitComms->prach_buffer_3,
                                    prachSectionId_o0, prachSectionId_o1, prachSectionId_o2, prachSectionId_o3,
                                    orderKernelConfigParamsCpuInitComms->prach_prb_x_slot,
                                    orderKernelConfigParamsCpuInitComms->prach_prb_x_symbol,
                                    orderKernelConfigParamsCpuInitComms->prach_prb_x_symbol_x_antenna, orderKernelConfigParamsCpuInitComms->prach_prb_stride,
                                    orderKernelConfigParamsCpuInitComms->prach_ordered_prbs,
                                    num_order_cells_nonSrsUl
                                ))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Run ORDER {} error", getId());
                return EINVAL;
            }                        
        }
        else
        {
            uint32_t fh_buf_slot_idx=pdctx->getConfigOkTbNumSlots();
            NVLOGI_FMT(TAG,"fh_buf_slot_idx:{} frameId:{} subframeId:{} slotId:{}",fh_buf_slot_idx,oran_ind.oframe_id_,oran_ind.osfid_,oran_ind.oslotid_);
            bool srsForRuTypeAllSym= false;

            memset(order_kernel_config_params, 0, sizeof(orderKernelConfigParams_t));
            
            ACCESS_ONCE(*((int*)order_barrier_flag->addrh())) = 0;
            order_kernel_config_params->barrier_flag=(int*)order_barrier_flag->addrd();
        
            first_strm=pdctx->getUlOrderStreamPd();
            
            NVLOGD_FMT(TAG, "Cell mask for non SRS UL channels: {}", nonSrsUlCellMask);

            cell_count = 0;

            //Loop to populate Order kernel config params for all the needed cells
            for(cell_idx=0;cell_idx<num_order_cells;cell_idx++)
            {
                if(nonSrsUlCellMask & (1 << cell_idx))
                {
                    cell_ptr = pdctx->getCellById(cell_id[cell_idx]);
                    if(cell_ptr == nullptr)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " No cell for this ORDER {}", getId());
                        return EINVAL;
                    }
                    if(
                        (buf_st_1[cell_idx] == nullptr || puschNumPrb[cell_idx] == 0) &&
                        ( (buf_st_3_o0[cell_idx] == nullptr && buf_st_3_o1[cell_idx] == nullptr && buf_st_3_o2[cell_idx] == nullptr && buf_st_3_o3[cell_idx] == nullptr) || prachNumPrb[cell_idx] == 0)
                    )
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Nothing to reorder ORDER {}", getId());
                        return EINVAL;
                    }

                    if(
                        (buf_st_1[cell_idx] != nullptr && puschNumPrb[cell_idx] == 0) ||
                        ((buf_st_3_o0[cell_idx] != nullptr && buf_st_3_o1[cell_idx] != nullptr && buf_st_3_o2[cell_idx] != nullptr && buf_st_3_o3[cell_idx] != nullptr) && prachNumPrb[cell_idx] == 0)
                    )
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Nothing to reorder ORDER {}", getId());
                        return EINVAL;
                    }
                    
                    pusch_prbs_x_slot[cell_count] = puschNumPrb[cell_idx];
                    prach_prbs_x_slot[cell_count] = prachNumPrb[cell_idx];

                    order_kernel_config_params->rxq_info_gpu[cell_count]=cell_ptr->docaGetRxqInfo()->eth_rxq_gpu;
                    order_kernel_config_params->sem_gpu[cell_count]=cell_ptr->docaGetRxqInfo()->sem_gpu;
                    order_kernel_config_params->sem_gpu_aerial_fh[cell_count]=cell_ptr->docaGetRxqInfo()->sem_gpu_aerial_fh;
                    order_kernel_config_params->sem_order_num[cell_count]=cell_ptr->docaGetRxqInfo()->nitems;

                    order_kernel_config_params->cell_id[cell_count]=cell_ptr->getPhyId();
                    order_kernel_config_params->ru_type[cell_count]=(int)cell_ptr->getRUType();
                    order_kernel_config_params->cell_health[cell_count]=cell_ptr->isHealthy();

                    order_kernel_config_params->start_cuphy_d[cell_count]= (uint32_t*)start_cuphy_gdr[cell_idx]->addrd();
                    order_kernel_config_params->order_kernel_exit_cond_d[cell_count]=(uint32_t*)order_kernel_exit_cond_gdr[cell_idx]->addrd();

                    order_kernel_config_params->last_sem_idx_rx_h[cell_count]=cell_ptr->getLastRxItem();
                    order_kernel_config_params->last_sem_idx_order_h[cell_count]=cell_ptr->getLastOrderedItem();

                    order_kernel_config_params->comp_meth[cell_count]=(int)cell_ptr->getULCompMeth();
                    order_kernel_config_params->bit_width[cell_count]=(int)cell_ptr->getULBitWidth();
                    if((order_kernel_config_params->bit_width[cell_count] != BFP_NO_COMPRESSION &&
                        order_kernel_config_params->bit_width[cell_count] != BFP_COMPRESSION_14_BITS &&
                        order_kernel_config_params->bit_width[cell_count] != BFP_COMPRESSION_9_BITS &&
                        order_kernel_config_params->comp_meth[cell_count] == (int)aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT) ||
                    (order_kernel_config_params->bit_width[cell_count] != 16 &&
                        order_kernel_config_params->comp_meth[cell_count] == (int)aerial_fh::UserDataCompressionMethod::NO_COMPRESSION))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Run ORDER {} error, cell id {}, unsupported comp_meth {}, bitwidth {}\n", getId(), cell_idx, order_kernel_config_params->comp_meth[cell_count], order_kernel_config_params->bit_width[cell_count]);
                        return EINVAL;
                    }

                    order_kernel_config_params->beta[cell_count]=(float)cell_ptr->getBetaUlPowerScaling();

                    order_kernel_config_params->done_shared[cell_count]=(uint8_t*)done_shared[cell_idx]->addr();

                    order_kernel_config_params->early_rx_packets[cell_count]=(uint32_t*)early_rx_packets[cell_idx]->addrd();
                    order_kernel_config_params->on_time_rx_packets[cell_count]=(uint32_t*)on_time_rx_packets[cell_idx]->addrd();
                    order_kernel_config_params->late_rx_packets[cell_count]=(uint32_t*)late_rx_packets[cell_idx]->addrd();
                    order_kernel_config_params->next_slot_early_rx_packets[cell_count]=cell_ptr->getNextSlotEarlyRxPackets();
                    order_kernel_config_params->next_slot_on_time_rx_packets[cell_count]=cell_ptr->getNextSlotOnTimeRxPackets();
                    order_kernel_config_params->next_slot_late_rx_packets[cell_count]=cell_ptr->getNextSlotLateRxPackets();
                    order_kernel_config_params->next_slot_num_prb_ch1[cell_count]=cell_ptr->getNextSlotNumPrbCh1();
                    order_kernel_config_params->next_slot_num_prb_ch2[cell_count]=cell_ptr->getNextSlotNumPrbCh2();
                    order_kernel_config_params->rx_packets_ts[cell_count]=(uint64_t*)rx_packets_ts[cell_idx]->addrd();
                    order_kernel_config_params->rx_packets_ts_earliest[cell_count]=(uint64_t*)rx_packets_ts_earliest[cell_idx]->addrd();
                    order_kernel_config_params->rx_packets_ts_latest[cell_count]=(uint64_t*)rx_packets_ts_latest[cell_idx]->addrd();
                    order_kernel_config_params->next_slot_rx_packets_ts[cell_count]=(uint64_t*)cell_ptr->getNextSlotRxPacketTs();
                    order_kernel_config_params->next_slot_rx_packets_count[cell_count]=(uint32_t*)cell_ptr->getNextSlotRxPacketCount();  
                    order_kernel_config_params->next_slot_rx_bytes_count[cell_count]=(uint32_t*)cell_ptr->getNextSlotRxByteCount();            

                    if(cell_ptr->getRUType() == SINGLE_SECT_MODE)
                    {
                        order_kernel_config_params->srs_eAxC_map[cell_count]=(uint16_t*)eAxC_srs_map_gdr[cell_idx]->addrd();
                        order_kernel_config_params->srs_eAxC_num[cell_count]=eAxC_srs_num[cell_idx];
                        order_kernel_config_params->srs_buffer[cell_count]=buf_srs[cell_idx];
                        order_kernel_config_params->srs_prb_x_slot[cell_count]= srsNumPrb[cell_idx];
                        if(srsNumPrb[cell_idx] > 0) srsForRuTypeAllSym=true;
                        order_kernel_config_params->srs_prb_stride[cell_count]=cell_ptr->getSrsPrbStride();
                        order_kernel_config_params->srs_ordered_prbs[cell_count]=(uint32_t*)ordered_prbs_srs[cell_idx]->addrd();
                        order_kernel_config_params->srs_start_sym[cell_count]=srs_start_symbol[cell_idx];
                        order_kernel_config_params->order_kernel_last_timeout_error_time[cell_count]=(uint64_t*)cell_ptr->getOrderkernelSrsLastTimeoutErrorTimeItem();
                    }

                    if(pdctx->enableOKTb())
                    {
                        order_kernel_config_params->fh_buf_ok_tb_slot[cell_count]=pdctx->getFhBufOkTb(cell_idx)+ (fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB)*(MAX_PKTS_PER_SLOT_OK_TB*pdctx->getConfigOkTbMaxPacketSize());
                        order_kernel_config_params->fh_buf_ok_tb_next_slot[cell_count]=pdctx->getFhBufOkTb(cell_idx)+ ((fh_buf_slot_idx+1)%MAX_UL_SLOTS_OK_TB)*(MAX_PKTS_PER_SLOT_OK_TB*pdctx->getConfigOkTbMaxPacketSize());
                        pdctx->getOkTbConfig(cell_idx)->num_rx_packets[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=0;
                        pdctx->getOkTbConfig(cell_idx)->num_pusch_prbs[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=puschNumPrb[cell_idx];
                        pdctx->getOkTbConfig(cell_idx)->num_prach_prbs[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=prachNumPrb[cell_idx];
                        pdctx->getOkTbConfig(cell_idx)->frameId[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=oran_ind.oframe_id_;
                        pdctx->getOkTbConfig(cell_idx)->subframeId[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=oran_ind.osfid_;
                        pdctx->getOkTbConfig(cell_idx)->slotId[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB]=oran_ind.oslotid_;
                        pdctx->getOkTbConfig(cell_idx)->pusch_eAxC_num=eAxC_num[cell_idx];
                        pdctx->getOkTbConfig(cell_idx)->prach_eAxC_num=eAxC_prach_num[cell_idx];
                        for(int tmp = 0; tmp < eAxC_num[cell_idx]; tmp++)
                        {
                            pdctx->getOkTbConfig(cell_idx)->pusch_eAxC_map[tmp]=((uint16_t*)(eAxC_map_gdr[cell_idx]->addrh()))[tmp];
                        }
                        for(int tmp = 0; tmp < eAxC_prach_num[cell_idx]; tmp++)
                        {
                            pdctx->getOkTbConfig(cell_idx)->prach_eAxC_map[tmp]=((uint16_t*)(eAxC_prach_map_gdr[cell_idx]->addrh()))[tmp];
                        }                                                    
                        pdctx->getOkTbConfig(cell_idx)->cell_id=cell_ptr->getPhyId();
                        order_kernel_config_params->rx_packets_count[cell_count]=(uint32_t*)&pdctx->getOkTbConfig(cell_idx)->num_rx_packets[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB];
                        //TODO order_kernel_config_params->rx_bytes_count[cell_count]=(uint32_t*)rx_bytes_count[cell_idx]->addrd();
                    }
                    else
                    {
                        order_kernel_config_params->rx_packets_count[cell_count]=(uint32_t*)rx_packets_count[cell_idx]->addrd();
                        order_kernel_config_params->rx_bytes_count[cell_count]=(uint32_t*)rx_bytes_count[cell_idx]->addrd();
                        order_kernel_config_params->rx_packets_dropped_count[cell_count]=(uint32_t*)rx_packets_dropped_count[cell_idx]->addrd();
                    }          

                    order_kernel_config_params->slot_start[cell_count]=slot_start[cell_idx].count();
                    order_kernel_config_params->ta4_min_ns[cell_count]=ta4_min_ns[cell_idx];
                    order_kernel_config_params->ta4_max_ns[cell_count]=ta4_max_ns[cell_idx];
                    order_kernel_config_params->slot_duration[cell_count]=cell_ptr->getTtiNsFromMu(cell_ptr->getMu());

                    order_kernel_config_params->pusch_eAxC_map[cell_count]=(uint16_t*)eAxC_map_gdr[cell_idx]->addrd();

                    order_kernel_config_params->pusch_eAxC_num[cell_count]=eAxC_num[cell_idx];
                    order_kernel_config_params->pusch_buffer[cell_count]=buf_st_1[cell_idx];
                    order_kernel_config_params->pcap_buffer[cell_count]=buf_pcap_capture[cell_idx];
                    order_kernel_config_params->pcap_buffer_ts[cell_count]=buf_pcap_capture_ts[cell_idx];
                    order_kernel_config_params->pcap_buffer_index[cell_count]=cell_ptr->getULPcapCaptureBufferIndex();
                    order_kernel_config_params->pusch_prb_x_slot[cell_count]=pusch_prbs_x_slot[cell_count];
                    order_kernel_config_params->pusch_prb_x_symbol[cell_count]=pusch_prbs_x_slot[cell_count];
                    order_kernel_config_params->pusch_prb_x_symbol_x_antenna[cell_count]=pusch_prbs_x_slot[cell_count];
                    order_kernel_config_params->pusch_prb_stride[cell_count]=cell_ptr->getPuschPrbStride();

                    order_kernel_config_params->pusch_ordered_prbs[cell_count]=(uint32_t*)ordered_prbs_pusch[cell_idx]->addrd();

                    order_kernel_config_params->prach_eAxC_map[cell_count]=(uint16_t*)eAxC_prach_map_gdr[cell_idx]->addrd();

                    order_kernel_config_params->prach_eAxC_num[cell_count]=eAxC_prach_num[cell_idx];
                    order_kernel_config_params->prach_buffer_0[cell_count]=buf_st_3_o0[cell_idx];
                    order_kernel_config_params->prach_buffer_1[cell_count]=buf_st_3_o1[cell_idx];
                    order_kernel_config_params->prach_buffer_2[cell_count]=buf_st_3_o2[cell_idx];
                    order_kernel_config_params->prach_buffer_3[cell_count]=buf_st_3_o3[cell_idx];
                    order_kernel_config_params->prach_prb_x_slot[cell_count]=prach_prbs_x_slot[cell_count];
                    order_kernel_config_params->prach_prb_x_symbol[cell_count]=prach_prbs_x_slot[cell_count];
                    order_kernel_config_params->prach_prb_x_symbol_x_antenna[cell_count]=prachNumPrb[cell_count];
                    order_kernel_config_params->prach_prb_stride[cell_count]=cell_ptr->getPrachPrbStride();

                    order_kernel_config_params->prach_ordered_prbs[cell_count]=(uint32_t*)ordered_prbs_prach[cell_idx]->addrd();
                    order_kernel_config_params->order_kernel_last_timeout_error_time[cell_count]=(uint64_t*)cell_ptr->getOrderkernelLastTimeoutErrorTimeItem();

                    for(sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
                        ACCESS_ONCE(*((uint32_t*)pusch_prb_symbol_map_gdr->addrh()+(cell_count*ORAN_PUSCH_SYMBOLS_X_SLOT)+sym_idx))=(uint32_t)pusch_prb_symbol_map[(cell_idx*ORAN_PUSCH_SYMBOLS_X_SLOT)+sym_idx];
                        if(pdctx->enableOKTb())
                            pdctx->getOkTbConfig(cell_idx)->pusch_prb_symbol_map[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB][sym_idx]=(uint32_t)pusch_prb_symbol_map[(cell_idx*ORAN_PUSCH_SYMBOLS_X_SLOT)+sym_idx];                            
                    }
                    DOCA_GPUNETIO_VOLATILE(((uint32_t*)(order_kernel_exit_cond_gdr[cell_count]->addrh()))[0]) = ORDER_KERNEL_RUNNING;

                    num_order_cells_nonSrsUl++;
                    cell_count++;
        
                    /* 
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-0] DOCA GPU RxQ Info({}),DOCA Sem gpu({}), DOCA Sem Num({}),Last Sem Item({}),Last Ordered Item({})",(void*)order_kernel_config_params->rxq_info_gpu[cell_count],order_kernel_config_params->sem_gpu[cell_count],order_kernel_config_params->sem_order_num[cell_count],(void*)order_kernel_config_params->last_sem_idx_rx_h[cell_count],(void*)order_kernel_config_params->last_sem_idx_order_h[cell_count]);
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-1] Cell ID({}),SFN({}),Sub-frame({}),Slot({}),BFP({}),Beta({})",order_kernel_config_params->cell_id[cell_count],oran_ind.oframe_id_,oran_ind.osfid_,oran_ind.oslotid_,order_kernel_config_params->comp_meth[cell_count],order_kernel_config_params->beta[cell_count]);
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-2] Ordered Barrier flag addr({}),Done shared addr({}),SlotStart({}),ta4_min_ns({}),ta4_max_ns({}),slot_duration({})",(void*)order_kernel_config_params->barrier_flag,(void*)order_kernel_config_params->done_shared[cell_count],order_kernel_config_params->slot_start[cell_count],order_kernel_config_params->ta4_min_ns[cell_count],order_kernel_config_params->ta4_max_ns[cell_count],order_kernel_config_params->slot_duration[cell_count]);
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-3] pusch_eAxC_map({}),eAxC_pusch_num({}),pusch_buffer({}),puschNumPrb({}),pusch_prb_stride({}),pusch_ordered_prb_addr({})",(void*)order_kernel_config_params->pusch_eAxC_map[cell_count],order_kernel_config_params->pusch_eAxC_num[cell_count],(void*)order_kernel_config_params->pusch_buffer[cell_count],order_kernel_config_params->pusch_prb_x_slot[cell_count],order_kernel_config_params->pusch_prb_stride[cell_count],((void*)order_kernel_config_params->pusch_ordered_prbs[cell_count]));
                    NVLOGI_FMT(TAG,"[Order Kernel config Params Before launch-4] prach_eAxC_map({}),eAxC_prach_num({}),prach_buffer({}),prachSectionId_o0({}),prachNumPrb({}),prach_prb_stride({}),prach_ordered_prb_addr({})",(void*)order_kernel_config_params->prach_eAxC_map[cell_count],order_kernel_config_params->prach_eAxC_num[cell_count],(void*)order_kernel_config_params->prach_buffer_0[cell_count],prachSectionId_o0,order_kernel_config_params->prach_prb_x_slot[cell_count],order_kernel_config_params->prach_prb_stride[cell_count],((void*)order_kernel_config_params->prach_ordered_prbs[cell_count]));
                    */
                }
            }

            for(sym_idx=0;sym_idx<ORAN_PUSCH_SYMBOLS_X_SLOT;sym_idx++){
                    ACCESS_ONCE(*((uint32_t*)num_order_cells_sym_mask_arr->addrh()+sym_idx))=(uint32_t)num_order_cells_sym_mask[sym_idx];
                    if(pdctx->enableOKTb())
                        pdctx->getOkTbConfig(0)->num_order_cells_sym_mask[fh_buf_slot_idx%MAX_UL_SLOTS_OK_TB][sym_idx]=(uint32_t)num_order_cells_sym_mask[sym_idx];
            }            
            order_kernel_config_params->pusch_prb_symbol_map_d=(uint32_t*)pusch_prb_symbol_map_gdr->addrd();
            order_kernel_config_params->sym_ord_done_sig_arr=sym_ord_arr_addr;
            order_kernel_config_params->sym_ord_done_mask_arr=(uint32_t*)sym_ord_done_mask_arr->addrd();
            order_kernel_config_params->num_order_cells_sym_mask_arr=(uint32_t*)num_order_cells_sym_mask_arr->addrd();
            if(pdctx->enableOKTb())
            {
                pdctx->setConfigOkTbNumSlots((fh_buf_slot_idx+1));
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////// Start Order Kernel
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_idle, first_strm));
            CUDA_CHECK_PHYDRIVER(cudaEventRecord(start_order, first_strm));

            if(pdctx->enableOKTb())
            {
                    if(launch_receive_kernel_for_test_bench(first_strm,
                                            order_kernel_config_params->rxq_info_gpu,
                                            order_kernel_config_params->sem_gpu,
                                            order_kernel_config_params->sem_order_num,

                                            order_kernel_config_params->cell_id,
                                            order_kernel_config_params->order_kernel_exit_cond_d, // Notify CPU order kernel completion
                                            order_kernel_config_params->last_sem_idx_rx_h,
                                            order_kernel_config_params->bit_width,

                                            pdctx->getUlOrderTimeoutGPU(),
                                            pdctx->getUlOrderTimeoutFirstPktGPU(),
                                            pdctx->getUlOrderMaxRxPkts(),
                                            pdctx->getConfigOkTbMaxPacketSize(),

                                            oran_ind.oframe_id_,
                                            oran_ind.osfid_,
                                            oran_ind.oslotid_,

                                            order_kernel_config_params->rx_packets_count,
                                            order_kernel_config_params->next_slot_rx_packets_count,
                                            order_kernel_config_params->next_slot_num_prb_ch1,
                                            order_kernel_config_params->next_slot_num_prb_ch2,

                                            /*FH buffer*/
                                            order_kernel_config_params->fh_buf_ok_tb_slot,
                                            order_kernel_config_params->fh_buf_ok_tb_next_slot,

                                            /* PUSCH/PRACH info */
                                            order_kernel_config_params->pusch_prb_x_slot,
                                            order_kernel_config_params->prach_prb_x_slot,
                                            prachSectionId_o0, prachSectionId_o1, prachSectionId_o2, prachSectionId_o3,
                                            nullptr,
                                            num_order_cells_nonSrsUl
                                        ))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Run ORDER {} error", getId());
                        return EINVAL;
                    }    
            }
            else
            {
                    int srs_mode = srsForRuTypeAllSym ? 3 : 0;
                    if(launch_order_kernel_doca_single_subSlot(first_strm,
                                            order_kernel_config_params->rxq_info_gpu,
                                            order_kernel_config_params->sem_gpu,
                                            order_kernel_config_params->sem_gpu_aerial_fh,
                                            order_kernel_config_params->sem_order_num,

                                            order_kernel_config_params->cell_id,
                                            order_kernel_config_params->ru_type,
                                            order_kernel_config_params->cell_health,

                                            order_kernel_config_params->start_cuphy_d, // Unblock cuPHY
                                            order_kernel_config_params->order_kernel_exit_cond_d, // Notify CPU order kernel completion

                                            order_kernel_config_params->last_sem_idx_rx_h,
                                            order_kernel_config_params->last_sem_idx_order_h,

                                            pdctx->getUlOrderKernelMode(),
                                            pdctx->getUlOrderTimeoutGPU(),
                                            pdctx->getUlOrderTimeoutFirstPktGPU(),
                                            pdctx->getUlOrderTimeoutLogInterval(),
                                            pdctx->getUlOrderTimeoutGPULogEnable(),
                                            pdctx->getUlOrderMaxRxPkts(),
                                            pdctx->getUlOrderRxPktsTimeout(),
                                            pdctx->gpuCommEnabledViaCpu(),

                                            oran_ind.oframe_id_,
                                            oran_ind.osfid_,
                                            oran_ind.oslotid_,
                                            order_kernel_config_params->comp_meth,
                                            order_kernel_config_params->bit_width,
                                            DEFAULT_PRB_STRIDE,
                                            order_kernel_config_params->beta,
                                            order_kernel_config_params->barrier_flag, //((int*)order_barrier_flag->addr()),
                                            order_kernel_config_params->done_shared,

                                            /* RX time stats */
                                            order_kernel_config_params->early_rx_packets,
                                            order_kernel_config_params->on_time_rx_packets,
                                            order_kernel_config_params->late_rx_packets,
                                            order_kernel_config_params->next_slot_early_rx_packets,
                                            order_kernel_config_params->next_slot_on_time_rx_packets,
                                            order_kernel_config_params->next_slot_late_rx_packets,                                         
                                            order_kernel_config_params->slot_start,
                                            order_kernel_config_params->ta4_min_ns,
                                            order_kernel_config_params->ta4_max_ns,
                                            order_kernel_config_params->slot_duration,
                                            order_kernel_config_params->order_kernel_last_timeout_error_time,
                                            (uint8_t)pdctx->getUlRxPktTracingLevel(),
                                            order_kernel_config_params->rx_packets_ts,
                                            order_kernel_config_params->rx_packets_count,     
                                            order_kernel_config_params->rx_bytes_count,     
                                            order_kernel_config_params->rx_packets_ts_earliest,
                                            order_kernel_config_params->rx_packets_ts_latest,
                                            order_kernel_config_params->next_slot_rx_packets_ts,
                                            order_kernel_config_params->next_slot_rx_packets_count,
                                            order_kernel_config_params->next_slot_rx_bytes_count,
                                            order_kernel_config_params->rx_packets_dropped_count,
                                            order_kernel_config_params->next_slot_num_prb_ch1,
                                            order_kernel_config_params->next_slot_num_prb_ch2,
                                            
                                            /*Sub-slot Processing*/
                                            order_kernel_config_params->sym_ord_done_sig_arr,
                                            order_kernel_config_params->sym_ord_done_mask_arr,
                                            order_kernel_config_params->pusch_prb_symbol_map_d,
                                            order_kernel_config_params->num_order_cells_sym_mask_arr,
                                            pusch_prb_non_zero,

                                            /* PUSCH info */
                                            order_kernel_config_params->pusch_eAxC_map,
                                            order_kernel_config_params->pusch_eAxC_num,
                                            order_kernel_config_params->pusch_buffer,
                                            order_kernel_config_params->pusch_prb_x_slot,
                                            order_kernel_config_params->pusch_prb_x_symbol,
                                            order_kernel_config_params->pusch_prb_x_symbol_x_antenna, order_kernel_config_params->pusch_prb_stride,
                                            order_kernel_config_params->pusch_ordered_prbs,

                                            /* PRACH info */
                                            order_kernel_config_params->prach_eAxC_map,
                                            order_kernel_config_params->prach_eAxC_num,
                                            order_kernel_config_params->prach_buffer_0, order_kernel_config_params->prach_buffer_1, order_kernel_config_params->prach_buffer_2, order_kernel_config_params->prach_buffer_3,
                                            prachSectionId_o0, prachSectionId_o1, prachSectionId_o2, prachSectionId_o3,
                                            order_kernel_config_params->prach_prb_x_slot,
                                            order_kernel_config_params->prach_prb_x_symbol,
                                            order_kernel_config_params->prach_prb_x_symbol_x_antenna, order_kernel_config_params->prach_prb_stride,
                                            order_kernel_config_params->prach_ordered_prbs,

                                            /* SRS info (not used for non-SRS UL launch, but used for combined OK)*/
                                            order_kernel_config_params->srs_eAxC_map,
                                            order_kernel_config_params->srs_eAxC_num,
                                            order_kernel_config_params->srs_buffer,
                                            order_kernel_config_params->srs_prb_x_slot,
                                            order_kernel_config_params->srs_prb_stride,
                                            order_kernel_config_params->srs_ordered_prbs,
                                            order_kernel_config_params->srs_start_sym,

                                            num_order_cells_nonSrsUl,
                                            /* PCAP capture info */
                                            order_kernel_config_params->pcap_buffer,
                                            order_kernel_config_params->pcap_buffer_ts,
                                            order_kernel_config_params->pcap_buffer_index,
                                            (uint8_t)pdctx->get_ul_pcap_capture_enable(),
                                            pcap_capture_cell_bitmask,
                                            (uint16_t)pdctx->get_ul_pcap_capture_mtu(),
                                            srs_mode
                                        ))
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, " Run ORDER {} error", getId());
                        return EINVAL;
                    }            
            }            
        }                    
            
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(end_order, first_strm));
        // Notify CPU RX thread (can be replaced by cudaEvent?)
        launch_kernel_write(first_strm, (uint32_t*)start_cuphy_cpu_h->addr(), 1);
        setOrderLaunchedStatus(true);
	if (pdctx->get_ru_type_for_srs_proc() == SINGLE_SECT_MODE) {
            setOrderLaunchedStatusSrs(true);
	}
    }

    return 0;
}

uint32_t* OrderEntity::getWaitSingleOrderGpuFlag(int cell_idx)
{
    return ((uint32_t*)(start_cuphy_gdr[cell_idx]->addrd()));
}


bool   OrderEntity::getOrderLaunchedStatus() 
{
    return order_launched;
}
void  OrderEntity::setOrderLaunchedStatus(bool val)
{
    order_launched=val;
}

bool   OrderEntity::getOrderLaunchedStatusSrs() 
{
    return order_launched_srs;
}
void  OrderEntity::setOrderLaunchedStatusSrs(bool val)
{
    order_launched_srs=val;
}

int OrderEntity::checkOrderCPU(bool isSrs)
{
    if(isSrs)
    {
        if(ACCESS_ONCE(*((uint32_t*)start_cuphy_srs_cpu_h->addr())) == 1)
            return 1;    
    }
    else
    {
        if(ACCESS_ONCE(*((uint32_t*)start_cuphy_cpu_h->addr())) == 1)
            return 1;    
    }
    return 0;
}


int OrderEntity::waitOrder(int wait_ns)
{
    t_ns          threshold_t, start_t = Time::nowNs();
    threshold_t = t_ns(wait_ns);

    while(ACCESS_ONCE(*((uint32_t*)start_cuphy_cpu_h->addr())) == 0)
    {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: Order kernel Object {} waiting for more than {} ns",
                getId(), wait_ns);
            return -1;
        }
    }

    return 0;
}

int OrderEntity::waitOrderLaunched(int wait_ns)
{
    t_ns          threshold_t, start_t = Time::nowNs();
    threshold_t = t_ns(wait_ns);

    while(!order_launched)
    {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: Order kernel Object {} waiting for more than {} ns for Order kernel launch",
                getId(), wait_ns);
            return -1;
        }
    }

    return 0;
}

int OrderEntity::waitOrderLaunchedSrs(int wait_ns)
{
    t_ns          threshold_t, start_t = Time::nowNs();
    threshold_t = t_ns(wait_ns);

    while(!order_launched_srs)
    {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: Order kernel Object {} waiting for more than {} ns for SRS Order kernel launch",
                getId(), wait_ns);
            return -1;
        }
    }
    return 0;
}

void OrderEntity::setOrderCPU()
{
    ACCESS_ONCE(*((uint32_t*)start_cuphy_cpu_h->addr())) = 1;
}

float OrderEntity::getGPUIdleTime() {
    return 1000.0f * GetCudaEventElapsedTime(start_idle, start_order, __func__, getId());
}

float OrderEntity::getGPUOrderTime() {
    return 1000.0f * GetCudaEventElapsedTime(start_order, end_order, __func__, getId());
}

float OrderEntity::getGPUIdleTimeSrs() {
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx->get_ru_type_for_srs_proc() != SINGLE_SECT_MODE)
    {
        return 1000.0f * GetCudaEventElapsedTime(start_idle_srs, start_order_srs, __func__, getId());
    }
    return 0.0f;
}

float OrderEntity::getGPUOrderTimeSrs() {
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx->get_ru_type_for_srs_proc() != SINGLE_SECT_MODE)
    {
        return 1000.0f * GetCudaEventElapsedTime(start_order_srs, end_order_srs, __func__, getId());
    }
    return 0.0f;
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
uint32_t  OrderEntity::getRxPacketCount(int cell_idx,int sym_idx){
    return *((uint32_t*)rx_packets_count[cell_idx]->addrh()+sym_idx);
}
uint32_t  OrderEntity::getRxByteCount(int cell_idx,int sym_idx){
    return *((uint32_t*)rx_bytes_count[cell_idx]->addrh()+sym_idx);
}
uint32_t OrderEntity::getRxPacketsDroppedCount(int cell_idx){
    return *((uint32_t*)rx_packets_dropped_count[cell_idx]->addrh());
}
uint64_t  OrderEntity::getRxPacketTs(int cell_idx,int sym_idx,int pkt_idx){
    return *((uint64_t*)rx_packets_ts[cell_idx]->addrh()+ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*sym_idx+pkt_idx);
}
uint64_t  OrderEntity::getRxPacketTsEarliest(int cell_idx,int sym_idx){
    return *((uint64_t*)rx_packets_ts_earliest[cell_idx]->addrh()+sym_idx);
}
uint64_t  OrderEntity::getRxPacketTsLatest(int cell_idx,int sym_idx){
    return *((uint64_t*)rx_packets_ts_latest[cell_idx]->addrh()+sym_idx);
}

uint32_t OrderEntity::getOrderExitCondition(int cell_idx){
    return *((uint32_t*)order_kernel_exit_cond_gdr[cell_idx]->addrh());
}

uint32_t OrderEntity::getOrderSrsExitCondition(int cell_idx){
    return *((uint32_t*)order_kernel_srs_exit_cond_gdr[cell_idx]->addrh());
}

uint32_t OrderEntity::getEarlyRxPacketsSRS(int cell_idx) {
    return *((uint32_t*)early_rx_packets_srs[cell_idx]->addrh());
}
uint32_t OrderEntity::getOnTimeRxPacketsSRS(int cell_idx) {
    return *((uint32_t*)on_time_rx_packets_srs[cell_idx]->addrh());
}
uint32_t OrderEntity::getLateRxPacketsSRS(int cell_idx) {
    return *((uint32_t*)late_rx_packets_srs[cell_idx]->addrh());
}
uint32_t  OrderEntity::getRxPacketCountSRS(int cell_idx){
    return *((uint32_t*)rx_packets_count_srs[cell_idx]->addrh());
}
uint32_t  OrderEntity::getRxByteCountSRS(int cell_idx){
    return *((uint32_t*)rx_bytes_count_srs[cell_idx]->addrh());
}
uint32_t OrderEntity::getRxPacketsDroppedCountSRS(int cell_idx){
    return *((uint32_t*)rx_packets_dropped_count_srs[cell_idx]->addrh());
}

uint32_t  OrderEntity::getRxPacketCountPerSymSRS(int cell_idx,int sym_idx){
    return *((uint32_t*)rx_packets_count_per_sym_srs[cell_idx]->addrh()+sym_idx);
}
uint64_t  OrderEntity::getRxPacketTsSRS(int cell_idx,int sym_idx,int pkt_idx){
    return *((uint64_t*)rx_packets_ts_srs[cell_idx]->addrh()+ORDER_KERNEL_MAX_PKTS_PER_OFDM_SYM*sym_idx+pkt_idx);
}
uint64_t  OrderEntity::getRxPacketTsEarliestSRS(int cell_idx,int sym_idx){
    return *((uint64_t*)rx_packets_ts_earliest_srs[cell_idx]->addrh()+sym_idx);
}
uint64_t  OrderEntity::getRxPacketTsLatestSRS(int cell_idx,int sym_idx){
    return *((uint64_t*)rx_packets_ts_latest_srs[cell_idx]->addrh()+sym_idx);
}

void OrderEntity::enableOrder(int cell_idx,int start_type)
{
    ACCESS_ONCE(*((uint32_t*)order_kernel_exit_cond_gdr[cell_idx]->addrh()))      = start_type;
}
