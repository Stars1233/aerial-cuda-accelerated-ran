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

#include "worker.hpp"

#include <algorithm>

#include "fh_generator.hpp"
#include "utils.hpp"
#include "gpudevice.hpp"
#include "yaml_parser.hpp"
#include "doca_utils.hpp"

namespace fh_gen
{

void fronthaul_generator_ul_rx_worker(Worker* worker)
{
    uint32_t cpu;
    int ret;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        THROW(StringBuilder() << "getcpu failed for " << __FUNCTION__);
    }

    NVLOGC_FMT(TAG,"Start UL RX worker on CPU {}", cpu);
    char threadname[30];
    sprintf(threadname, "%s", "ULRX");
    SET_THREAD_NAME(threadname);

    PrimaryCtxGuard ctx_guard(0);

    auto& context = worker->get_context();
    auto& nic = context.nic;
    cudaStream_t stream;
    CUresult cuResult = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    if (cuResult != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(cuResult, &errStr);
        NVLOGE_FMT(TAG, AERIAL_CUDA_KERNEL_EVENT, "[{}:{}] cu failed with {} ", __FILE__, __LINE__, errStr ? errStr : "unknown");
        exit(EXIT_FAILURE);
    }
    ACCESS_ONCE(((uint32_t*)context.ul_rx_worker_context.exit_flag[0]->addrh())[0]) = 0;

    auto& time_anchor = context.time_anchor;
    auto t0 = time_anchor;
    auto pattern_slot_id = 0;
    auto test_slot_count = 0;
    auto order_entity_id = 0;
    auto active_order_entity_id = 0;
    auto& order_entities = context.ul_rx_worker_context.order_entities;
    wait_ns(t0 - 3 * context.slot_duration);
    //T0, expectPRBs per cell/slot, timing windows
    while(!worker->exit_signal() && (test_slot_count < context.test_slots || active_order_entity_id != order_entity_id))
    {
        auto oran_slot_number   = context.oran_slot_iterator.get_next();
        if(test_slot_count < context.test_slots)
        {
            if(context.ul_rx_worker_context.has_expected_rx_prbs[pattern_slot_id])
            {
                auto oentity = order_entities[order_entity_id];
                auto SFN = oran_slot_number.SFN;
                auto SFN_slot = oran_slot_number.subframe_id * ORAN_MAX_SLOT_ID + oran_slot_number.slot_id;
                if(oentity->reserve() == 0)
                {
                    auto& params = oentity->order_kernel_config_params;
                    params->frame_id = oran_slot_number.frame_id;
                    params->subframe_id = oran_slot_number.subframe_id;
                    params->slot_id = oran_slot_number.slot_id;
                    params->SFN = oran_slot_number.SFN;
                    params->slot_t0 = t0;
                    params->num_cells = context.ul_rx_worker_context.peers.size();
                    params->exit_flag_d = (uint32_t*)context.ul_rx_worker_context.exit_flag[0]->addrd();
                    params->slot_duration = context.slot_duration;
                    for(int i = 0; i < params->num_cells; ++i)
                    {
                        params->prb_x_slot[i] = context.ul_rx_worker_context.expected_rx_prbs_h[pattern_slot_id][i];
                    }
                    oentity->runOrder(stream);
                    order_entity_id = (order_entity_id + 1) % kOrderEntityNum;
                }
                else
                {
                    NVLOGC_FMT(TAG, "[FHGEN] SFN {}.{} ORDER {} reserve failed", SFN, SFN_slot, order_entity_id);
                }
            }
        }

        pattern_slot_id = (pattern_slot_id + 1) % context.slot_count;
        ++test_slot_count;
        t0 += context.slot_duration;

        //check next order kernel for completion
        while(now_ns() < t0 - 3 * context.slot_duration)
        {
            auto& oentity = order_entities[active_order_entity_id];
            if(oentity->isActive())
            {
                if(oentity->checkOrderCPU())
                {
                    for(int i = 0; i < oentity->order_kernel_config_params->num_cells; ++i)
                    {
                        auto early = oentity->getEarlyRxPackets(i);
                        auto ontime = oentity->getOnTimeRxPackets(i);
                        auto late = oentity->getLateRxPackets(i);
                        auto SFN = oentity->order_kernel_config_params->SFN;
                        auto SFN_slot = oentity->order_kernel_config_params->subframe_id * ORAN_MAX_SLOT_ID + oentity->order_kernel_config_params->slot_id;
                        auto frame_id = oentity->order_kernel_config_params->frame_id;
                        auto subframe_id = oentity->order_kernel_config_params->subframe_id;
                        auto slot_id = oentity->order_kernel_config_params->slot_id;
                        NVLOGI_FMT(TAG, "[FHGEN] SFN {}.{} F{}S{}S{} ORDER {} Cell {} [RX Packet Times] {{ EARLY: {} ONTIME: {} LATE: {} }}", SFN, SFN_slot, frame_id, subframe_id, slot_id, active_order_entity_id, i, early, ontime, late);

                        auto frame_slots = ORAN_MAX_SLOT_X_SUBFRAME_ID;
                        auto frame_cycle = MAX_LAUNCH_PATTERN_SLOTS / frame_slots;
                        auto slot_80 = (SFN % frame_cycle) * frame_slots + SFN_slot;
                        context.fhgen->getULUPacketStatistics()->increment_counters(i, PacketCounterTiming::EARLY, slot_80, early);
                        context.fhgen->getULUPacketStatistics()->increment_counters(i, PacketCounterTiming::ONTIME, slot_80, ontime);
                        context.fhgen->getULUPacketStatistics()->increment_counters(i, PacketCounterTiming::LATE, slot_80, late);
                        context.fhgen->getULUPacketStatistics()->set_active_slot(slot_80);

                        uint64_t earliest_times[ORAN_ALL_SYMBOLS];
                        uint64_t latest_times[ORAN_ALL_SYMBOLS];
                        for(int sym_idx=0;sym_idx<ORAN_ALL_SYMBOLS;sym_idx++){
                            earliest_times[sym_idx] = oentity->getRxPacketTsEarliest(i,sym_idx);
                            latest_times[sym_idx] = oentity->getRxPacketTsLatest(i,sym_idx);
                        }
                        NVLOGI_FMT(TAG_SYMBOL_TIMINGS,"SFN {}.{} Cell {} early/late ts per sym: {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{} {}:{}",
                        SFN,SFN_slot, i,
                        earliest_times[0],latest_times[0],
                        earliest_times[1],latest_times[1],
                        earliest_times[2],latest_times[2],
                        earliest_times[3],latest_times[3],
                        earliest_times[4],latest_times[4],
                        earliest_times[5],latest_times[5],
                        earliest_times[6],latest_times[6],
                        earliest_times[7],latest_times[7],
                        earliest_times[8],latest_times[8],
                        earliest_times[9],latest_times[9],
                        earliest_times[10],latest_times[10],
                        earliest_times[11],latest_times[11],
                        earliest_times[12],latest_times[12],
                        earliest_times[13],latest_times[13]
                        );
                    }
                    oentity->cleanup();
                    oentity->release();
                    active_order_entity_id = (active_order_entity_id + 1) % kOrderEntityNum;
                }
            }
            usleep(1);
        }
    }

    ACCESS_ONCE(((uint32_t*)context.ul_rx_worker_context.exit_flag[0]->addrh())[0]) = 1;
    NVLOGC_FMT(TAG,"UL RX worker on CPU {} exit after {} slots", cpu, test_slot_count);
    sleep(3);
    context.fhgen->set_workers_exit_signal();
}

}