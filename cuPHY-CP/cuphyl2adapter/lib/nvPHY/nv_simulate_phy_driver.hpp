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

#if !defined(NV_SIMULATE_PHY_DRIVER_HPP_INCLUDED_)
#define NV_SIMULATE_PHY_DRIVER_HPP_INCLUDED_

#include <semaphore.h>

#include <atomic>
#include <thread>
#include <memory>
#include <mutex>
#include <queue>

#include "slot_command/slot_command.hpp"
#include "nv_phy_utils.hpp"

using namespace slot_command_api;

class SlotTask {
public:
    SlotTask(slot_command& command)
    {
        slot_cmd = &command;
    };

    ~SlotTask(){};

    slot_command* get_slot_cmd()
    {
        return slot_cmd;
    }

private:
    slot_command* slot_cmd;
};

class SimulatePhyDriver {
public:
    SimulatePhyDriver(nv::thread_config* cfg);
    ~SimulatePhyDriver();

public:
    void send_uci_indications(slot_indication& si, cell_sub_command& csc);

    int onSlotTask(SlotTask* task);

    int enqueue_phy_work(slot_command& command);
    int set_output_callback(callbacks& cb);

    void phy_driver_thread_func();

    int       enqueue_task(SlotTask* task);
    SlotTask* dequeue_task();

private:
    cuphyPuschDataOut_t  puschDataOut;
    cuphyPuschStatPrms_t puschStatPrms{};
    cuphyPucchDataOut_t  pucchDataOut;
    cuphyPrachDataOut_t  prachDataout;
    cuphySrsDataOut_t    srsDataOut;
    cuphySrsStatPrms_t   srsStatPrms;

    std::vector<uint32_t> tbOffset;

    slot_command_api::uci_output_params outParams = {.numHarq             = 1,
                                                     .harqConfidenceLevel = 0,
                                                     .harq_pdu            = {0x80},
                                                     .srIndication        = 1,
                                                     .srConfidenceLevel   = 0};

    static uint8_t  zero_u8[];
    static uint16_t zero_u16[];
    static uint32_t zero_u32[];
    static float    zero_float[];

    uint32_t pStartOffsetsCbCrc[slot_command_api::MAX_PUSCH_UE_PER_TTI];
    uint32_t pStartOffsetsTbCrc[slot_command_api::MAX_PUSCH_UE_PER_TTI];
    uint32_t pStartOffsetsTbPayload[slot_command_api::MAX_PUSCH_UE_PER_TTI];

    slot_command_api::dl_slot_callbacks dl_cb;
    slot_command_api::ul_slot_callbacks ul_cb;

    nv::thread_config* worker_cfg;
    sem_t              sem;
    std::atomic<bool>  stop_{false};
    std::thread        thread;

    std::mutex            queue_mutex;
    std::queue<SlotTask*> task_queue;
};

#endif // NV_SIMULATE_PHY_DRIVER_HPP_INCLUDED_
