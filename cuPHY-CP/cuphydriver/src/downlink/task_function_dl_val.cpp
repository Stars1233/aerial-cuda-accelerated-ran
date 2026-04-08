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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 17) // "DRV.FUNC_DL"
#define TAG_DRV_CUPHY_PTI ("DRV.CUPHY_PTI")

#include "cuphydriver_api.hpp"
#include "constant.hpp"
#include "context.hpp"
#include "time.hpp"
#include "task.hpp"
#include "cell.hpp"
#include "slot_map_dl.hpp"
#include "worker.hpp"
#include "phychannel.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "order_entity.hpp"
#include <unordered_map>
#include "aerial-fh-driver/oran.hpp"
#include <sched.h>
#include <unistd.h>
#include "memtrace.h"
#include "nvlog_fmt.hpp"
#include "scf_5g_fapi.h"
#include "app_config.hpp"

int64_t get_packet_time_from_t0(int64_t slot_duration, int64_t beginning_of_time, int64_t frame_cycle, int64_t packet_time, int frame_id, int subframe_id, int slot_id, int symbol_id)
{
    int64_t time_offset = 0;
    int64_t toa;
    uint64_t t0;
    int64_t diff_from_t0 = 0;
    time_offset = ((int64_t)frame_id * ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID + (int64_t)subframe_id * ORAN_MAX_SLOT_ID + (int64_t)slot_id);
    time_offset *= slot_duration * NS_X_US;
    time_offset += (int)(slot_duration * NS_X_US * (float)symbol_id / ORAN_ALL_SYMBOLS);

    int64_t frame_cycle_time_ns = frame_cycle * slot_duration; // TTI MU=1
    frame_cycle_time_ns *= NS_X_US;

    toa = packet_time;
    t0 = beginning_of_time + ((toa - beginning_of_time) / frame_cycle_time_ns) * frame_cycle_time_ns + time_offset;
    if(toa < t0)
    {
        t0 -= frame_cycle_time_ns;
    }
    if(toa > t0 + frame_cycle_time_ns)
    {
        t0 += frame_cycle_time_ns;
    }
    diff_from_t0 = toa - t0;

    if(diff_from_t0 > frame_cycle_time_ns/2)
    {
        diff_from_t0 -= frame_cycle_time_ns;
        t0 += frame_cycle_time_ns;
    }
    return diff_from_t0;
}

// Persistent task to receive and validate DL traffic
int task_work_function_dl_validation(Worker* worker, void* param, int placeholder_1, int placeholder_2, int placeholder_3)
{
    int                                                                          task_num = 1, ret = 0;
    int                                                                          sfn = 0, slot = 0;
    uint32_t cpu;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getcpu failed for {}", __FUNCTION__);
        return -1;
    }
    DLValidationParams* dlparams = reinterpret_cast<DLValidationParams*>(param);
    PhyDriverCtx* pdctx = reinterpret_cast<PhyDriverCtx*>(dlparams->getPhyDriverHandler());
    int start_cell = dlparams->getStartCell();
    int num_cells = dlparams->getNumCells();
    NVLOGC_FMT(TAG, "Start DL Validation thread on CPU {} for RX for start_cell {} num_cells {}", cpu, start_cell, num_cells);

    int peerAbsoluteId = start_cell;
    aerial_fh::MsgReceiveInfo info[512]{};
    int message_type, message_dir, frame_id, subframe_id, slot_id, symbol_id;
    int local_counter[ORAN_MAX_SUBFRAME_ID*ORAN_MAX_SLOT_ID][Packet_Statistics::MAX_DL_PACKET_TYPES][Packet_Statistics::MAX_TIMING_TYPES];

    // Calculate beginning of time (anchor for SFN 0 Slot 0 with GPS time offset)
    int64_t beginning_of_time = 0;
    int64_t toa;
    int64_t diff_from_t0 = 0;
    uint64_t packet_time = 0;
    uint64_t slot_duration_us = 500;
    uint64_t dl_window_size = 51000;
    int64_t frame_cycle = ORAN_MAX_FRAME_ID;
    frame_cycle *= ORAN_MAX_SUBFRAME_ID;
    frame_cycle *= ORAN_MAX_SLOT_ID;

    beginning_of_time = std::chrono::system_clock::now().time_since_epoch().count();
    beginning_of_time /= (10000000ULL * 1024ULL);
    beginning_of_time++;
    beginning_of_time *= (1024ULL * 10000000ULL);
    beginning_of_time += (364ULL * 10000000ULL); // adjust to SFN = 0, accounting for GPS vs TIA conversion

    while(!pdctx->get_exit_dl_validation())
    {
        size_t num_msgs = 512;
        ret = pdctx->getFhProxy()->UserPlaneReceivePacketsCPU(peerAbsoluteId, info, num_msgs);
        if(ret != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UserPlaneReceivePacketsCPU failed for {}", __FUNCTION__);
        }
        // NVLOGC_FMT(TAG, "Received {} packets for cell {}", num_msgs, peerAbsoluteId);

        for(int i = 0; i < num_msgs; ++i)
        {
            auto packet_buffer = (uint8_t*)info[i].buffer;
            //Determine U or C plane
            message_dir = oran_msg_get_data_direction(packet_buffer);
            message_type = oran_msg_get_message_type(packet_buffer);
            packet_time = info[i].rx_timestamp;
            if(message_type == ECPRI_MSG_TYPE_RTC)
            {
                frame_id = oran_cmsg_get_frame_id(packet_buffer);
                subframe_id = oran_cmsg_get_subframe_id(packet_buffer);
                slot_id = oran_cmsg_get_slot_id(packet_buffer);
                symbol_id = oran_cmsg_get_startsymbol_id(packet_buffer);
            }
            else if(message_type == ECPRI_MSG_TYPE_IQ)
            {
                frame_id = oran_umsg_get_frame_id(packet_buffer);
                subframe_id = oran_umsg_get_subframe_id(packet_buffer);
                slot_id = oran_umsg_get_slot_id(packet_buffer);
                symbol_id = oran_umsg_get_symbol_id(packet_buffer);
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{} Ecpri message type not supported {}", __FUNCTION__, message_type);
                continue;
            }

            diff_from_t0 = get_packet_time_from_t0(slot_duration_us, beginning_of_time, frame_cycle, packet_time, frame_id, subframe_id, slot_id, symbol_id);

            int lowerbound;
            int upperbound;
            int packet_type = 0;
            if(message_dir == DIRECTION_UPLINK)
            {
                lowerbound = -1 * dlparams->cell_T1a_max_cp_ul_ns[peerAbsoluteId];
                upperbound = lowerbound + dl_window_size;
                packet_type = Packet_Statistics::dl_packet_type::ULC;
            }
            else if(message_type == ECPRI_MSG_TYPE_RTC)
            {
                lowerbound = -1 * dlparams->cell_T1a_max_up_ns[peerAbsoluteId] - dlparams->cell_Tcp_adv_dl_ns[peerAbsoluteId];
                upperbound = lowerbound + dl_window_size;
                packet_type = Packet_Statistics::dl_packet_type::DLC;
            }
            else
            {
                lowerbound = -1 * dlparams->cell_T1a_max_up_ns[peerAbsoluteId];
                upperbound = lowerbound + dl_window_size;
                packet_type = Packet_Statistics::dl_packet_type::DLU;
            }
            auto slot_80 = subframe_id * ORAN_MAX_SLOT_ID + slot_id;

            if(diff_from_t0 <= lowerbound)
            {
                //early
                local_counter[slot_80][packet_type][Packet_Statistics::EARLY]++;
            }
            else if(diff_from_t0 >= upperbound)
            {
                //late
                local_counter[slot_80][packet_type][Packet_Statistics::LATE]++;
            }
            else
            {
                //ontime
                local_counter[slot_80][packet_type][Packet_Statistics::ONTIME]++;
            }

        }

        for(int i = 0; i < ORAN_MAX_SUBFRAME_ID*ORAN_MAX_SLOT_ID; ++i)
        {
            for(int type = 0; type < Packet_Statistics::MAX_DL_PACKET_TYPES; ++type)
            {
                for(int timing = 0; timing < Packet_Statistics::MAX_TIMING_TYPES; ++timing)
                {
                    if(local_counter[i][type][timing] > 0)
                    {
                        pdctx->getDLPacketStatistics(type)->increment_counter(peerAbsoluteId, (Packet_Statistics::timing_type)timing, i, local_counter[i][type][timing]);
                        pdctx->getDLPacketStatistics(type)->set_active_slot(i);
                    }
                }
            }
        }
        memset(local_counter, 0, sizeof(int) * ORAN_MAX_SUBFRAME_ID*ORAN_MAX_SLOT_ID * Packet_Statistics::MAX_TIMING_TYPES * Packet_Statistics::MAX_DL_PACKET_TYPES);

        ret = pdctx->getFhProxy()->UserPlaneFreePacketsCPU(info, num_msgs);
        if(ret != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UserPlaneFreePacketsCPU failed for {}", __FUNCTION__);
        }

        peerAbsoluteId++;
        if(peerAbsoluteId == start_cell + num_cells)
            peerAbsoluteId = start_cell;
    }
    NVLOGC_FMT(TAG, "Exit {}", cpu);

    return 0;
}