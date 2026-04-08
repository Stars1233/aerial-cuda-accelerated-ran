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

namespace fh_gen
{

int64_t get_recent_frame_cycle_anchor()
{
    int64_t beginning_of_time = now_ns();
    beginning_of_time /= (10000000ULL * 1024ULL);
    beginning_of_time++;
    beginning_of_time *= (1024ULL * 10000000ULL);
    beginning_of_time += (364ULL * 10000000ULL); // adjust to SFN = 0, accounting for GPS vs TIA conversion
    return beginning_of_time;
}

int fss_to_launch_pattern_slot(int frame, int subframe, int slot, int max_count)
{
    return frame % (max_count / (ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID)) * (ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID) + subframe * ORAN_MAX_SLOT_ID + slot;
}

void fronthaul_generator_dl_rx_worker(Worker* worker)
{
    usleep(500);
    uint32_t cpu;
    int ret;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        THROW(StringBuilder() << "getcpu failed for " << __FUNCTION__);
    }
    char threadname[30];
    sprintf(threadname, "%s", "DLRX");
    SET_THREAD_NAME(threadname);
    nvlog_fmtlog_thread_init();
    auto& context = worker->get_context();
    auto& nic = context.nic;
    auto& fhgen = context.fhgen;
    NVLOGC_FMT(TAG,"Start DL RX worker on CPU {}, handling {} cells", cpu, context.dl_rx_worker_context.num_peers);
    aerial_fh::MsgReceiveInfo info[512]{};
    size_t nb_mbufs_rx = 0;

    int64_t frame_cycle_time_ns = ORAN_MAX_FRAME_ID;
    frame_cycle_time_ns *= ORAN_MAX_SUBFRAME_ID;
    frame_cycle_time_ns *= ORAN_MAX_SLOT_ID;
    frame_cycle_time_ns *= context.slot_duration;

    int64_t beginning_of_time = get_recent_frame_cycle_anchor();

    uint64_t packet_time = 0;
    int64_t time_offset;
    uint64_t t0;
    int64_t toa;

    int peer_index = 0;
    int num_peers = context.dl_rx_worker_context.peers.size();

    int64_t window_start = 0;
    int64_t window_end = 0;
    DLPacketCounterType packet_type;
    do
    {
        nb_mbufs_rx = 512;
        ret = aerial_fh::receive(context.dl_rx_worker_context.peers[peer_index], &info[0], &nb_mbufs_rx);
        if(nb_mbufs_rx != 0)
        {
            fhgen->start_ul_tx();
        }

        for(int i = 0; i < nb_mbufs_rx; ++i)
        {
            auto buffer = (uint8_t*)info[i].buffer;
            auto frameId = oran_cmsg_get_frame_id(buffer);
            auto subframeId = oran_cmsg_get_subframe_id(buffer);
            auto slotId = oran_cmsg_get_slot_id(buffer);
            auto symbolId = oran_cmsg_get_startsymbol_id(buffer);
            // Validate symbol ID to prevent buffer overrun
            if(unlikely(symbolId >= ORAN_ALL_SYMBOLS))
            {
                NVLOGW_FMT(TAG, "Invalid symbolId {} in packet from peer {}, F{}S{}S{}, skipping",
                           symbolId, peer_index, frameId, subframeId, slotId);
                continue;
            }
            auto msgType = oran_msg_get_message_type(buffer);
            auto dir = oran_msg_get_data_direction(buffer);
            packet_time = info[i].rx_timestamp;
            toa = packet_time;
            beginning_of_time += ((toa - beginning_of_time) / frame_cycle_time_ns) * frame_cycle_time_ns;
            time_offset = ((int64_t)frameId * ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID + (int64_t)subframeId * ORAN_MAX_SLOT_ID + (int64_t)slotId);
            time_offset *= context.slot_duration;
            time_offset += (int)(context.slot_duration * (float)symbolId / ORAN_ALL_SYMBOLS);
            t0 = beginning_of_time + time_offset;
            toa -= t0;
            if(toa > frame_cycle_time_ns/2)
            {
                toa -= frame_cycle_time_ns;
                t0 += frame_cycle_time_ns;
            }

            if(msgType == ECPRI_MSG_TYPE_IQ)
            {
                window_start = -((int64_t)context.dl_rx_worker_context.t1a_max_up_ns[peer_index]);
                window_end = window_start + context.dl_rx_worker_context.window_end_ns[peer_index];
                packet_type = DLPacketCounterType::DLU;
            }
            else if(dir == DIRECTION_UPLINK)
            {
                window_start = -((int64_t)context.dl_rx_worker_context.t1a_max_cp_ul_ns[peer_index]);
                window_end = window_start + context.dl_rx_worker_context.window_end_ns[peer_index];
                packet_type = DLPacketCounterType::ULC;
            }
            else
            {
                window_start = -((int64_t)context.dl_rx_worker_context.t1a_max_up_ns[peer_index] + context.dl_rx_worker_context.tcp_adv_dl_ns[peer_index]);
                window_end = window_start + context.dl_rx_worker_context.window_end_ns[peer_index];
                packet_type = DLPacketCounterType::DLC;
            }

            auto& packet_timer = fhgen->get_packet_timer()->timers[packet_type][context.dl_rx_worker_context.peer_ids[peer_index]][subframeId * 2 + slotId];
            {
                const std::lock_guard<aerial_fh::FHMutex> lock(packet_timer.mtx);
                if (packet_timer.fss.frameId != frameId && packet_timer.first_packet != true)
                {
                    // Print previous slot info
                    flush_packet_timers(dir, msgType, context.dl_rx_worker_context.peer_ids[peer_index], packet_timer);
                    packet_timer.fss.frameId = frameId;
                    packet_timer.fss.subframeId = subframeId;
                    packet_timer.fss.slotId = slotId;
                    packet_timer.reset();
                    packet_timer.t0 = t0 - (int)(context.slot_duration * (float)symbolId / ORAN_ALL_SYMBOLS); // store t0 for symbol 0
                }

                if (packet_timer.first_packet == true)
                {
                    packet_timer.first_packet = false;
                    packet_timer.fss.frameId = frameId;
                    packet_timer.fss.subframeId = subframeId;
                    packet_timer.fss.slotId = slotId;
                    packet_timer.reset();
                    packet_timer.t0 = t0 - (int)(context.slot_duration * (float)symbolId / ORAN_ALL_SYMBOLS); // store t0 for symbol 0
                }

                int slot = frameId * ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID + subframeId * ORAN_MAX_SLOT_ID + slotId;
                if(toa < window_start)
                {
                    fhgen->increment_dl_counter(context.dl_rx_worker_context.peer_ids[peer_index], packet_type, slot % context.slot_count, PacketCounterTiming::EARLY);
                    ++packet_timer.early;
                }
                else if(toa > window_end)
                {
                    fhgen->increment_dl_counter(context.dl_rx_worker_context.peer_ids[peer_index], packet_type, slot % context.slot_count, PacketCounterTiming::LATE);
                    if(packet_type == DLPacketCounterType::DLU)
                    {
                        int slot_index = fss_to_launch_pattern_slot(frameId, subframeId, slotId, context.slot_count);
                        NVLOGI_FMT(TAG, "F{}S{}S{} 3GPP Slot {} Sym{} Cell {} {} TOA {} Window [{},{}] late", frameId, subframeId, slotId, slot_index, symbolId, peer_index,
                            (packet_type == DLPacketCounterType::DLU) ? "DL U" : (packet_type == DLPacketCounterType::DLC) ? "DL C" : "UL C", toa, window_start, window_end);
                        if(slot_index < kMaxSlotCount)
                            ++fhgen->late_packet_counters[slot_index];
                        else
                        {
                            NVLOGC_FMT(TAG, "SLOT INDEX OUT OF RANGE: F{}S{}S{} 3GPP Slot {}",frameId, subframeId, slotId, slot_index);
                            THROW(StringBuilder() << "Slot index out of bounds " << (int)frameId << " " << (int)subframeId << " " << (int)slotId << " " << (int)slot_index);
                        }
                    }
                    ++packet_timer.late;
                    ++packet_timer.late_packets_per_symbol[symbolId];
                }
                else
                {
                    fhgen->increment_dl_counter(context.dl_rx_worker_context.peer_ids[peer_index], packet_type, slot % context.slot_count, PacketCounterTiming::ONTIME);
                    ++packet_timer.ontime;
                }
                ++packet_timer.num_packets_per_symbol[symbolId];
                ++packet_timer.packet_count;
                packet_timer.max_toa = (packet_timer.max_toa > toa) ? packet_timer.max_toa : toa;

                packet_timer.latest_packet_per_slot = (packet_timer.latest_packet_per_slot > packet_time) ? packet_timer.latest_packet_per_slot : packet_time;
                packet_timer.latest_packet_symbol_num = (packet_timer.latest_packet_per_slot > packet_time) ? packet_timer.latest_packet_symbol_num : symbolId;

                packet_timer.min_toa = (packet_timer.min_toa < toa) ? packet_timer.min_toa : toa;
                packet_timer.earliest_packet_per_slot = (packet_timer.earliest_packet_per_slot < packet_time) ? packet_timer.earliest_packet_per_slot : packet_time;
                packet_timer.earliest_packet_symbol_num = (packet_timer.earliest_packet_per_slot < packet_time) ? packet_timer.earliest_packet_symbol_num : symbolId;

                packet_timer.earliest_packet_per_symbol[symbolId] = packet_timer.earliest_packet_per_symbol[symbolId] > packet_time ? packet_time : packet_timer.earliest_packet_per_symbol[symbolId];
                packet_timer.latest_packet_per_symbol[symbolId] = packet_timer.latest_packet_per_symbol[symbolId] < packet_time ? packet_time : packet_timer.latest_packet_per_symbol[symbolId];

            }
        }

        ret = aerial_fh::free_rx_messages(&info[0],nb_mbufs_rx);
        peer_index = (peer_index + 1) % num_peers;
    } while(!worker->exit_signal());

    NVLOGC_FMT(TAG, "DL RX Worker on CPU {} exit", cpu);

}

}