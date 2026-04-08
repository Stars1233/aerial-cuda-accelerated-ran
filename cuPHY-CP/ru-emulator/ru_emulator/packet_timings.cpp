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

#include "ru_emulator.hpp"

#undef TAG_LATE_PACKETS
#define TAG_LATE_PACKETS (NVLOG_TAG_BASE_RU_EMULATOR + 3) // "RU.LATE_PACKETS"
#undef TAG_SYMBOL_TIMINGS
#define TAG_SYMBOL_TIMINGS (NVLOG_TAG_BASE_RU_EMULATOR + 4) // "RU.SYMBOL_TIMINGS"

void RU_Emulator::flush_packet_timers(uint8_t dir, uint8_t type, uint8_t cell_index, struct packet_timer_per_slot& packet_timer)
{
    auto packet_type = (dir == oran_pkt_dir::DIRECTION_DOWNLINK) ? ((type == ECPRI_MSG_TYPE_RTC) ? DL_C_PLANE : DL_U_PLANE) : UL_C_PLANE;
    ++slot_count[packet_type][cell_index];

    NVLOGI_FMT(TAG_LATE_PACKETS,"F{}S{}S{} Cell {} {} {} Plane T0 {} num pkts {} max {} min {} early {}, ontime {}, late {}, earliest packet {} sym {}, latest packet {} sym {}, duration {}",
        packet_timer.fss.frameId,
        packet_timer.fss.subframeId,
        packet_timer.fss.slotId,
        cell_index,
        (dir == oran_pkt_dir::DIRECTION_DOWNLINK) ? "DL" : "UL",
        (type == ECPRI_MSG_TYPE_RTC) ? "C" : "U",
        packet_timer.t0,
        packet_timer.packet_count,
        packet_timer.max_toa,
        packet_timer.min_toa,
        packet_timer.early,
        packet_timer.ontime,
        packet_timer.late,
        packet_timer.earliest_packet_per_slot,
        packet_timer.earliest_packet_symbol_num,
        packet_timer.latest_packet_per_slot,
        packet_timer.latest_packet_symbol_num,
        packet_timer.latest_packet_per_slot - packet_timer.earliest_packet_per_slot
        );


    NVLOGI_FMT(TAG_LATE_PACKETS,"F{}S{}S{} Cell {} {} {} Plane T0 {} late packets Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {} Sym {} {}",
        packet_timer.fss.frameId,
        packet_timer.fss.subframeId,
        packet_timer.fss.slotId,
        cell_index,
        (dir == oran_pkt_dir::DIRECTION_DOWNLINK) ? "DL" : "UL",
        (type == ECPRI_MSG_TYPE_RTC) ? "C" : "U",
        packet_timer.t0,
        0, packet_timer.late_packets_per_symbol[0],
        1, packet_timer.late_packets_per_symbol[1],
        2, packet_timer.late_packets_per_symbol[2],
        3, packet_timer.late_packets_per_symbol[3],
        4, packet_timer.late_packets_per_symbol[4],
        5, packet_timer.late_packets_per_symbol[5],
        6, packet_timer.late_packets_per_symbol[6],
        7, packet_timer.late_packets_per_symbol[7],
        8, packet_timer.late_packets_per_symbol[8],
        9, packet_timer.late_packets_per_symbol[9],
        10, packet_timer.late_packets_per_symbol[10],
        11, packet_timer.late_packets_per_symbol[11],
        12, packet_timer.late_packets_per_symbol[12],
        13, packet_timer.late_packets_per_symbol[13]
    );

    if(opt_debug_u_plane_prints != 0)
    {
        if(type == ECPRI_MSG_TYPE_IQ)
        {
            NVLOGI_FMT(TAG_LATE_PACKETS,"F{}S{}S{} Cell {} packet_timer.max_toa / NS_X_US {} -(int64_t)oran_timing_info.dl_u_plane_timing_delay {}",
                packet_timer.fss.frameId,
                packet_timer.fss.subframeId,
                packet_timer.fss.slotId,
                cell_index,packet_timer.max_toa / NS_X_US,  (-(int64_t)oran_timing_info.dl_u_plane_timing_delay)
                );
            if(opt_debug_u_plane_threshold < (packet_timer.max_toa / NS_X_US) - (-(int64_t)oran_timing_info.dl_u_plane_timing_delay))
            {
                for(int sym = 0; sym < ORAN_ALL_SYMBOLS; ++sym)
                {
                    for(int i = 0; i < packet_timer.num_packets_per_symbol[sym]; ++i)
                    {
                        NVLOGI_FMT(TAG_LATE_PACKETS, "F{}S{}S{} Sym {} Cell {} {} {} Plane T0 {} Packet {}/{} TS {} TOA {}",
                            packet_timer.fss.frameId,
                            packet_timer.fss.subframeId,
                            packet_timer.fss.slotId,
                            sym,
                            cell_index,
                            (dir == oran_pkt_dir::DIRECTION_DOWNLINK) ? "DL" : "UL",
                            (type == ECPRI_MSG_TYPE_RTC) ? "C" : "U",
                            packet_timer.t0,
                            i+1,
                            packet_timer.num_packets_per_symbol[sym],
                            packet_timer.packet_arrive_abs[sym][i],
                            packet_timer.packet_arrive_t0s[sym][i]);
                    }
                }
            }
        }
    }

    NVLOGI_FMT(TAG_SYMBOL_TIMINGS,"F{}S{}S{} Cell {} {} {} Plane T0 {} earliest latest TS Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {} Sym {} {} {}",
        packet_timer.fss.frameId,
        packet_timer.fss.subframeId,
        packet_timer.fss.slotId,
        cell_index,
        (dir == oran_pkt_dir::DIRECTION_DOWNLINK) ? "DL" : "UL",
        (type == ECPRI_MSG_TYPE_RTC) ? "C" : "U",
        packet_timer.t0,
        0, packet_timer.earliest_packet_per_symbol[0], packet_timer.latest_packet_per_symbol[0],
        1, packet_timer.earliest_packet_per_symbol[1], packet_timer.latest_packet_per_symbol[1],
        2, packet_timer.earliest_packet_per_symbol[2], packet_timer.latest_packet_per_symbol[2],
        3, packet_timer.earliest_packet_per_symbol[3], packet_timer.latest_packet_per_symbol[3],
        4, packet_timer.earliest_packet_per_symbol[4], packet_timer.latest_packet_per_symbol[4],
        5, packet_timer.earliest_packet_per_symbol[5], packet_timer.latest_packet_per_symbol[5],
        6, packet_timer.earliest_packet_per_symbol[6], packet_timer.latest_packet_per_symbol[6],
        7, packet_timer.earliest_packet_per_symbol[7], packet_timer.latest_packet_per_symbol[7],
        8, packet_timer.earliest_packet_per_symbol[8], packet_timer.latest_packet_per_symbol[8],
        9, packet_timer.earliest_packet_per_symbol[9], packet_timer.latest_packet_per_symbol[9],
        10, packet_timer.earliest_packet_per_symbol[10], packet_timer.latest_packet_per_symbol[10],
        11, packet_timer.earliest_packet_per_symbol[11], packet_timer.latest_packet_per_symbol[11],
        12, packet_timer.earliest_packet_per_symbol[12], packet_timer.latest_packet_per_symbol[12],
        13, packet_timer.earliest_packet_per_symbol[13], packet_timer.latest_packet_per_symbol[13]
    );
}

void RU_Emulator::increment_oran_packet_counters(uint8_t type, uint8_t cell_index, struct packet_timer_per_slot& packet_timer, uint8_t curr_launch_pattern_slot)
{
    switch(type)
    {
        case rx_packet_type::DL_C_PLANE:
        {
            if(oran_packet_counters.dl_c_plane[cell_index].total_slot.load() >= opt_dl_warmup_slots)
            {
                if(packet_timer.late != 0)
                {
                    ++oran_packet_counters.dl_c_plane[cell_index].late_slot;
                    ++oran_packet_counters.dl_c_plane[cell_index].late_slots_for_slot_num[curr_launch_pattern_slot];
                }
                else if(packet_timer.early != 0)
                {
                    ++oran_packet_counters.dl_c_plane[cell_index].early_slot;
                    ++oran_packet_counters.dl_c_plane[cell_index].early_slots_for_slot_num[curr_launch_pattern_slot];
                }
                else
                {
                    ++oran_packet_counters.dl_c_plane[cell_index].ontime_slot;
                    ++oran_packet_counters.dl_c_plane[cell_index].ontime_slots_for_slot_num[curr_launch_pattern_slot];
                }
                ++oran_packet_counters.dl_c_plane[cell_index].total_slots_for_slot_num[curr_launch_pattern_slot];
            }
            ++oran_packet_counters.dl_c_plane[cell_index].total_slot;
            break;
        }
        case rx_packet_type::DL_U_PLANE:
        {
            if(oran_packet_counters.dl_u_plane[cell_index].total_slot.load() >= opt_dl_warmup_slots)
            {
                if(packet_timer.late != 0)
                {
                    ++oran_packet_counters.dl_u_plane[cell_index].late_slot;
                    ++oran_packet_counters.dl_u_plane[cell_index].late_slots_for_slot_num[curr_launch_pattern_slot];
                }
                else if(packet_timer.early != 0)
                {
                    ++oran_packet_counters.dl_u_plane[cell_index].early_slot;
                    ++oran_packet_counters.dl_u_plane[cell_index].early_slots_for_slot_num[curr_launch_pattern_slot];
                }
                else
                {
                    ++oran_packet_counters.dl_u_plane[cell_index].ontime_slot;
                    ++oran_packet_counters.dl_u_plane[cell_index].ontime_slots_for_slot_num[curr_launch_pattern_slot];
                }
                ++oran_packet_counters.dl_u_plane[cell_index].total_slots_for_slot_num[curr_launch_pattern_slot];
            }
            ++oran_packet_counters.dl_u_plane[cell_index].total_slot;
            break;
        }
        case rx_packet_type::UL_C_PLANE:
        {
            if(oran_packet_counters.ul_c_plane[cell_index].total_slot.load() >= opt_ul_warmup_slots)
            {
                if(packet_timer.late != 0)
                {
                    ++oran_packet_counters.ul_c_plane[cell_index].late_slot;
                    ++oran_packet_counters.ul_c_plane[cell_index].late_slots_for_slot_num[curr_launch_pattern_slot];
                }
                else if(packet_timer.early != 0)
                {
                    ++oran_packet_counters.ul_c_plane[cell_index].early_slot;
                    ++oran_packet_counters.ul_c_plane[cell_index].early_slots_for_slot_num[curr_launch_pattern_slot];
                }
                else
                {
                    ++oran_packet_counters.ul_c_plane[cell_index].ontime_slot;
                    ++oran_packet_counters.ul_c_plane[cell_index].ontime_slots_for_slot_num[curr_launch_pattern_slot];
                }
                ++oran_packet_counters.ul_c_plane[cell_index].total_slots_for_slot_num[curr_launch_pattern_slot];

            }
            ++oran_packet_counters.ul_c_plane[cell_index].total_slot;
            break;
        }
        default:
            break;
    }
}

void RU_Emulator::process_cplane_timing(uint8_t cell_index, struct packet_timer_per_slot& packet_timer, oran_c_plane_info_t& c_plane_info, uint64_t packet_time, int64_t toa, int64_t slot_t0)
{
    std::lock_guard<aerial_fh::FHMutex> lock(packet_timer.mtx);
    if(packet_timer.fss.frameId != c_plane_info.fss.frameId && packet_timer.first_packet != true)
    {
        //Print previous slot info
        flush_packet_timers(c_plane_info.dir, ECPRI_MSG_TYPE_RTC, cell_index, packet_timer);
        increment_oran_packet_counters(c_plane_info.dir, cell_index, packet_timer, c_plane_info.launch_pattern_slot);
        packet_timer.reset();

        packet_timer.fss.frameId = c_plane_info.fss.frameId;
        packet_timer.fss.subframeId = c_plane_info.fss.subframeId;
        packet_timer.fss.slotId = c_plane_info.fss.slotId;
    }

    if(packet_timer.first_packet == true)
    {
        packet_timer.first_packet = false;
        packet_timer.reset();
        packet_timer.fss.frameId = c_plane_info.fss.frameId;
        packet_timer.fss.subframeId = c_plane_info.fss.subframeId;
        packet_timer.fss.slotId = c_plane_info.fss.slotId;
    }

    if(packet_timer.earliest_packet_per_slot > packet_time)
    {
        packet_timer.earliest_packet_per_slot = packet_time;
        packet_timer.earliest_packet_symbol_num = c_plane_info.startSym;
    }

    if(packet_timer.latest_packet_per_slot < packet_time)
    {
        packet_timer.latest_packet_per_slot = packet_time;
        packet_timer.latest_packet_symbol_num = c_plane_info.startSym;
    }

    if(packet_timer.latest_packet_per_symbol[c_plane_info.startSym] < packet_time)
    {
        packet_timer.latest_packet_per_symbol[c_plane_info.startSym] = packet_time;
    }
    if(packet_timer.earliest_packet_per_symbol[c_plane_info.startSym] > packet_time)
    {
        packet_timer.earliest_packet_per_symbol[c_plane_info.startSym] = packet_time;
    }
    ++packet_timer.packet_count;
    packet_timer.max_toa = (packet_timer.max_toa < toa) ? toa : packet_timer.max_toa;
    packet_timer.min_toa = (packet_timer.min_toa > toa) ? toa : packet_timer.min_toa;
    packet_timer.t0 = slot_t0;

    if(c_plane_info.dir == oran_pkt_dir::DIRECTION_DOWNLINK)
    {
        if(toa < -((int64_t)oran_timing_info.dl_c_plane_timing_delay * NS_X_US))
        {
            ++oran_packet_counters.dl_c_plane[cell_index].early_packet;
            ++packet_timer.early;
            dl_c_packet_stats.increment_counters(cell_index, PacketCounterTiming::EARLY, c_plane_info.launch_pattern_slot, 1);
        }
        else if(toa > -((int64_t)oran_timing_info.dl_c_plane_timing_delay * NS_X_US) + (int64_t)oran_timing_info.dl_c_plane_window_size * NS_X_US)
        {
            ++oran_packet_counters.dl_c_plane[cell_index].late_packet;
            ++packet_timer.late;
            ++packet_timer.late_packets_per_symbol[c_plane_info.startSym];
            dl_c_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, 1);
        }
        else
        {
            ++oran_packet_counters.dl_c_plane[cell_index].ontime_packet;
            ++packet_timer.ontime;
            dl_c_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, 1);
        }
    }
    else
    {
        if(toa < -((int64_t)oran_timing_info.ul_c_plane_timing_delay * NS_X_US))
        {
            ++oran_packet_counters.ul_c_plane[cell_index].early_packet;
            ++packet_timer.early;
            ul_c_packet_stats.increment_counters(cell_index, PacketCounterTiming::EARLY, c_plane_info.launch_pattern_slot, 1);
        }
        else if(toa > -((int64_t)oran_timing_info.ul_c_plane_timing_delay * NS_X_US) + (int64_t)oran_timing_info.ul_c_plane_window_size * NS_X_US)
        {
            ++oran_packet_counters.ul_c_plane[cell_index].late_packet;
            ++packet_timer.late;
            ++packet_timer.late_packets_per_symbol[c_plane_info.startSym];
            ul_c_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, c_plane_info.launch_pattern_slot, 1);
        }
        else
        {
            ++oran_packet_counters.ul_c_plane[cell_index].ontime_packet;
            ++packet_timer.ontime;
            ul_c_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, c_plane_info.launch_pattern_slot, 1);
        }
    }
}