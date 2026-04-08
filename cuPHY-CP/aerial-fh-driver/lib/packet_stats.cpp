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

#include "aerial-fh-driver/packet_stats.hpp"
#include "utils.hpp"

#define TAG 650 //FHGEN

void Packet_Statistics::increment_counters(int cell, int timing, int slot, int inc)
{
    if(cell >= STAT_ARRAY_CELL_SIZE || slot >= MAX_LAUNCH_PATTERN_SLOTS || timing >= CounterTimingMax)
    {
        // Invalid parameters
        return;
    }
    set_active_slot(slot);
    stats[cell][slot][timing] += inc;
    total_stats[cell][timing] += inc;
}

uint64_t Packet_Statistics::get_cell_timing_slot_count(int cell, int timing, int slot)
{
    if(cell >= STAT_ARRAY_CELL_SIZE || slot >= MAX_LAUNCH_PATTERN_SLOTS || timing >= CounterTimingMax)
    {
        // Invalid parameters
        return 0;
    }

    return stats[cell][slot][timing].load();
}

uint64_t Packet_Statistics::get_cell_timing_count(int cell, int timing)
{
    if(cell >= STAT_ARRAY_CELL_SIZE || timing >= CounterTimingMax)
    {
        // Invalid parameters
        return 0;
    }

    return total_stats[cell][timing].load();
}

uint64_t Packet_Statistics::get_cell_total_count(int cell)
{
    if(cell >= STAT_ARRAY_CELL_SIZE)
    {
        // Invalid parameters
        return 0;
    }
    auto ret = get_cell_timing_count(cell, PacketCounterTiming::EARLY);
    ret += get_cell_timing_count(cell, PacketCounterTiming::ONTIME);
    ret += get_cell_timing_count(cell, PacketCounterTiming::LATE);
    return ret;
}

uint64_t Packet_Statistics::get_cell_slot_total_count(int cell, int slot)
{
    if(cell >= STAT_ARRAY_CELL_SIZE || slot >= MAX_LAUNCH_PATTERN_SLOTS)
    {
        // Invalid parameters
        return 0;
    }
    auto ret = get_cell_timing_slot_count(cell, PacketCounterTiming::EARLY, slot);
    ret += get_cell_timing_slot_count(cell, PacketCounterTiming::ONTIME, slot);
    ret += get_cell_timing_slot_count(cell, PacketCounterTiming::LATE, slot);
    return ret;
}

float Packet_Statistics::get_cell_timing_percentage(int cell, int timing)
{
    if(cell >= STAT_ARRAY_CELL_SIZE || timing >= CounterTimingMax)
    {
        // Invalid parameters
        return 0;
    }
    auto total = get_cell_total_count(cell);
    total = (total == 0) ? 1 : total;
    auto ret = (float)(get_cell_timing_count(cell, timing) / total * 100.0);
    return ret;
}

float Packet_Statistics::get_cell_timing_slot_percentage(int cell, int timing, int slot)
{
    if(cell >= STAT_ARRAY_CELL_SIZE || timing >= CounterTimingMax)
    {
        // Invalid parameters
        return 0;
    }
    auto total = get_cell_slot_total_count(cell, slot);
    total = (total == 0) ? 1 : total;
    auto ret = ((float)(get_cell_timing_slot_count(cell, timing, slot)) / total * 100.0);
    return ret;
}

bool Packet_Statistics::pass_slot_percentage(int cell, float threshold)
{
    bool pass = true;
    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
        if(active_slots[slot])
        {
            auto ontime_percentage = get_cell_timing_slot_percentage(cell, PacketCounterTiming::ONTIME, slot);
            if(threshold > ontime_percentage)
            {
                NVLOGC_FMT(650, "Cell {} slot {} on time {}\% did not pass {}\%", cell, slot, ontime_percentage, threshold);
                pass = false;
            }
        }
    }
    return pass;
}


void Packet_Statistics::clear_counter(int cell, int type, int slot)
{
    if(cell >= STAT_ARRAY_CELL_SIZE || slot >= MAX_LAUNCH_PATTERN_SLOTS || type >= CounterTimingMax)
    {
        // Invalid parameters
        return;
    }
    stats[cell][slot][type].store(0);
}

void Packet_Statistics::set_active_slot(int slot)
{
    active_slots[slot] = true;
    active_ = true;
}

void Packet_Statistics::reset()
{
    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
            for(int cell = 0; cell < STAT_ARRAY_CELL_SIZE; ++cell)
            {
                for(int timing = 0; timing < CounterTimingMax; ++timing)
                {
                    stats[cell][slot][timing].store(0);
                    total_stats[cell][timing].store(0);
                }
            }
    }
}

void Packet_Statistics::flush_counters(int num_cells, int packet_type)
{
    if(num_cells > STAT_ARRAY_CELL_SIZE)
    {
        // Invalid parameters
        return;
    }

    int buffer_index = 0;
    char buffer[MAX_PRINT_LOG_LENGTH];

    buffer_index = 0;
    buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "slot,");
    for(int cell = 0; cell < num_cells; ++cell)
    {
        buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "cell_%d_early,cell_%d_ontime,cell_%d_late,", cell, cell, cell);
    }

    NVLOGC_FMT(PACKET_SUMMARY_TAG, "{}",buffer);

    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
        if(active_slots[slot])
        {
            buffer_index = 0;
            switch(packet_type) {
            case -1:
                break;
            case DLPacketCounterType::DLC:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "DLC ");
                break;
            case DLPacketCounterType::DLU:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "DLU ");
                break;
            case DLPacketCounterType::ULC:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULC ");
                break;
            case ULUPacketCounterType::ULU_PRACH:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULU TX PRACH ");
                break;
            case ULUPacketCounterType::ULU_PUCCH:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULU TX PUCCH ");
                break;
            case ULUPacketCounterType::ULU_PUSCH:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULU TX PUSCH ");
                break;
            case ULUPacketCounterType::ULU_SRS:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULU TX SRS ");
                break;
            default:
                break;
            }

            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "Slot %d |", slot);

            for(int cell = 0; cell < num_cells; ++cell)
            {
                auto early = stats[cell][slot][EARLY].load();
                auto ontime = stats[cell][slot][ONTIME].load();
                auto late = stats[cell][slot][LATE].load();
                auto total = early + ontime + late;
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, " %lu,%lu,%lu |", early, ontime, late);
            }
            NVLOGC_FMT(PACKET_SUMMARY_TAG, "{}",buffer);
        }
    }

    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
        if(active_slots[slot])
        {
            buffer_index = 0;
            switch(packet_type) {
            case -1:
                break;
            case DLPacketCounterType::DLC:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "DLC ");
                break;
            case DLPacketCounterType::DLU:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "DLU ");
                break;
            case DLPacketCounterType::ULC:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULC ");
                break;
            case ULUPacketCounterType::ULU_PRACH:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULU TX PRACH ");
                break;
            case ULUPacketCounterType::ULU_PUCCH:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULU TX PUCCH ");
                break;
            case ULUPacketCounterType::ULU_PUSCH:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULU TX PUSCH ");
                break;
            case ULUPacketCounterType::ULU_SRS:
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "ULU TX SRS ");
                break;
            default:
                break;
            }

            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "Slot %d |", slot);

            for(int cell = 0; cell < num_cells; ++cell)
            {
                auto early = stats[cell][slot][EARLY].load();
                auto ontime = stats[cell][slot][ONTIME].load();
                auto late = stats[cell][slot][LATE].load();
                auto total = early + ontime + late;
                total = (total == 0) ? 1 : total;
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, " %.2f,%.2f,%.2f |", (float)early / total * 100.0, (float)ontime / total * 100.0, (float)late / total * 100.0);
            }
            NVLOGC_FMT(PACKET_SUMMARY_TAG, "{}",buffer);
        }
    }
}


void Packet_Statistics::flush_counters_file(int num_cells, std::string filename)
{
    if(num_cells > STAT_ARRAY_CELL_SIZE)
    {
        // Invalid parameters
        return;
    }

    int buffer_index = 0;
    char buffer[MAX_PRINT_LOG_LENGTH];

    FILE *fp;

    fp = fopen(filename.c_str(), "w+");
    if(unlikely(fp == nullptr))
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "flush_counters_file: Failed to open file {}", filename);
        return;
    }

    buffer_index = 0;
    buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "slot,");
    for(int cell = 0; cell < num_cells; ++cell)
    {
        buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "cell_%d_early,cell_%d_ontime,cell_%d_late,", cell, cell, cell);
    }

    fprintf(fp, "%s\n", buffer);


    for(int slot = 0; slot < MAX_LAUNCH_PATTERN_SLOTS; ++slot)
    {
        if(active_slots[slot])
        {
            buffer_index = 0;
            buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "%d,", slot);

            for(int cell = 0; cell < num_cells; ++cell)
            {
                auto early = stats[cell][slot][EARLY].load();
                auto ontime = stats[cell][slot][ONTIME].load();
                auto late = stats[cell][slot][LATE].load();
                auto total = early + ontime + late;
                buffer_index += snprintf(buffer+buffer_index, MAX_PRINT_LOG_LENGTH-buffer_index, "%lu,%lu,%lu,", early, ontime, late);
            }
            fprintf(fp, "%s\n", buffer);
        }
    }
   fclose(fp);
}

// Called function at the end of every slot to print the early/late/ontime packets per symbol on the slot
void flush_packet_timers(uint8_t dir, uint8_t type, uint8_t cell_index, struct packet_timer_per_slot& packet_timer)
{
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