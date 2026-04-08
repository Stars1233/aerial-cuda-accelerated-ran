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
#include <chrono>
#include "timing_utils.hpp"

#define RX_TIMESTAMP

int RU_Emulator::find_eAxC_index_from_flowVal(cell_config& cell_config,uint16_t flowVal,uint8_t& index)
{
    int ret=0;
    auto it = find(cell_config.eAxC_DL.begin(), cell_config.eAxC_DL.end(), flowVal);
    if(it==cell_config.eAxC_DL.end())
        ret=-1;
    else
        index = it - cell_config.eAxC_DL.begin();
    return ret;
}

void RU_Emulator::validate_dl_channels(uint8_t cell_index, uint8_t curr_launch_pattern_slot, const struct oran_packet_header_info &header_info, void *section_buffer, uint64_t &prev_pbch_time)
{
    if (opt_pbch_validation == RE_ENABLED && pbch_object.launch_pattern[curr_launch_pattern_slot].size() > 0)
    {
        validate_pbch(cell_index, header_info, section_buffer, prev_pbch_time);
    }
    if (opt_pdsch_validation == RE_ENABLED && pdsch_object.launch_pattern[curr_launch_pattern_slot].size() > 0)
    {
        validate_pdsch(cell_index, header_info, section_buffer);
    }
    if (opt_pdcch_ul_validation == RE_ENABLED && pdcch_ul_object.launch_pattern[curr_launch_pattern_slot].size() > 0)
    {
        validate_pdcch(cell_index, header_info, section_buffer, dl_channel::PDCCH_UL);
    }
    if (opt_pdcch_dl_validation == RE_ENABLED && pdcch_dl_object.launch_pattern[curr_launch_pattern_slot].size() > 0)
    {
        validate_pdcch(cell_index, header_info, section_buffer, dl_channel::PDCCH_DL);
    }
    if (opt_csirs_validation == RE_ENABLED && csirs_object.launch_pattern[curr_launch_pattern_slot].size() > 0)
    {
        validate_csirs(cell_index, header_info, section_buffer);
    }
}

// validate_uplane_section_id_match moved to sectionid_validation.cpp (cold)

void *uplane_proc_rx_core_wrapper(void *arg)
{
    if (!arg) {
        do_throw(sb() << "Error: arg == nullptr with uplane_proc_rx_core_wrapper");
    }
    auto* dl_core_info = reinterpret_cast<struct RU_Emulator::dl_core_info *>(arg);
    if (!dl_core_info->rue) {
        do_throw(sb() << "Error: rue == nullptr with uplane_proc_rx_core_wrapper");
    }
    RU_Emulator *rue = static_cast<RU_Emulator*>(dl_core_info->rue);
    return rue->uplane_proc_rx_core(arg);
}

void *RU_Emulator::uplane_proc_rx_core(void *arg)
{
    nvlog_fmtlog_thread_init();
    re_cons("Thread {} initialized fmtlog", __FUNCTION__);
    struct dl_core_info dl_core_info = *(struct dl_core_info *)arg;
    int core_index = dl_core_info.core_index;
    uint16_t buf_write_idx=0;
    uint32_t wrap_index=0;
    int cell_index = 0;
    int flow_index = 0;
    char threadname[30];
    sprintf(threadname, "%s-%d", "uplane_proc_rx", dl_core_info.core_index);
    SET_THREAD_NAME(threadname);
    uint32_t cpu;
    int ret = getcpu(&cpu, nullptr);

    int flow_count = std::max(1, (int)dl_core_info.flow_count);
    re_cons("Uplane Proc Rx Core {} CPU Id {} handling {} flows", core_index, cpu, dl_core_info.flow_count);

    size_t nb_mbufs_rx = 0;

    uint8_t flow_counter = 0;


    pkt_buf_write_idx[core_index]=buf_write_idx;

    while(!check_force_quit())
    {
        cell_index = dl_core_info.flow_infos[flow_counter].cell_index;
        flow_index = dl_core_info.flow_infos[flow_counter].flowId;
        nb_mbufs_rx = MAX_PACKET_PER_RX_BURST;
        ret = aerial_fh::receive_flow(dl_peer_flow_map[cell_index][flow_index], &mbuf_info[core_index*PACKET_RX_BUFFER_COUNT*MAX_PACKET_PER_RX_BURST+buf_write_idx*MAX_PACKET_PER_RX_BURST], &nb_mbufs_rx);

        if ( ret != 0){
            do_throw(sb() <<"PEER: "<<cell_index<<" Could not receive data on flow \n"<<flow_counter<<"\n");
        }

        if(nb_mbufs_rx == 0)
        {
            ++flow_counter;
            if(flow_counter == flow_count)
            {
                flow_counter = 0;
            }
            continue;
        }

        re_info("[dl_rx_core] Cell {} Core {} buf_write_idx {} nb_mbufs_rx {} flow_counter{} wrap_index{}",
                cell_index,core_index,buf_write_idx,nb_mbufs_rx,flow_counter,wrap_index);

        num_mbufs_rx[core_index*PACKET_RX_BUFFER_COUNT+buf_write_idx]=nb_mbufs_rx;
        pkt_flow_counter[core_index*PACKET_RX_BUFFER_COUNT+buf_write_idx]=flow_counter;
        if((buf_write_idx+1)==PACKET_RX_BUFFER_COUNT)
        {
            wrap_index+=1;
        }
        buf_write_idx=(buf_write_idx+1)%PACKET_RX_BUFFER_COUNT;
        pkt_buf_write_idx[core_index]=buf_write_idx;

        ++flow_counter;
        if(flow_counter == flow_count)
        {
            flow_counter = 0;
        }
    }
    re_cons("Thread {} exiting", threadname);

    return NULL;
}

void *uplane_proc_validate_core_wrapper(void *arg)
{
    if (!arg) {
        do_throw(sb() << "Error: arg == nullptr with uplane_proc_validate_core_wrapper");
    }
    auto* dl_core_info = reinterpret_cast<struct RU_Emulator::dl_core_info *>(arg);
    if (!dl_core_info->rue) {
        do_throw(sb() << "Error: rue == nullptr with uplane_proc_validate_core_wrapper");
    }
    RU_Emulator *rue = static_cast<RU_Emulator*>(dl_core_info->rue);
    return rue->uplane_proc_validate_core(arg);
}

void *RU_Emulator::uplane_proc_validate_core(void *arg)
{
    nvlog_fmtlog_thread_init();
    re_cons("Thread {} initialized fmtlog", __FUNCTION__);
    struct dl_core_info dl_core_info = *(struct dl_core_info *)arg;
    int core_index = dl_core_info.core_index;
    uint32_t wrap_index=0;
    int cell_index = 0;
    int flow_index = 0;
    char threadname[30];
    sprintf(threadname, "%s-%d", "uplane_proc_validate", dl_core_info.core_index);
    SET_THREAD_NAME(threadname);

    int flow_count = std::max(1, (int)dl_core_info.flow_count);
    re_cons("Uplane Proc Validate Core {} handling {} flows", core_index, dl_core_info.flow_count);

    size_t nb_mbufs_rx = 0;
    uint64_t slot_t0;
    uint64_t packet_time = 0;
    int ret;

    uint64_t validate_start_t = 0;
    uint64_t validate_end_t = 0;
    int64_t toa;
    struct oran_packet_header_info header_info;
    uint64_t prev_pbch_time;
    uint8_t flow_counter = 0;
    uint8_t* buffer;

    uint8_t curr_launch_pattern_slot = 0;
    struct fssId curr_fss{0,0,0};
    std::array<std::array<std::array<uint64_t, CounterTimingMax>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> local_packet_counters{};

    uint16_t buf_read_idx;
    uint16_t buf_write_idx;

    //Profiling variables : Only use for accurate interpretation if all the flows of a cell are mapped to the same CPU core
    bool first_packet_of_slot_rx_burst[MAX_FLOWS_PER_DL_CORE];
    uint64_t slot_validate_start_t[MAX_FLOWS_PER_DL_CORE];
    uint64_t slot_validate_end_t[MAX_FLOWS_PER_DL_CORE];
    size_t packet_count_slot[MAX_FLOWS_PER_DL_CORE];
    uint8_t prev_launch_pattern_slot[MAX_FLOWS_PER_DL_CORE];
    struct fssId prev_fss[MAX_FLOWS_PER_DL_CORE];
    uint64_t prev_t0[MAX_FLOWS_PER_DL_CORE];

    // Init peofiling variables
    for (int i = 0; i < MAX_FLOWS_PER_DL_CORE; i++)
    {
        first_packet_of_slot_rx_burst[i] = true;
        slot_validate_start_t[i] = 0;
        slot_validate_end_t[i] = 0;
        packet_count_slot[i] = 0;
        prev_t0[i] = 0;
        prev_launch_pattern_slot[i] = MAX_LAUNCH_PATTERN_SLOTS;
        prev_fss[i] = {0,0,0};
    }

    int64_t frame_cycle_time_ns = get_frame_cycle_time_ns(max_slot_id, opt_tti_us);
    int64_t first_f0s0s0_time = get_first_f0s0s0_time();

    pkt_buf_read_idx[core_index]=0;

    while(!check_force_quit())
    {
        buf_read_idx=pkt_buf_read_idx[core_index];
        buf_write_idx=pkt_buf_write_idx[core_index];

        while(buf_read_idx!=buf_write_idx)
        {
            flow_counter=pkt_flow_counter[core_index*PACKET_RX_BUFFER_COUNT+buf_read_idx];
            nb_mbufs_rx=num_mbufs_rx[core_index*PACKET_RX_BUFFER_COUNT+buf_read_idx];
            cell_index = dl_core_info.flow_infos[flow_counter].cell_index;
            flow_index = dl_core_info.flow_infos[flow_counter].flowId;

            validate_start_t = get_ns();
            for(int i = 0; i < nb_mbufs_rx; ++i)
            {
                auto byte_buffer = (uint8_t*)mbuf_info[core_index*PACKET_RX_BUFFER_COUNT*MAX_PACKET_PER_RX_BURST+buf_read_idx*MAX_PACKET_PER_RX_BURST+i].buffer;
                if (opt_dl_up_sanity_check == RE_ENABLED)
                {
                    bool res = uplane_pkt_sanity_check(byte_buffer, cell_configs[cell_index].dl_bit_width, (int)cell_configs[cell_index].dl_comp_meth);
                    if (!res)
                    {
                        do_throw(sb() << "DL U Plane pkt sanity check failed, it could be erroneous BFP, numPrb or ecpri payload len, or other reasons... ");
                    }
                }

                header_info.fss.frameId = oran_cmsg_get_frame_id(byte_buffer);
                header_info.fss.subframeId = oran_cmsg_get_subframe_id(byte_buffer);
                header_info.fss.slotId = oran_cmsg_get_slot_id(byte_buffer);
                header_info.launch_pattern_slot = fss_to_launch_pattern_slot(header_info.fss, launch_pattern_slot_size);
                curr_launch_pattern_slot = header_info.launch_pattern_slot;
                header_info.symbolId = oran_cmsg_get_startsymbol_id(byte_buffer);
                header_info.flowValue = oran_msg_get_flowid(byte_buffer);
                header_info.sectionId = oran_umsg_get_section_id(byte_buffer);
                header_info.rb = oran_umsg_get_rb(byte_buffer);
                header_info.startPrb = oran_umsg_get_start_prb(byte_buffer);
                header_info.numPrb = oran_umsg_get_num_prb(byte_buffer);
                header_info.payload_len = oran_umsg_get_ecpri_payload(byte_buffer);
                {
                    int res=find_eAxC_index_from_flowVal(cell_configs[cell_index],header_info.flowValue,header_info.flow_index);
                    if (res<0)
                    {
                        do_throw(sb() << "Flow index not found from Flow value... ");
                    }
                }

                if (unlikely(opt_sectionid_validation == RE_ENABLED))
                {
                    validate_uplane_section_id_match(cell_index, header_info);
                }

                if(first_packet_of_slot_rx_burst[header_info.flow_index]==false)
                {
                    if(curr_launch_pattern_slot != prev_launch_pattern_slot[header_info.flow_index])
                    {
                        re_info("[Inside uplane_proc_validate_core loop] Cell {}-{} Flow index {}: 3GPP {} F{} S{} S{} Validate {} packets complete in Total Time {:4.2f} us t0 {} curr time {} start time {} end time {}",
                                 cell_index, core_index,header_info.flow_index,prev_launch_pattern_slot[header_info.flow_index],
                                 prev_fss[header_info.flow_index].frameId,prev_fss[header_info.flow_index].subframeId,prev_fss[header_info.flow_index].slotId,
                                    packet_count_slot[header_info.flow_index],(float)(slot_validate_end_t[header_info.flow_index] - slot_validate_start_t[header_info.flow_index])/1000,
                                    ((prev_t0[header_info.flow_index] / (opt_tti_us * NS_X_US)) * opt_tti_us * NS_X_US),get_ns(),slot_validate_start_t[header_info.flow_index],slot_validate_end_t[header_info.flow_index]);
                        packet_count_slot[header_info.flow_index]=0;
                        first_packet_of_slot_rx_burst[header_info.flow_index]=true;
                    }
                    else
                    {
                        packet_count_slot[header_info.flow_index]++;
                        slot_validate_end_t[header_info.flow_index]=get_ns();
                    }
                }


                if(first_packet_of_slot_rx_burst[header_info.flow_index]==true)
                {
                   first_packet_of_slot_rx_burst[header_info.flow_index]=false;
                   slot_validate_start_t[header_info.flow_index]=get_ns();
                   packet_count_slot[header_info.flow_index]++;
                   prev_launch_pattern_slot[header_info.flow_index]=header_info.launch_pattern_slot;
                   prev_fss[header_info.flow_index].frameId=header_info.fss.frameId;
                   prev_fss[header_info.flow_index].subframeId=header_info.fss.subframeId;
                   prev_fss[header_info.flow_index].slotId=header_info.fss.slotId;
                }

                if(header_info.symbolId >= ORAN_ALL_SYMBOLS)
                {
                    do_throw(sb() << "U Plane pkt startSym exceeds 14... ");
                }

                if(header_info.numPrb == 0)
                {
                    header_info.numPrb = cell_configs[cell_index].dlGridSize;
                }
                re_dbg("Cell {} Core {} F{}S{}S{} symId {} Flow {} packet_flow_index {} queue_flow_index {} startPrb {} numPrb {} first_packet_of_slot_rx_burst[header_info.flow_index] {} prev_launch_pattern_slot[header_info.flow_index] {} curr_launch_pattern_slot {}",
                cell_index,core_index,header_info.fss.frameId, header_info.fss.subframeId, header_info.fss.slotId, header_info.symbolId, header_info.flowValue, header_info.flow_index, flow_counter, header_info.startPrb, header_info.numPrb,first_packet_of_slot_rx_burst[header_info.flow_index],prev_launch_pattern_slot[header_info.flow_index],curr_launch_pattern_slot);

                //Check if packet is coming out of slot boundaries
                if(opt_validate_dl_timing == RE_ENABLED)
                {
                    packet_time = mbuf_info[core_index*PACKET_RX_BUFFER_COUNT*MAX_PACKET_PER_RX_BURST+buf_read_idx*MAX_PACKET_PER_RX_BURST+i].rx_timestamp;

                    // Use the calculate_t0_toa function instead of inline timing calculation
                    t0_toa_result timing_result = calculate_t0_toa(
                        packet_time,
                        first_f0s0s0_time,
                        frame_cycle_time_ns,
                        header_info.fss.frameId,
                        header_info.fss.subframeId,
                        header_info.fss.slotId,
                        header_info.symbolId,
                        max_slot_id,
                        opt_tti_us
                    );

                    slot_t0 = timing_result.slot_t0;
                    toa = timing_result.toa;
                    // re_dbg("F{} S{} S{} Sym {} U Packet time {} - t0 {} = {} TOA {:5d} time_offset {}", header_info.fss.frameId, header_info.fss.subframeId, header_info.fss.slotId, header_info.symbolId, packet_time, t0, (int64_t)(packet_time - t0), toa, time_offset);
                    auto& packet_timer = oran_packet_slot_timers.timers[DLPacketCounterType::DLU][cell_index][header_info.launch_pattern_slot];
                    {
                        const std::lock_guard<aerial_fh::FHMutex> lock(packet_timer.mtx);
                        if (packet_timer.fss.frameId != header_info.fss.frameId && packet_timer.first_packet != true)
                        {
                            // Print previous slot info
                            flush_packet_timers(oran_pkt_dir::DIRECTION_DOWNLINK, ECPRI_MSG_TYPE_IQ, cell_index, packet_timer);
                            increment_oran_packet_counters(rx_packet_type::DL_U_PLANE, cell_index, packet_timer, curr_launch_pattern_slot);

                            packet_timer.fss.frameId = header_info.fss.frameId;
                            packet_timer.fss.subframeId = header_info.fss.subframeId;
                            packet_timer.fss.slotId = header_info.fss.slotId;
                            packet_timer.reset();
                        }

                        if (packet_timer.first_packet == true)
                        {
                            packet_timer.first_packet = false;
                            packet_timer.fss.frameId = header_info.fss.frameId;
                            packet_timer.fss.subframeId = header_info.fss.subframeId;
                            packet_timer.fss.slotId = header_info.fss.slotId;
                            packet_timer.reset();
                        }

                        if (packet_timer.earliest_packet_per_slot > packet_time)
                        {
                            packet_timer.earliest_packet_per_slot = packet_time;
                            packet_timer.earliest_packet_symbol_num = header_info.symbolId;
                        }

                        if (packet_timer.earliest_packet_per_symbol[header_info.symbolId] > packet_time)
                        {
                            packet_timer.earliest_packet_per_symbol[header_info.symbolId] = packet_time;
                        }

                        if (packet_timer.latest_packet_per_slot < packet_time)
                        {
                            packet_timer.latest_packet_per_slot = packet_time;
                            packet_timer.latest_packet_symbol_num = header_info.symbolId;
                        }

                        if (packet_timer.latest_packet_per_symbol[header_info.symbolId] < packet_time)
                        {
                            packet_timer.latest_packet_per_symbol[header_info.symbolId] = packet_time;
                        }

                        ++packet_timer.packet_count;
                        packet_timer.max_toa = (packet_timer.max_toa < toa) ? toa : packet_timer.max_toa;
                        packet_timer.min_toa = (packet_timer.min_toa > toa) ? toa : packet_timer.min_toa;
                        packet_timer.t0 = slot_t0;
                        prev_t0[header_info.flow_index]=slot_t0;

                        if (opt_debug_u_plane_prints != 0)
                        {
                            packet_timer.packet_arrive_abs[header_info.symbolId][packet_timer.num_packets_per_symbol[header_info.symbolId]] = packet_time;
                            packet_timer.packet_arrive_t0s[header_info.symbolId][packet_timer.num_packets_per_symbol[header_info.symbolId]] = toa;
                        }

                        ++packet_timer.num_packets_per_symbol[header_info.symbolId];

                        if (toa < -((int64_t)oran_timing_info.dl_u_plane_timing_delay * NS_X_US))
                        {
                            ++oran_packet_counters.dl_u_plane[cell_index].early_packet;
                            ++packet_timer.early;
                            ++local_packet_counters[cell_index][curr_launch_pattern_slot][PacketCounterTiming::EARLY];
                        }
                        else if (toa > -((int64_t)oran_timing_info.dl_u_plane_timing_delay * NS_X_US) + oran_timing_info.dl_u_plane_window_size * NS_X_US)
                        {
                            ++oran_packet_counters.dl_u_plane[cell_index].late_packet;
                            ++packet_timer.late;
                            ++packet_timer.late_packets_per_symbol[header_info.symbolId];
                            ++local_packet_counters[cell_index][curr_launch_pattern_slot][PacketCounterTiming::LATE];
                        }
                        else
                        {
                            ++oran_packet_counters.dl_u_plane[cell_index].ontime_packet;
                            ++packet_timer.ontime;
                            ++local_packet_counters[cell_index][curr_launch_pattern_slot][PacketCounterTiming::ONTIME];
                        }
                    }

                    if(opt_timing_histogram == RE_ENABLED)
                    {
                        toa += STATS_MAX_BINS/2 * static_cast<int64_t>(opt_timing_histogram_bin_size);

                        if(toa < 0)
                        {
                            toa *= -1;
                            toa /= opt_timing_histogram_bin_size;
                            toa *= -1;
                        }
                        else
                        {
                            toa /= opt_timing_histogram_bin_size;
                        }

                        if(toa > STATS_MAX_BINS - 1)
                        {
                            toa = STATS_MAX_BINS - 1;
                        }
                        if(toa < 0)
                        {
                            toa = 0;
                        }
                        ++timing_bins[toa];
                    }
                }

                int section_buf_offset = ORAN_IQ_STATIC_OVERHEAD;
                // - 4 Ecpri payload already includes ecpriRtcid / ecpriPcid and ecpriSeqid
                int packet_len = header_info.payload_len + (ORAN_IQ_HDR_OFFSET - 4);
                while(section_buf_offset < packet_len) {
                    auto section_buffer = static_cast<uint8_t*>(&byte_buffer[section_buf_offset]);
                    header_info.startPrb = oran_umsg_get_start_prb_from_section_buf(section_buffer);
                    header_info.numPrb = oran_umsg_get_num_prb_from_section_buf(section_buffer);
                    if(header_info.numPrb == 0)
                    {
                        header_info.numPrb = cell_configs[cell_index].dlGridSize;
                    }

                    validate_dl_channels(cell_index, curr_launch_pattern_slot, header_info, section_buffer, prev_pbch_time);

                    section_buf_offset += ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + static_cast<size_t>(header_info.numPrb) * cell_configs[cell_index].dl_prb_size;
                }
            }
            ret = aerial_fh::free_rx_messages(&mbuf_info[core_index*PACKET_RX_BUFFER_COUNT*MAX_PACKET_PER_RX_BURST+buf_read_idx*MAX_PACKET_PER_RX_BURST],nb_mbufs_rx);
            if ( ret != 0)
            {
                do_throw(sb() <<"Error in FH free_rx_messages\n");
            }
            if((buf_read_idx+1)==PACKET_RX_BUFFER_COUNT)
            {
                wrap_index+=1;
            }
            buf_read_idx=(buf_read_idx+1)%PACKET_RX_BUFFER_COUNT;
            pkt_buf_read_idx[core_index]=buf_read_idx;
        }
        validate_end_t = get_ns();
        if (unlikely(nb_mbufs_rx == 0))
        {
            re_warn("Unexpected: nb_mbufs_rx is 0");
        }
        else
        {
            re_dbg("Cell {}-{}: 3GPP {} F{} S{} S{} Validate {} packets complete Avg Time {:4.2f} us per pkt Total Time {:4.2f} us t0 {} curr time {}",
                   cell_index, core_index, header_info.launch_pattern_slot, header_info.fss.frameId, header_info.fss.subframeId, header_info.fss.slotId, nb_mbufs_rx, (float)(validate_end_t - validate_start_t) / 1000 / nb_mbufs_rx, (float)(validate_end_t - validate_start_t) / 1000, slot_t0, get_ns());
        }
        if(opt_forever == RE_DISABLED)
        {
            //Assuming all slots have the same number of cells
            if(pdsch_object.total_slot_counters[0].load() >= opt_num_slots_dl)
            {
                break;
            }
        }
    }
    if(ul_slot_counters[0].load() >= opt_num_slots_ul)
    {
        set_force_quit();
    }

    for(int cell_index  = 0; cell_index < MAX_CELLS_PER_SLOT; ++cell_index)
    {
        for(int slot_index = 0; slot_index < MAX_LAUNCH_PATTERN_SLOTS; ++slot_index)
        {
            for(int counter = 0; counter < CounterTimingMax; ++counter)
            {
                if(local_packet_counters[cell_index][slot_index][counter] > 0)
                {
                    dl_u_packet_stats.increment_counters(cell_index, counter, slot_index, local_packet_counters[cell_index][slot_index][counter]);
                }
            }
        }
    }

    re_cons("Thread {} exiting", threadname);
    re_info("Cell {}-{} Exiting after {} DL slots\n", cell_index, core_index, pdsch_object.total_slot_counters[0].load());
    usleep(2000000);
    return NULL;
}

void *uplane_proc_core_wrapper(void *arg)
{
    if (!arg) {
        do_throw(sb() << "Error: arg == nullptr with uplane_proc_core_wrapper");
    }
    auto* dl_core_info = reinterpret_cast<struct RU_Emulator::dl_core_info *>(arg);
    if (!dl_core_info->rue) {
        do_throw(sb() << "Error: rue == nullptr with uplane_proc_core_wrapper");
    }
    RU_Emulator *rue = static_cast<RU_Emulator*>(dl_core_info->rue);
    return rue->uplane_proc_core(arg);
}

void *RU_Emulator::uplane_proc_core(void *arg)
{
    nvlog_fmtlog_thread_init();
    struct dl_core_info dl_core_info = *(struct dl_core_info *)arg;
    int core_index = dl_core_info.core_index;
    int cell_index = 0;
    int flow_index = 0;
    char threadname[30];
    sprintf(threadname, "%s-%d", "uplane_proc", dl_core_info.core_index);
    SET_THREAD_NAME(threadname);
    int flow_count = std::max(1, (int)dl_core_info.flow_count);
    re_cons("Thread {} started on CPU {} handling {} flows", threadname, core_index, flow_count);

    size_t nb_mbufs_rx = 0;
    uint64_t slot_t0;
    uint64_t packet_time = 0;

    uint64_t validate_start_t = 0;
    uint64_t validate_end_t = 0;
    int64_t toa;
    struct oran_packet_header_info header_info;
    uint64_t prev_pbch_time;
    uint8_t flow_counter = 0;
    uint8_t* buffer;

    uint8_t curr_launch_pattern_slot = 0;
    struct fssId curr_fss{0,0,0};
    std::array<std::array<std::array<uint64_t, CounterTimingMax>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> local_packet_counters{};

    //Profiling variables : Only use for accurate interpretation if all the flows of a cell are mapped to the same CPU core
    bool first_packet_of_slot_rx_burst[MAX_FLOWS_PER_DL_CORE];
    uint64_t slot_validate_start_t[MAX_FLOWS_PER_DL_CORE];
    uint64_t slot_validate_end_t[MAX_FLOWS_PER_DL_CORE];
    size_t packet_count_slot[MAX_FLOWS_PER_DL_CORE];
    uint8_t prev_launch_pattern_slot[MAX_FLOWS_PER_DL_CORE];
    struct fssId prev_fss[MAX_FLOWS_PER_DL_CORE];
    uint64_t prev_t0[MAX_FLOWS_PER_DL_CORE];

    //Init peofiling variables
    for (int i = 0; i < MAX_FLOWS_PER_DL_CORE; i++)
    {
        first_packet_of_slot_rx_burst[i] = true;
        slot_validate_start_t[i] = 0;
        slot_validate_end_t[i] = 0;
        packet_count_slot[i] = 0;
        prev_t0[i] = 0;
        prev_launch_pattern_slot[i] = MAX_LAUNCH_PATTERN_SLOTS;
        prev_fss[i] = {0,0,0};
    }

    // NOT SURE WHY THIS DOESNT WORK, have to split up the multiplication.
    // int64_t frame_cycle_time_ns = ORAN_MAX_FRAME_ID * ORAN_MAX_SUBFRAME_ID * (max_slot_id + 1) * opt_tti_us * NS_X_US;

    int64_t frame_cycle_time_ns = get_frame_cycle_time_ns(max_slot_id, opt_tti_us);
    int64_t first_f0s0s0_time = get_first_f0s0s0_time();

    aerial_fh::MsgReceiveInfo info[MAX_PACKET_PER_RX_BURST]{};

    while(!check_force_quit())
    {
        cell_index = dl_core_info.flow_infos[flow_counter].cell_index;
        flow_index = dl_core_info.flow_infos[flow_counter].flowId;
        nb_mbufs_rx = MAX_PACKET_PER_RX_BURST;
        int ret = aerial_fh::receive_flow(dl_peer_flow_map[cell_index][flow_index], &info[0], &nb_mbufs_rx);

        if ( ret != 0){
            do_throw(sb() <<"PEER: "<<cell_index<<" Could not receive data on flow \n"<<flow_counter<<"\n");
        }

        if(nb_mbufs_rx == 0)
        {
            ++flow_counter;
            if(flow_counter == flow_count)
            {
                flow_counter = 0;
            }
            usleep(10); // Busy spinning has caused other stalls in the system when running with other applications.
            // Issue can be reproduced by running the loopback test, DU cores will see "random" stalls when the usleep is commented out.
            continue;
        }

        validate_start_t = get_ns();
        for(int i = 0; i < nb_mbufs_rx; ++i)
        {
            auto byte_buffer = (uint8_t*)info[i].buffer;
            header_info.fss.frameId = oran_cmsg_get_frame_id(byte_buffer);
            header_info.fss.subframeId = oran_cmsg_get_subframe_id(byte_buffer);
            header_info.fss.slotId = oran_cmsg_get_slot_id(byte_buffer);
            header_info.launch_pattern_slot = fss_to_launch_pattern_slot(header_info.fss, launch_pattern_slot_size);
            curr_launch_pattern_slot = header_info.launch_pattern_slot;
            header_info.symbolId = oran_cmsg_get_startsymbol_id(byte_buffer);
            header_info.flowValue = oran_msg_get_flowid(byte_buffer);
            header_info.sectionId = oran_umsg_get_section_id(byte_buffer);
            header_info.rb = oran_umsg_get_rb(byte_buffer);
            header_info.startPrb = oran_umsg_get_start_prb(byte_buffer);
            header_info.numPrb = oran_umsg_get_num_prb(byte_buffer);
            header_info.payload_len = oran_umsg_get_ecpri_payload(byte_buffer);
            {
                int res=find_eAxC_index_from_flowVal(cell_configs[cell_index],header_info.flowValue,header_info.flow_index);
                if (res<0)
                {
                    do_throw(sb() << "Flow index not found from Flow value... ");
                }
            }

            if (unlikely(opt_sectionid_validation == RE_ENABLED))
            {
                validate_uplane_section_id_match(cell_index, header_info);
            }

            if(first_packet_of_slot_rx_burst[header_info.flow_index]==false)
            {
                if(curr_launch_pattern_slot != prev_launch_pattern_slot[header_info.flow_index])
                {
                    re_info("[Inside uplane_proc_core loop] Cell {}-{} Flow index {}: 3GPP {} F{} S{} S{} Validate {} packets complete in Total Time {:4.2f} us t0 {} curr time {} start time {} end time {}",
                             cell_index, core_index,header_info.flow_index,prev_launch_pattern_slot[header_info.flow_index],
                             prev_fss[header_info.flow_index].frameId,prev_fss[header_info.flow_index].subframeId,prev_fss[header_info.flow_index].slotId,
                             packet_count_slot[header_info.flow_index],(float)(slot_validate_end_t[header_info.flow_index] - slot_validate_start_t[header_info.flow_index])/1000,
                             ((prev_t0[header_info.flow_index] / (opt_tti_us * NS_X_US)) * opt_tti_us * NS_X_US),get_ns(),slot_validate_start_t[header_info.flow_index],slot_validate_end_t[header_info.flow_index]);
                    packet_count_slot[header_info.flow_index]=0;
                    first_packet_of_slot_rx_burst[header_info.flow_index]=true;
                }
                else
                {
                    packet_count_slot[header_info.flow_index]++;
                    slot_validate_end_t[header_info.flow_index]=get_ns();
                }
            }


            if(first_packet_of_slot_rx_burst[header_info.flow_index]==true)
            {
               first_packet_of_slot_rx_burst[header_info.flow_index]=false;
               slot_validate_start_t[header_info.flow_index]=get_ns();
               packet_count_slot[header_info.flow_index]++;
               prev_launch_pattern_slot[header_info.flow_index]=header_info.launch_pattern_slot;
               prev_fss[header_info.flow_index].frameId=header_info.fss.frameId;
               prev_fss[header_info.flow_index].subframeId=header_info.fss.subframeId;
               prev_fss[header_info.flow_index].slotId=header_info.fss.slotId;
            }

            if(header_info.symbolId >= ORAN_ALL_SYMBOLS)
            {
                do_throw(sb() << "U Plane pkt startSym exceeds 14... ");
            }

            if(header_info.numPrb == 0)
            {
                header_info.numPrb = cell_configs[cell_index].dlGridSize;
            }
            re_dbg("Cell {} Core {} Mbuf {} F{}S{}S{} symId {} Flow {} packet_flow_index {} queue_flow_index {} startPrb {} numPrb {} first_packet_of_slot_rx_burst[header_info.flow_index] {} prev_launch_pattern_slot[header_info.flow_index] {} curr_launch_pattern_slot {}",
            cell_index,core_index,reinterpret_cast<void*>(&info[i]), header_info.fss.frameId, header_info.fss.subframeId, header_info.fss.slotId, header_info.symbolId, header_info.flowValue, header_info.flow_index, flow_counter, header_info.startPrb, header_info.numPrb,first_packet_of_slot_rx_burst[header_info.flow_index],prev_launch_pattern_slot[header_info.flow_index],curr_launch_pattern_slot);

            //Check if packet is coming out of slot boundaries
            if(opt_validate_dl_timing == RE_ENABLED)
            {
                packet_time = info[i].rx_timestamp;

                // Use the calculate_t0_toa function instead of inline timing calculation
                t0_toa_result timing_result = calculate_t0_toa(
                    packet_time,
                    first_f0s0s0_time,
                    frame_cycle_time_ns,
                    header_info.fss.frameId,
                    header_info.fss.subframeId,
                    header_info.fss.slotId,
                    header_info.symbolId,
                    max_slot_id,
                    opt_tti_us
                );

                slot_t0 = timing_result.slot_t0;
                toa = timing_result.toa;
                // re_dbg("F{} S{} S{} Sym {} U Packet time {} - t0 {} = {} TOA {:5d} time_offset {}", header_info.fss.frameId, header_info.fss.subframeId, header_info.fss.slotId, header_info.symbolId, packet_time, t0, (int64_t)(packet_time - t0), toa, time_offset);
                auto& packet_timer = oran_packet_slot_timers.timers[DLPacketCounterType::DLU][cell_index][header_info.launch_pattern_slot];
                {
                    const std::lock_guard<aerial_fh::FHMutex> lock(packet_timer.mtx);

                    // packet_timer is per packet type, cell, slot in the launch pattern (currently based on 80 for our performance patterns)
                    // when the packet_timer is reused, it means it is from the same slot in the launch pattern (4ms apart).
                    // the below condition checks if it is the same frame ID, frame ID wraps around 255, so it is much longer than our launch patterns (so currently this condition stands until we have patterns longer than 5120 slots)
                    // In the condition, check if the counters for the slot are non-zero, each channel also tracks received PRB counts for that launch pattern slot
                    // If the counters are non-zero, it means that the channel did not complete, having residual PRBs
                    // As a heuristic, assume worst case of maximum PRB allocation for the slot, use the difference from the residual PRBs to calculate an approximate number of missing packets
                    // Note that it will be an overestimate because of how DU side creates packet sections, section header overhead, non-maximum allocation.
                    if (packet_timer.fss.frameId != header_info.fss.frameId && packet_timer.first_packet != true)
                    {
                        int residual_prbs = 0;
                        residual_prbs += pbch_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].load();
                        residual_prbs += pdsch_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].load();
                        residual_prbs += pdcch_dl_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].load();
                        residual_prbs += pdcch_ul_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].load();
                        residual_prbs += csirs_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].load();
                        if(residual_prbs > 0)
                        {
                            // Heuristic based on MTU/PRB size for compression method
                            int missing_prbs = cell_configs[cell_index].eAxC_DL.size() * ORAN_MAX_PRB_X_SLOT * ORAN_ALL_SYMBOLS - residual_prbs;
                            int missing_pkts = missing_prbs * cell_configs[cell_index].dl_prb_size / (opt_afh_mtu - ORAN_IQ_HDR_OFFSET);
                            re_info("Residual PRBs detected, incomplete slot received F{}S{}S{} Cell {} residual prbs {} missing pkts heuristic {}", packet_timer.fss.frameId, packet_timer.fss.subframeId, packet_timer.fss.slotId, cell_index, residual_prbs, missing_pkts);
                            // Cannot determine how many packets are missing due to DU packet creation
                            // Assume worst case 273 * 14 for all symbols all antennas
                            pbch_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].store(0);
                            pdsch_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].store(0);
                            pdcch_dl_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].store(0);
                            pdcch_ul_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].store(0);
                            csirs_object.atomic_received_prbs[header_info.launch_pattern_slot][cell_index].store(0);
                            oran_packet_counters.dl_u_plane[cell_index].late_packet += missing_pkts;
                        }
                    }

                    auto cur_t = get_ns();
                    if (cur_t - csirs_object.fss_atomic_received_res_prev_ts[cell_index][header_info.fss.frameId][header_info.fss.subframeId * ORAN_MAX_SLOT_ID + header_info.fss.slotId] > 10000000)
                    {
                        csirs_object.fss_atomic_received_res[cell_index][header_info.fss.frameId][header_info.fss.subframeId * ORAN_MAX_SLOT_ID + header_info.fss.slotId].store(0);
                    }
                    csirs_object.fss_atomic_received_res_prev_ts[cell_index][header_info.fss.frameId][header_info.fss.subframeId * ORAN_MAX_SLOT_ID + header_info.fss.slotId] = cur_t;

                    if (packet_timer.fss.frameId != header_info.fss.frameId && packet_timer.first_packet != true)
                    {
                        // Print previous slot info
                        flush_packet_timers(oran_pkt_dir::DIRECTION_DOWNLINK, ECPRI_MSG_TYPE_IQ, cell_index, packet_timer);
                        increment_oran_packet_counters(rx_packet_type::DL_U_PLANE, cell_index, packet_timer, curr_launch_pattern_slot);

                        packet_timer.fss.frameId = header_info.fss.frameId;
                        packet_timer.fss.subframeId = header_info.fss.subframeId;
                        packet_timer.fss.slotId = header_info.fss.slotId;
                        packet_timer.reset();
                    }

                    if (packet_timer.first_packet == true)
                    {
                        packet_timer.first_packet = false;
                        packet_timer.fss.frameId = header_info.fss.frameId;
                        packet_timer.fss.subframeId = header_info.fss.subframeId;
                        packet_timer.fss.slotId = header_info.fss.slotId;
                        packet_timer.reset();
                    }

                    if (packet_timer.earliest_packet_per_slot > packet_time)
                    {
                        packet_timer.earliest_packet_per_slot = packet_time;
                        packet_timer.earliest_packet_symbol_num = header_info.symbolId;
                    }

                    if (packet_timer.earliest_packet_per_symbol[header_info.symbolId] > packet_time)
                    {
                        packet_timer.earliest_packet_per_symbol[header_info.symbolId] = packet_time;
                    }

                    if (packet_timer.latest_packet_per_slot < packet_time)
                    {
                        packet_timer.latest_packet_per_slot = packet_time;
                        packet_timer.latest_packet_symbol_num = header_info.symbolId;
                    }

                    if (packet_timer.latest_packet_per_symbol[header_info.symbolId] < packet_time)
                    {
                        packet_timer.latest_packet_per_symbol[header_info.symbolId] = packet_time;
                    }

                    ++packet_timer.packet_count;
                    packet_timer.max_toa = (packet_timer.max_toa < toa) ? toa : packet_timer.max_toa;
                    packet_timer.min_toa = (packet_timer.min_toa > toa) ? toa : packet_timer.min_toa;
                    packet_timer.t0 = slot_t0;
                    prev_t0[header_info.flow_index]=slot_t0;

                    if (opt_debug_u_plane_prints != 0)
                    {
                        packet_timer.packet_arrive_abs[header_info.symbolId][packet_timer.num_packets_per_symbol[header_info.symbolId]] = packet_time;
                        packet_timer.packet_arrive_t0s[header_info.symbolId][packet_timer.num_packets_per_symbol[header_info.symbolId]] = toa;
                    }

                    ++packet_timer.num_packets_per_symbol[header_info.symbolId];

                    if (toa < -((int64_t)oran_timing_info.dl_u_plane_timing_delay * NS_X_US))
                    {
                        ++oran_packet_counters.dl_u_plane[cell_index].early_packet;
                        ++packet_timer.early;
                        ++local_packet_counters[cell_index][curr_launch_pattern_slot][PacketCounterTiming::EARLY];
                        // dl_u_packet_stats.increment_counters(cell_index, PacketCounterTiming::EARLY, curr_launch_pattern_slot, 1);
                    }
                    else if (toa > -((int64_t)oran_timing_info.dl_u_plane_timing_delay * NS_X_US) + oran_timing_info.dl_u_plane_window_size * NS_X_US)
                    {
                        ++oran_packet_counters.dl_u_plane[cell_index].late_packet;
                        ++packet_timer.late;
                        ++packet_timer.late_packets_per_symbol[header_info.symbolId];
                        ++local_packet_counters[cell_index][curr_launch_pattern_slot][PacketCounterTiming::LATE];
                        // dl_u_packet_stats.increment_counters(cell_index, PacketCounterTiming::LATE, curr_launch_pattern_slot, 1);
                    }
                    else
                    {
                        ++oran_packet_counters.dl_u_plane[cell_index].ontime_packet;
                        ++packet_timer.ontime;
                        ++local_packet_counters[cell_index][curr_launch_pattern_slot][PacketCounterTiming::ONTIME];
                        // dl_u_packet_stats.increment_counters(cell_index, PacketCounterTiming::ONTIME, curr_launch_pattern_slot, 1);
                    }
                }

                if(opt_timing_histogram == RE_ENABLED)
                {
                    toa += STATS_MAX_BINS/2 * static_cast<int64_t>(opt_timing_histogram_bin_size);

                    if(toa < 0)
                    {
                        toa *= -1;
                        toa /= opt_timing_histogram_bin_size;
                        toa *= -1;
                    }
                    else
                    {
                        toa /= opt_timing_histogram_bin_size;
                    }

                    if(toa > STATS_MAX_BINS - 1)
                    {
                        toa = STATS_MAX_BINS - 1;
                    }
                    if(toa < 0)
                    {
                        toa = 0;
                    }
                    ++timing_bins[toa];
                }
            }

            if (opt_dl_up_sanity_check == RE_ENABLED)
            {
                bool res = uplane_pkt_sanity_check(byte_buffer, cell_configs[cell_index].dl_bit_width, (int)cell_configs[cell_index].dl_comp_meth);
                if (!res)
                {
                    re_cons("Cell {}-{}: 3GPP {} F{} S{} S{} symbol {}", cell_index, core_index, header_info.launch_pattern_slot, header_info.fss.frameId, header_info.fss.subframeId, header_info.fss.slotId, header_info.symbolId);
                    sleep(1);
                    do_throw(sb() << "DL U Plane pkt sanity check failed, it could be erroneous BFP, numPrb, udCompHdr or ecpri payload len, or other reasons... ");
                }
            }

            int section_buf_offset = ORAN_IQ_STATIC_OVERHEAD;
            // - 4 Ecpri payload already includes ecpriRtcid / ecpriPcid and ecpriSeqid
            int packet_len = header_info.payload_len + (ORAN_IQ_HDR_OFFSET - 4);
            while(section_buf_offset < packet_len) {
                auto section_buffer = static_cast<uint8_t*>(&byte_buffer[section_buf_offset]);
                header_info.startPrb = oran_umsg_get_start_prb_from_section_buf(section_buffer);
                header_info.numPrb = oran_umsg_get_num_prb_from_section_buf(section_buffer);
                if(header_info.numPrb == 0)
                {
                    header_info.numPrb = cell_configs[cell_index].dlGridSize;
                }

                int sect_iq_data_sz = 0;
                if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
                {
                    auto com_meth = oran_umsg_get_com_meth_from_section_buf(section_buffer);
                    auto iq_width = oran_umsg_get_iq_width_from_section_buf(section_buffer);
                    auto reserved_bits = oran_umsg_get_comp_hdr_reserved_bits_from_section_buf(section_buffer);

                    if (unlikely(static_cast<aerial_fh::UserDataCompressionMethod>(com_meth) != cell_configs[cell_index].dl_comp_meth || reserved_bits))
                    {
                        re_cons("Cell {}-{}: 3GPP {} F{} S{} S{} symbol {} com_meth {} iq_width {} reserved_bits {}", cell_index, core_index, header_info.launch_pattern_slot, header_info.fss.frameId, header_info.fss.subframeId, header_info.fss.slotId, header_info.symbolId, com_meth, iq_width, reserved_bits);
                        sleep(1);
                        do_throw(sb() << "Wrong udCompHdr...\n");
                    }
                    sect_iq_data_sz = static_cast<size_t>(header_info.numPrb) * iq_width * 3;
                }
                else
                {
                    sect_iq_data_sz = static_cast<size_t>(header_info.numPrb) * cell_configs[cell_index].dl_prb_size;
                }

                validate_dl_channels(cell_index, curr_launch_pattern_slot, header_info, section_buffer, prev_pbch_time);

                if (cell_configs[cell_index].dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
                {
                    section_buf_offset += ORAN_IQ_COMPRESSED_SECTION_OVERHEAD + sect_iq_data_sz;
                }
                else
                {
                    section_buf_offset += ORAN_IQ_UNCOMPRESSED_SECTION_OVERHEAD + sect_iq_data_sz;
                }
            }
        }
        validate_end_t = get_ns();
        if (unlikely(nb_mbufs_rx == 0))
        {
            re_warn("Unexpected: nb_mbufs_rx is 0");
        }
        else
        {
            ret = aerial_fh::free_rx_messages(&info[0],nb_mbufs_rx);
            if ( ret != 0)
            {
                do_throw(sb() <<"Error in FH free_rx_messages\n");
            }
            re_dbg("Cell {}-{}: 3GPP {} F{} S{} S{} Validate {} packets complete Avg Time {:4.2f} us per pkt Total Time {:4.2f} us t0 {} curr time {}",
                   cell_index, core_index, header_info.launch_pattern_slot, header_info.fss.frameId, header_info.fss.subframeId, header_info.fss.slotId, nb_mbufs_rx, (float)(validate_end_t - validate_start_t) / 1000 / nb_mbufs_rx, (float)(validate_end_t - validate_start_t) / 1000, slot_t0, get_ns());
        }
        ++flow_counter;
        if(flow_counter == flow_count)
        {
            flow_counter = 0;
        }
        if(opt_forever == RE_DISABLED)
        {
            //Assuming all slots have the same number of cells
            if(pdsch_object.total_slot_counters[0].load() >= opt_num_slots_dl)
            {
                break;
            }
        }
    }
    if(ul_slot_counters[0].load() >= opt_num_slots_ul)
    {
        set_force_quit();
    }

    for(int cell_index  = 0; cell_index < MAX_CELLS_PER_SLOT; ++cell_index)
    {
        for(int slot_index = 0; slot_index < MAX_LAUNCH_PATTERN_SLOTS; ++slot_index)
        {
            for(int counter = 0; counter < CounterTimingMax; ++counter)
            {
                if(local_packet_counters[cell_index][slot_index][counter] > 0)
                {
                    dl_u_packet_stats.increment_counters(cell_index, counter, slot_index, local_packet_counters[cell_index][slot_index][counter]);
                }
            }
        }
    }
    re_cons("Thread {} exiting", threadname);
    re_info("Cell {}-{} Exiting after {} DL slots\n", cell_index, core_index, pdsch_object.total_slot_counters[0].load());
    usleep(2000000);
    return NULL;
}
