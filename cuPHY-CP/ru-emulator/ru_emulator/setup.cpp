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

void RU_Emulator::setup_slots()
{
    for(int i = 0; i < opt_num_cells; ++i)
    {
        ecpriSeqid_vectormap.emplace_back(std::map<uint16_t, uint8_t>());
    }

    pusch_object.channel_string = "PUSCH";
    prach_object.channel_string = "PRACH";
    pucch_object.channel_string = "PUCCH";
    pucch_object.channel_string = "SRS";
    pdsch_object.channel_string = "PDSCH";
    pbch_object.channel_string = "PBCH";
    pdcch_ul_object.channel_string = "PDCCH_UL";
    pdcch_dl_object.channel_string = "PDCCH_DL";
    csirs_object.channel_string = "CSI_RS";

    setup_ul_slots();
    setup_dl_slots();
}

void RU_Emulator::setup_ul_slots()
{
    pucch_object.channel_type = ul_channel::PUCCH;
    srs_object.channel_type = ul_channel::SRS;
    pusch_object.channel_type = ul_channel::PUSCH;
    prach_object.channel_type = ul_channel::PRACH;

    for(int i = 0; i < MAX_CELLS_PER_SLOT; ++i)
    {
        ul_slot_counters[i].store(0);
    }

    setup_ul_counters(pusch_object);
    setup_ul_counters(prach_object);
    setup_ul_counters(pucch_object);
    setup_ul_counters(srs_object);
}

void RU_Emulator::setup_ul_counters(struct ul_tv_object& tv_object)
{
    for(int cell_idx = 0; cell_idx < MAX_CELLS_PER_SLOT; ++cell_idx)
    {
        tv_object.throughput_counters[cell_idx].store(0);
        tv_object.throughput_slot_counters[cell_idx].store(0);
        tv_object.good_slot_counters[cell_idx].store(0);
        tv_object.error_slot_counters[cell_idx].store(0);
        tv_object.total_slot_counters[cell_idx].store(0);

        tv_object.c_plane_rx[cell_idx].store(0);
        tv_object.c_plane_rx_tot[cell_idx].store(0);

        tv_object.u_plane_tx[cell_idx].store(0);
        tv_object.u_plane_tx_tot[cell_idx].store(0);

        for(int i = 0; i < ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID; ++i)
        {
            tv_object.section_rx_counters[cell_idx][i].store(0);
            tv_object.prb_rx_counters[cell_idx][i].store(0);
            tv_object.prb_rx_counter_frame[cell_idx][i].store(PRB_RX_FRAME_UNSET);
        }
    }

    auto pg_sz = sysconf(_SC_PAGESIZE);
    if(pg_sz == -1)
    {
        do_throw(sb() << "failed to get page size");
    }
    void * fh_mem = aerial_fh::allocate_memory(sizeof(uint32_t) * MAX_NUM_PRBS_PER_SYMBOL * PRB_SIZE_16F, pg_sz);
    if(fh_mem == nullptr)
    {
        do_throw(sb() << "aerial_fh::allocate_memory failure ");
    }
    tv_object.blank_prbs.reset(memset(fh_mem, 0, sizeof(uint32_t) * MAX_NUM_PRBS_PER_SYMBOL * PRB_SIZE_16F));
}

void RU_Emulator::setup_dl_slots()
{
    pdsch_object.channel_type = dl_channel::PDSCH;
    pbch_object.channel_type = dl_channel::PBCH;
    pdcch_ul_object.channel_type = dl_channel::PDCCH_UL;
    pdcch_dl_object.channel_type = dl_channel::PDCCH_DL;
    csirs_object.channel_type = dl_channel::CSI_RS;
    bfw_dl_object.channel_type = dl_channel::BFW_DL;
    bfw_ul_object.channel_type = dl_channel::BFW_UL;

    pdsch_object.nrsim_ch_type = nrsim_tv_type::PDSCH;
    pbch_object.nrsim_ch_type = nrsim_tv_type::SSB;
    pdcch_ul_object.nrsim_ch_type = nrsim_tv_type::PDCCH;
    pdcch_dl_object.nrsim_ch_type = nrsim_tv_type::PDCCH;
    csirs_object.nrsim_ch_type = nrsim_tv_type::CSI_RS;
    bfw_dl_object.nrsim_ch_type = nrsim_tv_type::BFW;
    bfw_ul_object.nrsim_ch_type = nrsim_tv_type::BFW;

    for(int i = 0; i < STATS_MAX_BINS; ++i)
    {
        timing_bins[i].store(0);
    }

    setup_dl_counters(pdsch_object);
    setup_dl_counters(pbch_object);
    setup_dl_counters(pdcch_ul_object);
    setup_dl_counters(pdcch_dl_object);
    setup_dl_counters(csirs_object);
    setup_dl_counters(bfw_dl_object);
    setup_dl_counters(bfw_ul_object);
}

void RU_Emulator::setup_dl_counters(struct dl_tv_object& tv_object)
{
    setup_dl_receive_bytes(tv_object);
    setup_dl_atomic_counters(tv_object);
}

void RU_Emulator::setup_dl_atomic_counters(struct dl_tv_object& tv_object)
{
    for(int cell_idx = 0; cell_idx < MAX_CELLS_PER_SLOT; ++cell_idx)
    {

        for(int slot_idx = 0; slot_idx < tv_object.launch_pattern.size(); ++slot_idx)
        {
            tv_object.invalid_flag[cell_idx].emplace_back(false);
        }

        tv_object.throughput_counters[cell_idx].store(0);
        tv_object.throughput_slot_counters[cell_idx].store(0);
        tv_object.good_slot_counters[cell_idx].store(0);
        tv_object.error_slot_counters[cell_idx].store(0);
        tv_object.total_slot_counters[cell_idx].store(0);

        for(int i = 0; i < ALL_PACKET_TYPES; ++i)
        {
            slot_count[i][cell_idx].store(0);
        }
    }
}

void RU_Emulator::setup_dl_receive_bytes(struct dl_tv_object& tv_object)
{
    for(int slot_idx = 0; slot_idx < tv_object.launch_pattern.size(); ++slot_idx)
    {
        tv_object.received_res.emplace_back(std::vector<std::vector<uint32_t>>());
        if(tv_object.launch_pattern[slot_idx].empty())
        {
            continue;
        }

        for(int cell_idx = 0; cell_idx < opt_num_cells; ++cell_idx)
        {
            tv_object.received_res[slot_idx].emplace_back(std::vector<uint32_t>());
            for(int core_index = 0; core_index < dl_cores_per_cell; ++core_index)
            {
                tv_object.received_res[slot_idx][cell_idx].emplace_back(0);
            }
        }
    }
}

void RU_Emulator::setup_rings()
{
    char ring_name[512];
#ifdef STANDALONE
    for(int i = 0; i < opt_num_cells; ++i)
    {
        snprintf(ring_name, sizeof(ring_name), "standalone_c_plane_ring_%02d", i);
        aerial_fh::RingBufferHandle ring{};
        aerial_fh::RingBufferInfo ring_info;
        ring_info.count = RE_RING_ELEMS;
        ring_info.socket_id = dpdk.socket_id;
        ring_info.multi_producer = false;
        ring_info.multi_consumer = true;
        ring_info.name = (const char *)ring_name;

        int ret = aerial_fh::ring_create(fronthaul,&ring_info,&ring);
        if(ret)
        {
            do_throw(sb() << "fh_ring_create dl_ring error\n");
        }

        re_cons("Set up {} with {} elems {}", ring_name, aerial_fh::ring_free_count(ring), ring);

        standalone_c_plane_rings.push_back(ring);
        if(standalone_c_plane_rings[i] == nullptr)
        {
            do_throw(sb() << "fh_ring_create standalone ring[" << i << "] error\n");
        }
    }
#endif
}
