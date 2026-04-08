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

// This file provides a thin wrapper around RU_Emulator to isolate it from
// type conflicts with the test bench headers

#include "ru_emulator_wrapper.hpp"
#include "ru_emulator.hpp"

// Helper to cast void* to RU_Emulator*
static inline RU_Emulator* get_ru_emulator(void* ru_emulator) {
    return static_cast<RU_Emulator*>(ru_emulator);
}

// This is a static global record of the PRBs seen so far for MMIMO verification. 
static FssPdschPrbSeenArray fss_pdsch_prb_seen{}; 

// Create/destroy wrappers
void* create_ru_emulator() 
{
    return new RU_Emulator();
}

void destroy_ru_emulator(void* ru_emulator) 
{
    delete get_ru_emulator(ru_emulator);
}

// RU_Emulator method wrappers
void ru_emulator_init(void* ru_emulator, int argc, char** argv) 
{
    get_ru_emulator(ru_emulator)->init_minimal(argc, argv);
    get_ru_emulator(ru_emulator)->verify_and_apply_configs();
    get_ru_emulator(ru_emulator)->setup_slots();
    get_ru_emulator(ru_emulator)->load_tvs();
}

int ru_emulator_start(void* ru_emulator) 
{
    return get_ru_emulator(ru_emulator)->start();
}

int ru_emulator_finalize(void* ru_emulator) 
{
    return get_ru_emulator(ru_emulator)->finalize_dlc_tb();
}

// Configuration method wrappers
void ru_emulator_set_default_configs(void* ru_emulator) 
{
    get_ru_emulator(ru_emulator)->set_default_configs();
}

void ru_emulator_parse_yaml(void* ru_emulator, const std::string& yaml_file) 
{
    get_ru_emulator(ru_emulator)->parse_yaml(yaml_file);
}

void ru_emulator_verify_and_apply_configs(void* ru_emulator) 
{
    get_ru_emulator(ru_emulator)->verify_and_apply_configs();
}

void ru_emulator_print_configs(void* ru_emulator) 
{
    get_ru_emulator(ru_emulator)->print_configs();
}

// TV and setup method wrappers
void ru_emulator_load_tvs(void* ru_emulator) 
{
    get_ru_emulator(ru_emulator)->load_tvs();
}

void ru_emulator_setup_slots(void* ru_emulator) 
{
    get_ru_emulator(ru_emulator)->setup_slots();
}

void ru_emulator_add_flows(void* ru_emulator) 
{
    get_ru_emulator(ru_emulator)->add_flows();
}

void ru_emulator_setup_rings(void* ru_emulator) 
{
    get_ru_emulator(ru_emulator)->setup_rings();
}

void ru_emulator_oam_init(void* ru_emulator) 
{
    get_ru_emulator(ru_emulator)->oam_init();
}

void ru_emulator_verify_dl_cplane_content(void* ru_emulator, uint8_t *mbuf_payload, size_t buffer_length, int cell_index)
{
    oran_c_plane_info_t c_plane_info{};
    aerial_fh::MsgReceiveInfo msg_info{};
    msg_info.buffer_length = buffer_length;

    // First parse the C-Plane message to construct the c_plane_info
    get_ru_emulator(ru_emulator)->parse_c_plane(c_plane_info, 0 /* nb_rx */, 0 /* index_rx */, 0 /* rte_rx_time */, mbuf_payload, buffer_length, cell_index);
    get_ru_emulator(ru_emulator)->verify_dl_cplane_content(c_plane_info, cell_index, mbuf_payload, msg_info, fss_pdsch_prb_seen);

}

void ru_emulator_construct_bfw(void* ru_emulator, int sfn, int slot, int cell_idx, std::vector<uint8_t*> &out) 
{

    int lp_slot = sfn * 20 + slot; 

    auto &tv_object = get_ru_emulator(ru_emulator)->get_bfw_dl_obj(); 

    if (lp_slot >= static_cast<int>(tv_object.launch_pattern.size()))
        return;

    auto &map = tv_object.launch_pattern[lp_slot];

    if (map.find(cell_idx) == map.end())
        return;

    auto tv_idx = map.at(cell_idx); 

    if (tv_idx >= tv_object.tv_info.size())
        return;

    auto &tv_info = tv_object.tv_info[tv_idx];

    for (int i = 0; i < tv_info.bfw_infos.size() /* # BFW PDUs */; ++i) {

        int sample_index = i; 
        for(int k = 0; k < tv_idx; ++k) {
            sample_index += tv_object.tv_info[k].bfw_infos.size();
        }

        auto &bfw_info = tv_info.bfw_infos[i];
        // The buff points to the parsed memory of BFW weights parsed by RUE
        // that is going to be reused for generating BFW in CP Packets. 
        auto buff = (uint8_t *) tv_object.qams[bfw_info.compressBitWidth][sample_index].data.get(); 
        out.push_back(buff); 
    }
}

void ru_emulator_get_total_slt_counters (void *ru_emulator, int cell_idx, ru_emulator_total_slot_counters_t &counters)
{

    RU_Emulator *rue = get_ru_emulator(ru_emulator); 
    
    counters.pdsch = rue->get_pdsch_object().total_slot_counters.at(cell_idx).load();
    counters.pdcch_dl = rue->get_pdcch_dl_object().total_slot_counters.at(cell_idx).load();
    counters.pdcch_ul = rue->get_pdcch_ul_object().total_slot_counters.at(cell_idx).load();
    counters.pbch = rue->get_pbch_object().total_slot_counters.at(cell_idx).load();
    counters.csi_rs = rue->get_csirs_object().total_slot_counters.at(cell_idx).load();
    counters.bfw_dl = rue->get_bfw_dl_object().total_slot_counters.at(cell_idx).load();
    
    return; 
}

void ru_emulator_get_cplane_err_sections (void *ru_emulator, int cell_idx, uint64_t &err_sections)
{
    err_sections = get_ru_emulator(ru_emulator)->get_error_dl_section_count(cell_idx); 
}

void ru_emulator_get_cplane_tot_sections (void *ru_emulator, int cell_idx, uint64_t &tot_sections)
{
    tot_sections = get_ru_emulator(ru_emulator)->get_total_dl_section_count(cell_idx); 
}


