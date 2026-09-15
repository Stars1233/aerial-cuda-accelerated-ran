/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "slot_command/slot_command.hpp"
#include "aerial/casts/casts.hpp"

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
    printf("[RU] C-Plane testbench finalizing...\n");
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

void ru_emulator_verify_ul_cplane_content(void* ru_emulator, uint8_t *mbuf_payload, size_t buffer_length, int cell_index)
{
    oran_c_plane_info_t c_plane_info{};
    slot_tx_info slot_tx{};

    get_ru_emulator(ru_emulator)->parse_c_plane(c_plane_info, 0 /* nb_rx */, 0 /* index_rx */, 0 /* rte_rx_time */, mbuf_payload, buffer_length, cell_index);
    get_ru_emulator(ru_emulator)->verify_ul_cplane_content(c_plane_info, cell_index, mbuf_payload, slot_tx);
}

// Shared BFW PDU collector for a DL/UL bfw tv_object. Maps (sfn, slot, cell) to
// the launch-pattern TV entry, then emits one ru_emulator_bfw_pdu_t per BFW PDU
// with the host IQ pointer (from the parsed qams) and the metadata needed to
// build a framework DynamicBfwCompletionRecord.
static void construct_bfw_pdus(const dl_tv_object& tv_object, int sfn, int slot, int cell_idx,
                               std::vector<ru_emulator_bfw_pdu_t>& out)
{
    const int lp_slot = sfn * 20 + slot;
    if (lp_slot < 0 || lp_slot >= static_cast<int>(tv_object.launch_pattern.size())) {
        return;
    }

    const auto& map = tv_object.launch_pattern[static_cast<size_t>(lp_slot)];
    const auto it = map.find(cell_idx);
    if (it == map.end()) {
        return;
    }

    const int tv_idx = it->second;
    if (tv_idx < 0 || tv_idx >= static_cast<int>(tv_object.tv_info.size())) {
        return;
    }
    const auto& tv_info = tv_object.tv_info[static_cast<size_t>(tv_idx)];

    // Sample index into the contiguous qams layout = count of BFW PDUs in all
    // prior TVs plus the local index (matches the RU emulator parse order).
    size_t base_sample = 0;
    for (int k = 0; k < tv_idx; ++k) {
        base_sample += tv_object.tv_info[static_cast<size_t>(k)].bfw_infos.size();
    }

    for (size_t i = 0; i < tv_info.bfw_infos.size(); ++i) {
        const auto& bfw = tv_info.bfw_infos[i];
        const auto bitwidth = static_cast<size_t>(bfw.compressBitWidth);
        const size_t sample_index = base_sample + i;
        if (bitwidth >= tv_object.qams.size() || sample_index >= tv_object.qams[bitwidth].size()) {
            continue;
        }

        ru_emulator_bfw_pdu_t pdu{};
        pdu.host_bfws = aerial::casts::assume_cast<uint8_t>(tv_object.qams[bitwidth][sample_index].data.get());
        pdu.rb_start = static_cast<uint16_t>(bfw.rbStart);
        pdu.rb_size = static_cast<uint16_t>(bfw.rbSize);
        pdu.num_prgs = static_cast<uint16_t>(bfw.numPRGs);
        pdu.prg_size = static_cast<uint16_t>(bfw.prgSize);
        pdu.bfw_iq_bitwidth = static_cast<uint8_t>(bfw.compressBitWidth);
        pdu.num_antenna_ports = static_cast<uint16_t>(slot_command_api::NUM_GNB_TX_RX_ANT_PORTS);
        out.push_back(pdu);
    }
}

void ru_emulator_construct_bfw_dl(void* ru_emulator, int sfn, int slot, int cell_idx,
                                  std::vector<ru_emulator_bfw_pdu_t>& out)
{
    construct_bfw_pdus(get_ru_emulator(ru_emulator)->get_bfw_dl_object(), sfn, slot, cell_idx, out);
}

void ru_emulator_construct_bfw_ul(void* ru_emulator, int sfn, int slot, int cell_idx,
                                  std::vector<ru_emulator_bfw_pdu_t>& out)
{
    construct_bfw_pdus(get_ru_emulator(ru_emulator)->get_bfw_ul_object(), sfn, slot, cell_idx, out);
}

void ru_emulator_get_total_slt_counters (void *ru_emulator, int cell_idx, ru_emulator_total_slot_counters_t &counters)
{

    RU_Emulator *rue = get_ru_emulator(ru_emulator); 
    const auto ci = static_cast<size_t>(cell_idx);

    counters.pdsch = rue->get_pdsch_object().total_slot_counters.at(ci).load();
    counters.pdcch_dl = rue->get_pdcch_dl_object().total_slot_counters.at(ci).load();
    counters.pdcch_ul = rue->get_pdcch_ul_object().total_slot_counters.at(ci).load();
    counters.pbch = rue->get_pbch_object().total_slot_counters.at(ci).load();
    counters.csi_rs = rue->get_csirs_object().total_slot_counters.at(ci).load();
    counters.bfw_dl = rue->get_bfw_dl_object().total_slot_counters.at(ci).load();
    counters.pusch = rue->get_pusch_object().total_slot_counters.at(ci).load();
    counters.pucch = rue->get_pucch_object().total_slot_counters.at(ci).load();
    counters.prach = rue->get_prach_object().total_slot_counters.at(ci).load();
    counters.srs = rue->get_srs_object().total_slot_counters.at(ci).load();
    counters.bfw_ul = rue->get_bfw_ul_object().total_slot_counters.at(ci).load();
}

void ru_emulator_get_cplane_err_sections (void *ru_emulator, int cell_idx, uint64_t &err_sections)
{
    err_sections = get_ru_emulator(ru_emulator)->get_error_dl_section_count(cell_idx); 
}

void ru_emulator_get_cplane_dl_sections (void *ru_emulator, int cell_idx, uint64_t &dl_sections)
{
    dl_sections = get_ru_emulator(ru_emulator)->get_total_dl_section_count(cell_idx);
}

void ru_emulator_get_cplane_ul_sections (void *ru_emulator, int cell_idx, uint64_t &ul_sections)
{
    ul_sections = get_ru_emulator(ru_emulator)->get_total_ul_section_count(cell_idx);
}

void ru_emulator_set_pdcch_single_port(void* ru_emulator)
{
    // Framework optimization: PDCCH is sent on a single eAxC port.
    // Set numFlows=1 in all PDCCH TV entries so the RU emulator's
    // completion threshold (numPrb * min(numFlows, eAxC_DL.size()))
    // matches the single-port output.
    auto& pdcch_dl = const_cast<dl_tv_object&>(get_ru_emulator(ru_emulator)->get_pdcch_dl_object());
    for (auto& tv : pdcch_dl.tv_info) {
        tv.numFlows = 1;
    }
    auto& pdcch_ul = const_cast<dl_tv_object&>(get_ru_emulator(ru_emulator)->get_pdcch_ul_object());
    for (auto& tv : pdcch_ul.tv_info) {
        tv.numFlows = 1;
    }
}

